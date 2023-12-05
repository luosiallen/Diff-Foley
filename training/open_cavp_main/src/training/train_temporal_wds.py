import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_cast_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # images, texts = batch
        # images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        # texts = texts.to(device=device, non_blocking=True)

        # # Revise:
        spec, video, start_bias_index, end_bias_index = batch

        video = video.to(device, dtype=cast_dtype, non_blocking=True)
        spec = spec.to(device, dtype=cast_dtype, non_blocking=True)

        start_bias_index = start_bias_index.to(device, dtype=torch.long)
        end_bias_index = end_bias_index.to(device, dtype=torch.long)


        # video = batch['video'].to(device, dtype=cast_dtype, non_blocking=True)
        # spec = batch['spec'].to(device, dtype=cast_dtype, non_blocking=True)


        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                model_out = model(video, spec)
                logit_scale = model_out["logit_scale"]
                losses = loss(**model_out, start_bias_index=start_bias_index, end_bias_index=end_bias_index,output_dict=True)
                total_loss = losses['total_loss']
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss
            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts, output_dict=True)
                    logit_scale = model_out.pop("logit_scale")
                    for key, val in accum_features:
                        accumulated = accum_features[key]
                        accumulated = accumulated[:j] +  [model_out[key]] + accumulated[j + 1:]
                    losses = loss(**accumulated, logit_scale=logit_scale, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(video)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.accum_freq * args.batch_size * args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_loss_semantic = 0.0
        cumulative_loss_temporal = 0.0
        cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []
        all_video_mean_features, all_spec_mean_features = [], []
        all_video_temporal_features, all_spec_temporal_features = [], []
        all_start_bias_index, all_end_bias_index = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # images, texts = batch
                # images = batch['video']
                # texts = batch['spec']
                texts, images, start_bias_index, end_bias_index = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                start_bias_index = start_bias_index.to(device, dtype=torch.long)
                end_bias_index = end_bias_index.to(device, dtype=torch.long)

                with autocast():
                    model_out = model(images, texts, output_dict=True)

                    video_temporal_features, video_mean_features = model_out["video_temporal_features"], model_out["video_mean_features"]
                    spec_temporal_features, spec_mean_features = model_out["spec_temporal_features"], model_out["spec_mean_features"]
                    logit_scale = model_out["logit_scale"]

                    all_video_mean_features.append(video_mean_features.cpu())
                    all_spec_mean_features.append(spec_mean_features.cpu())
                    all_video_temporal_features.append(video_temporal_features.cpu())
                    all_spec_temporal_features.append(spec_temporal_features.cpu())
                    all_start_bias_index.append(start_bias_index.cpu())
                    all_end_bias_index.append(end_bias_index.cpu())

                    logit_scale = logit_scale.mean()

                    """Semantic Contrast Loss"""
                    logits_per_video_semantic = logit_scale * video_mean_features @ spec_mean_features.t()
                    logits_per_spec_semantic = logits_per_video_semantic.t()

                    batch_size = images.shape[0]
                    labels_semantic = torch.arange(batch_size, device=device).long()
                    semantic_contrast_loss = (F.cross_entropy(logits_per_video_semantic, labels_semantic) + F.cross_entropy(logits_per_spec_semantic, labels_semantic)) / 2

                    """Temporal Contrast Loss"""
                    temporal_len = video_temporal_features.shape[1]
                    logits_per_video_temporal = logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)
                    logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
                    
                    # labels_temporal = torch.arange(temporal_len, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

                    # logits_per_video_temporal = logits_per_video_temporal.reshape(batch_size * temporal_len, temporal_len)
                    # logits_per_spec_temporal = logits_per_spec_temporal.reshape(batch_size * temporal_len, temporal_len)
                    # labels_temporal = labels_temporal.reshape(batch_size * temporal_len)

                    # temporal_contrast_loss = (F.cross_entropy(logits_per_video_temporal, labels_temporal) + F.cross_entropy(logits_per_spec_temporal, labels_temporal)) / 2
                    truncate_len = (end_bias_index - start_bias_index)[:, 0] + 1
                    _, T, _ = logits_per_video_temporal.shape              # bs x T x T
                    device = logits_per_video_temporal.device

                    # Target Label &  Mask Label 
                    target_video2spec = []                                  # B x T
                    mask_video2spec = []                                    # B x T

                    target_spec2video = []
                    mask_spec2video = []

                    bs = start_bias_index.shape[0]

                    for i in range(bs):
                        if start_bias_index[i][0] != 0:      # Left Down Diagonal Matrix:
                            zero_pad_num = T - int(truncate_len[i])
                            # Target, Mask: Video2Spec
                            target_video2spec.extend([torch.zeros(zero_pad_num), torch.arange(T - zero_pad_num)])
                            mask_video2spec.extend([torch.zeros(zero_pad_num), torch.ones(T - zero_pad_num)])
                            # Target, Mask: Spec2Video
                            target_spec2video.extend([torch.arange(T - zero_pad_num) + zero_pad_num, torch.zeros(zero_pad_num)])
                            mask_spec2video.extend([torch.ones(T - zero_pad_num), torch.zeros(zero_pad_num)])

                        else:                           # Right Up Diagonal Matrix:
                            zero_pad_num = T - int(truncate_len[i])
                            # Target, Mask: Video2Spec
                            target_video2spec.extend([torch.arange(T - zero_pad_num) + zero_pad_num, torch.zeros(zero_pad_num)])
                            mask_video2spec.extend([torch.ones(T - zero_pad_num), torch.zeros(zero_pad_num)])
                            # Target, Mask: Spec2Video
                            target_spec2video.extend([torch.zeros(zero_pad_num), torch.arange(T - zero_pad_num)])
                            mask_spec2video.extend([torch.zeros(zero_pad_num), torch.ones(T - zero_pad_num)])

                    # Video2Spec:   B x T
                    target_video2spec = torch.cat(target_video2spec).to(torch.long).reshape(bs, T).to(device)
                    mask_video2spec = torch.cat(mask_video2spec).reshape(bs, T).to(device)
                    mask_video2spec_sum = mask_video2spec.sum(dim=1) 

                    # Spec2Video:   B x T
                    target_spec2video = torch.cat(target_spec2video).to(torch.long).reshape(bs, T).to(device)
                    mask_spec2video = torch.cat(mask_spec2video).reshape(bs, T).to(device)
                    mask_spec2video_sum = mask_spec2video.sum(dim=1) 

                    # Video ->  Spec Loss:
                    loss_video2spec = F.cross_entropy(logits_per_video_temporal.permute(0,2,1), target_video2spec, reduction='none')       # B x Video
                    loss_mask_video2spec = ((loss_video2spec * mask_video2spec).sum(dim=1) / mask_video2spec_sum).mean()

                    # Spec ->  Video Loss:
                    loss_spec2video = F.cross_entropy(logits_per_spec_temporal.permute(0,2,1), target_spec2video, reduction='none')        # B x Spec
                    loss_mask_spec2video = ((loss_spec2video * mask_spec2video).sum(dim=1) / mask_spec2video_sum).mean()

                    # Temporal Contrast Loss:
                    temporal_contrast_loss = (loss_mask_spec2video + loss_mask_video2spec) / 2

                    """Others"""
                    total_loss = temporal_contrast_loss * args.temporal_mix_weight + semantic_contrast_loss 
                    gen_loss = maybe_compute_generative_loss(model_out)
                
                cumulative_loss += total_loss * batch_size
                cumulative_loss_semantic += semantic_contrast_loss * batch_size
                cumulative_loss_temporal += temporal_contrast_loss * batch_size
                num_samples += batch_size

                if is_master(args) and (i & 10) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Total Loss: {cumulative_loss / num_samples:.6f}\t"
                        f"Clip Semantic Loss: {cumulative_loss_semantic / num_samples:.6f}\t"
                        f"Clip Temporal Loss: {cumulative_loss_temporal / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics_temporal_bias(
                video_mean_features = torch.cat(all_video_mean_features),
                spec_mean_features = torch.cat(all_spec_mean_features),
                video_temporal_features = torch.cat(all_video_temporal_features),
                spec_temporal_features = torch.cat(all_spec_temporal_features),
                start_bias_index = torch.cat(all_start_bias_index),
                end_bias_index = torch.cat(all_end_bias_index),
                logit_scale=logit_scale.cpu(),
            )

            loss_total = cumulative_loss / num_samples
            loss_semantic = cumulative_loss_semantic / num_samples
            loss_temporal = cumulative_loss_temporal / num_samples

            metrics.update(
                {**val_metrics, "clip_val_loss": loss_total.item(), "clip_val_temporal":loss_temporal.item(), "clip_val_semantic": loss_semantic.item(), "epoch": epoch, "num_samples": num_samples}
            )

            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def get_clip_metrics_temporal(video_mean_features, spec_mean_features, video_temporal_features, spec_temporal_features, logit_scale):
    metrics = {}

    # Semantic: B x B
    logits_per_video_semantic = (logit_scale * video_mean_features @ spec_mean_features.t()).detach().cpu()
    logits_per_spec_semantic = logits_per_video_semantic.t().detach().cpu()

    # Temporal: B x T x T
    logits_per_video_temporal = (logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)).detach().cpu()
    logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
    
    logits_semantic = {"semantic_video_to_spec": logits_per_video_semantic, "semantic_spec_to_video": logits_per_spec_semantic}
    logits_temporal = {"temporal_video_to_spec": logits_per_video_temporal, "temporal_spec_to_video": logits_per_spec_temporal}


    bs = len(video_mean_features)
    ground_truth_semantic = torch.arange(bs).view(-1, 1)
    ground_truth_temporal = torch.arange(video_temporal_features.shape[1]).unsqueeze(0).repeat(bs, 1).unsqueeze(2)

    # Semantic:
    for name, logit in logits_semantic.items():
        ranking = torch.argsort(logit, descending=True)            # B x B
        preds = torch.where(ranking == ground_truth_semantic)[1]   # 
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)


    # Temporal:
    for name, logit in logits_temporal.items():
        ranking = torch.argsort(logit, descending=True)             # B x T x T
        preds = torch.where(ranking == ground_truth_temporal)[2]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)


    return metrics
    


def get_clip_metrics_temporal_bias(video_mean_features, spec_mean_features, video_temporal_features, spec_temporal_features, start_bias_index, end_bias_index, logit_scale):
    metrics = {}

    # Semantic: B x B
    logits_per_video_semantic = (logit_scale * video_mean_features @ spec_mean_features.t()).detach().cpu()
    logits_per_spec_semantic = logits_per_video_semantic.t().detach().cpu()

    # Temporal: B x T x T
    logits_per_video_temporal = (logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)).detach().cpu()
    logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
    
    logits_semantic = {"semantic_video_to_spec": logits_per_video_semantic, "semantic_spec_to_video": logits_per_spec_semantic}
    # logits_temporal = {"temporal_video_to_spec": logits_per_video_temporal, "temporal_spec_to_video": logits_per_spec_temporal}


    bs = len(video_mean_features)
    ground_truth_semantic = torch.arange(bs).view(-1, 1)
    # ground_truth_temporal = torch.arange(video_temporal_features.shape[1]).unsqueeze(0).repeat(bs, 1).unsqueeze(2)    # B x T x 1

    # Semantic:
    for name, logit in logits_semantic.items():
        ranking = torch.argsort(logit, descending=True)            # B x B
        preds = torch.where(ranking == ground_truth_semantic)[1]   # 
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    # Temporal Bias:
    

    # Get: Target Label &  Mask Label 
    truncate_len = (end_bias_index - start_bias_index)[:, 0] + 1
    _, T, _ = logits_per_video_temporal.shape               # bs x T x T
    device = logits_per_video_temporal.device
    target_video2spec = []                                  # B x T
    mask_video2spec = []                                    # B x T

    target_spec2video = []
    mask_spec2video = []
    for i in range(bs):
        if start_bias_index[i][0] != 0:      # Left Down Diagonal Matrix:
            zero_pad_num = T - int(truncate_len[i])
            # Target, Mask: Video2Spec
            target_video2spec.extend([torch.zeros(zero_pad_num), torch.arange(T - zero_pad_num)])
            mask_video2spec.extend([torch.zeros(zero_pad_num), torch.ones(T - zero_pad_num)])
            # Target, Mask: Spec2Video
            target_spec2video.extend([torch.arange(T - zero_pad_num) + zero_pad_num, torch.zeros(zero_pad_num)])
            mask_spec2video.extend([torch.ones(T - zero_pad_num), torch.zeros(zero_pad_num)])

        else:                           # Right Up Diagonal Matrix:
            zero_pad_num = T - int(truncate_len[i])
            # Target, Mask: Video2Spec
            target_video2spec.extend([torch.arange(T - zero_pad_num) + zero_pad_num, torch.zeros(zero_pad_num)])
            mask_video2spec.extend([torch.ones(T - zero_pad_num), torch.zeros(zero_pad_num)])
            # Target, Mask: Spec2Video
            target_spec2video.extend([torch.zeros(zero_pad_num), torch.arange(T - zero_pad_num)])
            mask_spec2video.extend([torch.zeros(zero_pad_num), torch.ones(T - zero_pad_num)])

    # Target Video2Spec: B x T
    target_video2spec = torch.cat(target_video2spec).to(torch.long).reshape(bs, T).to(device)
    mask_video2spec = torch.cat(mask_video2spec).reshape(bs, T).to(device)
    mask_video2spec_sum = mask_video2spec.sum(dim=1) 

    # Target Spec2Video: B x T
    target_spec2video = torch.cat(target_spec2video).to(torch.long).reshape(bs, T).to(device)
    mask_spec2video = torch.cat(mask_spec2video).reshape(bs, T).to(device)
    mask_spec2video_sum = mask_spec2video.sum(dim=1) 


    # Video -> Spec:
    name = "temporal_bias_video_to_spec"
    logit = logits_per_video_temporal
    ranking = torch.argsort(logit, descending=True)                                     # B x T x T
    preds = torch.where(ranking == target_video2spec.unsqueeze(2))[2].reshape(bs, T)    # B x T
    preds_mean_rank = ((preds * mask_video2spec).sum(dim=1) / mask_video2spec_sum).mean().detach().cpu().numpy() + 1
    metrics[f"{name}_mean_rank"] = float(preds_mean_rank) + 1

    for k in [1,3,5]:
        preds_r_k = (((preds < k) * mask_video2spec).sum(dim=1) / mask_video2spec_sum).mean().detach().cpu().numpy()
        metrics[f"{name}_R@{k}"] = float(preds_r_k)


    # Spec -> Video:
    name = "temporal_bias_spec_to_video"
    logit = logits_per_spec_temporal
    ranking = torch.argsort(logit, descending=True)                                     # B x T x T
    preds = torch.where(ranking == target_spec2video.unsqueeze(2))[2].reshape(bs, T)    # B x T
    preds_mean_rank = ((preds * mask_spec2video).sum(dim=1) / mask_spec2video_sum).mean().detach().cpu().numpy() + 1
    metrics[f"{name}_mean_rank"] = float(preds_mean_rank) + 1

    for k in [1,3,5]:
        preds_r_k = (((preds < k) * mask_spec2video).sum(dim=1) / mask_spec2video_sum).mean().detach().cpu().numpy()
        metrics[f"{name}_R@{k}"] = float(preds_r_k) 


    return metrics




def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
