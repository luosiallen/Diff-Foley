import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss



class ClipLoss_Temporal_Semantic(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
            temporal_mix_weight = 0.5,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temporal_mix_weight = temporal_mix_weight

        # cache state
        self.prev_num_logits_semantic = 0
        self.prev_num_logits_temporal = 0
        self.labels_semantic = {}
        self.labels_temporal = {}

    def forward(self, video_temporal_features, video_mean_features, spec_temporal_features, spec_mean_features, logit_scale, output_dict=False):
        # device = image_features.device
        # if self.world_size > 1:
        #     all_image_features, all_text_features = gather_features(
        #         image_features, text_features,
        #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        #     if self.local_loss:
        #         logits_per_image = logit_scale * image_features @ all_text_features.T
        #         logits_per_text = logit_scale * text_features @ all_image_features.T
        #     else:
        #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
        #         logits_per_text = logits_per_image.T
        # else:
        #     logits_per_image = logit_scale * image_features @ text_features.T
        #     logits_per_text = logit_scale * text_features @ image_features.T


        # Semantic Contrastive Loss: B x C
        device = video_mean_features.device
        if self.world_size > 1:
            all_video_mean_features, all_spec_mean_features = gather_features(
                video_mean_features, spec_mean_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            if self.local_loss:
                logits_per_video_semantic = logit_scale * video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = logit_scale * spec_mean_features @ all_video_mean_features.T
            else:
                logits_per_video_semantic = logit_scale * all_video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = logits_per_video_semantic.T
        else:
            logits_per_video_semantic = logit_scale * video_mean_features @ spec_mean_features.T
            logits_per_spec_semantic = logit_scale * spec_mean_features @ video_mean_features.T


        # Temporal Contrastive Loss: B x T x C
        if self.world_size > 1:
            all_video_temporal_features, all_spec_temporal_features = gather_features(
                video_temporal_features, spec_temporal_features, 
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )

            if self.local_loss:
                logits_per_video_temporal = logit_scale * video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = logit_scale * spec_temporal_features @ all_video_temporal_features.permute(0, 2, 1)
            else:
                logits_per_video_temporal = logit_scale * all_video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
        else:
            logits_per_video_temporal = logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)
            logits_per_spec_temporal = logit_scale * spec_temporal_features @ video_temporal_features.permute(0, 2, 1)
                
        

        # Calculated Loss: (Semantic Contrastive Loss & Temporal Contrastive Loss)

        # Semantic Contrast Loss:
        num_logits_semantic = logits_per_video_semantic.shape[0]
        if self.prev_num_logits_semantic != num_logits_semantic or device not in self.labels_semantic:
            labels_semantic = torch.arange(num_logits_semantic, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels_semantic = labels_semantic + num_logits_semantic * self.rank
            if self.cache_labels:
                self.labels_semantic[device] = labels_semantic
                self.prev_num_logits_semantic = num_logits_semantic
        else:
            labels_semantic = self.labels_semantic[device]

        semantic_contrast_loss = (F.cross_entropy(logits_per_video_semantic, labels_semantic) + F.cross_entropy(logits_per_spec_semantic, labels_semantic)) / 2


        # Temporal Contrast Loss:
        bs, num_logits_temporal , _= logits_per_video_temporal.shape
        if self.prev_num_logits_temporal != num_logits_temporal or device not in self.labels_temporal:
            labels_temporal = torch.arange(num_logits_temporal, device=device, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
            # No Need Local Loss:
            if self.cache_labels:
                self.labels_temporal[device] = labels_temporal
                self.prev_num_logits_temporal = num_logits_temporal
        else:
            labels_temporal = self.labels_temporal[device]
        
        logits_per_video_temporal = logits_per_video_temporal.reshape(bs * num_logits_temporal, num_logits_temporal)
        logits_per_spec_temporal = logits_per_spec_temporal.reshape(bs * num_logits_temporal, num_logits_temporal)
        labels_temporal = labels_temporal.reshape(bs * num_logits_temporal)

        temporal_contrast_loss = (F.cross_entropy(logits_per_video_temporal, labels_temporal) + F.cross_entropy(logits_per_spec_temporal, labels_temporal)) / 2

        total_loss = self.temporal_mix_weight * temporal_contrast_loss + semantic_contrast_loss

        return {"semantic_contrast_loss": semantic_contrast_loss, "temporal_contrast_loss": temporal_contrast_loss, "total_loss": total_loss, "temp_mix_weight": torch.tensor(self.temporal_mix_weight)}






# Temporal Bias:
class ClipLoss_Temporal_Semantic_Bias(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
            temporal_mix_weight = 0.5,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temporal_mix_weight = temporal_mix_weight

        # cache state
        self.prev_num_logits_semantic = 0
        self.prev_num_logits_temporal = 0
        self.labels_semantic = {}
        self.labels_temporal = {}

    def forward(self, video_temporal_features, video_mean_features, spec_temporal_features, spec_mean_features, logit_scale, start_bias_index, end_bias_index,  output_dict=False):
        # Semantic Contrastive Loss: B x C
        device = video_mean_features.device
        if self.world_size > 1:
            all_video_mean_features, all_spec_mean_features = gather_features(
                video_mean_features, spec_mean_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            if self.local_loss:
                logits_per_video_semantic = logit_scale * video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = logit_scale * spec_mean_features @ all_video_mean_features.T
            else:
                logits_per_video_semantic = logit_scale * all_video_mean_features @ all_spec_mean_features.T
                logits_per_spec_semantic = logits_per_video_semantic.T
        else:
            logits_per_video_semantic = logit_scale * video_mean_features @ spec_mean_features.T
            logits_per_spec_semantic = logit_scale * spec_mean_features @ video_mean_features.T

        # Temporal Contrastive Loss: B x T x C
        if self.world_size > 1:
            all_video_temporal_features, all_spec_temporal_features = gather_features(
                video_temporal_features, spec_temporal_features, 
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )

            start_bias_index, end_bias_index = gather_features(
                start_bias_index, end_bias_index,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )

            if self.local_loss:
                logits_per_video_temporal = logit_scale * video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = logit_scale * spec_temporal_features @ all_video_temporal_features.permute(0, 2, 1)
            else:
                logits_per_video_temporal = logit_scale * all_video_temporal_features @ all_spec_temporal_features.permute(0, 2, 1)
                logits_per_spec_temporal = logits_per_video_temporal.permute(0, 2, 1)
        else:
            logits_per_video_temporal = logit_scale * video_temporal_features @ spec_temporal_features.permute(0, 2, 1)
            logits_per_spec_temporal = logit_scale * spec_temporal_features @ video_temporal_features.permute(0, 2, 1)
                
        
        # Calculated Loss: (Semantic Contrastive Loss & Temporal Contrastive Loss)

        # Semantic Contrast Loss:
        num_logits_semantic = logits_per_video_semantic.shape[0]
        if self.prev_num_logits_semantic != num_logits_semantic or device not in self.labels_semantic:
            labels_semantic = torch.arange(num_logits_semantic, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels_semantic = labels_semantic + num_logits_semantic * self.rank
            if self.cache_labels:
                self.labels_semantic[device] = labels_semantic
                self.prev_num_logits_semantic = num_logits_semantic
        else:
            labels_semantic = self.labels_semantic[device]
         
        
        semantic_contrast_loss = (F.cross_entropy(logits_per_video_semantic, labels_semantic) + F.cross_entropy(logits_per_spec_semantic, labels_semantic)) / 2

        """Temporal Contrast Loss:
        Logits_per_video_temporal: B x T x T      (B x Video x Spec)
        Logits_per_spec_temporal:  B x T x T      (B x Spec x Video)

        start_bias_index: B x 2      (video_start_index, spec_start_index) 0 ~ T - 1
        end_bias_index:   B x 2      (video_end_index,   spec_end_index)   0 ~ T - 1
        """

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

        total_loss = self.temporal_mix_weight * temporal_contrast_loss + semantic_contrast_loss

        return {"semantic_contrast_loss": semantic_contrast_loss, "temporal_contrast_loss": temporal_contrast_loss, "total_loss": total_loss, "temp_mix_weight": torch.tensor(self.temporal_mix_weight)}




# CLIP Loss Intra Contrast:
class ClipLoss_Intra_Contrast(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
            intra_contrast_weight=1,
            clip_num = 3,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # Intra Contrast Weight:
        self.intra_contrast_weight = intra_contrast_weight
        self.clip_num = clip_num

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        extra_contrast_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        # Logits_per_image: B x B
        # Slice the Similarity Matrix:
        # Get the Diagonal Block
        # intra_contrast_loss = 0
        bs = logits_per_image.shape[0]
        assert bs % self.clip_num == 0
        s = (range(bs // self.clip_num), np.s_[:], range(bs // self.clip_num), np.s_[:])
        intra_logits_per_image = logits_per_image.reshape(bs//self.clip_num, self.clip_num, bs//self.clip_num, self.clip_num)[s]   # b' x clip_num x clip_num
        intra_logits_per_text = logits_per_text.reshape(bs//self.clip_num, self.clip_num, bs//self.clip_num, self.clip_num)[s]     # b' x clip_num x clip_num
        bs_intra, num_logits_intra, _ = intra_logits_per_image.shape
        labels_intra = torch.arange(num_logits_intra, device=device, dtype=torch.long).unsqueeze(0).repeat(bs_intra, 1)
        # Intra Logits Per Image: bs x c
        intra_logits_per_image = intra_logits_per_image.reshape(bs_intra*num_logits_intra, num_logits_intra)
        intra_logits_per_text = intra_logits_per_text.reshape(bs_intra*num_logits_intra, num_logits_intra)
        labels_intra = labels_intra.reshape(bs_intra * num_logits_intra)
        # Intra Loss:
        intra_contrast_loss = (F.cross_entropy(intra_logits_per_image, labels_intra) + F.cross_entropy(intra_logits_per_text, labels_intra)) / 2

        total_loss = extra_contrast_loss + self.intra_contrast_weight * intra_contrast_loss

        # return {"contrastive_loss": total_loss} if output_dict else total_loss

        return {"total_loss": total_loss, "extra_contrast_loss": extra_contrast_loss, "intra_contrast_loss": intra_contrast_loss} if output_dict else total_loss





# CLIP Loss Intra Contrast:
class ClipLoss_Intra_Contrast_Temporal_Mean(nn.Module):
    """
    Use Max Pool for Extra Contrastive Loss,
    Use Mean Pool for Intra Contrastive Loss,
    """

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
            intra_contrast_weight=1,
            clip_num = 3,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # Intra Contrast Weight:
        self.intra_contrast_weight = intra_contrast_weight
        self.clip_num = clip_num

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, video_max_features, video_mean_features, spec_max_features, spec_mean_features, logit_scale, output_dict=False):
        device = video_max_features.device
        if self.world_size > 1:
            all_video_features, all_spec_features = gather_features(
                video_max_features, spec_max_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_video = logit_scale * video_max_features @ all_spec_features.T
                logits_per_spec = logit_scale * spec_max_features @ all_video_features.T
            else:
                logits_per_video = logit_scale * all_video_features @ all_spec_features.T
                logits_per_spec = logits_per_video.T
        else:
            logits_per_video = logit_scale * video_max_features @ spec_max_features.T
            logits_per_spec = logit_scale * spec_max_features @ video_max_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_video.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        extra_contrast_loss = (
                             F.cross_entropy(logits_per_video, labels) +
                             F.cross_entropy(logits_per_spec, labels)
                     ) / 2

        # Logits_per_image: B x B
        # Slice the Similarity Matrix:
        # Get the Diagonal Block
        # intra_contrast_loss = 0

        # Calculate CLIP Intra Contrastive Loss:
        bs, c = video_mean_features.shape
        video_mean_features_intra = video_mean_features.reshape(-1, self.clip_num, c)
        spec_mean_features_intra = spec_mean_features.reshape(-1, self.clip_num, c)


        if self.world_size > 1:
            all_video_mean_features, all_spec_mean_features = gather_features(
                video_mean_features_intra, spec_mean_features_intra,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
            if self.local_loss:
                logits_per_video_intra = logit_scale * video_mean_features_intra @ all_spec_mean_features.permute(0, 2, 1)
                logits_per_spec_intra = logit_scale * spec_mean_features_intra @ all_video_mean_features.permute(0, 2, 1)
            else:
                logits_per_video_intra = logit_scale * all_video_mean_features @ all_spec_mean_features.permute(0, 2, 1)
                logits_per_spec_intra = logits_per_video_intra.permute(0, 2, 1)
        else:
            logits_per_video_intra = logit_scale * video_mean_features_intra @ spec_mean_features_intra.permute(0, 2, 1)
            logits_per_spec_intra = logit_scale * spec_mean_features_intra @ video_mean_features_intra.permute(0, 2, 1)     # b x clip_num x clip_num
        
        # Intra Contrast Loss:
        bs_intra, num_logits_intra, _ = logits_per_video_intra.shape
        labels_intra = torch.arange(num_logits_intra, device=device, dtype=torch.long).unsqueeze(0).repeat(bs_intra, 1)
        # Intra Logits Per Image: bs x c
        logits_per_video_intra = logits_per_video_intra.reshape(bs_intra*num_logits_intra, num_logits_intra)
        logits_per_spec_intra = logits_per_spec_intra.reshape(bs_intra*num_logits_intra, num_logits_intra)
        labels_intra = labels_intra.reshape(bs_intra * num_logits_intra)
        # Intra Loss:
        intra_contrast_loss = (F.cross_entropy(logits_per_video_intra, labels_intra) + F.cross_entropy(logits_per_video_intra, labels_intra)) / 2
        total_loss = extra_contrast_loss + self.intra_contrast_weight * intra_contrast_loss
        return {"total_loss": total_loss, "extra_contrast_loss": extra_contrast_loss, "intra_contrast_loss": intra_contrast_loss} if output_dict else total_loss