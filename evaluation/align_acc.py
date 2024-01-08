from omegaconf import OmegaConf
from adm.util import instantiate_from_config

import os
import torch
import numpy as np
from tqdm import tqdm

import librosa
import soundfile as sf
import datetime

# distributed:
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler




def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_model_and_dataloaders(cfg, ckpt, device, is_ddp=False, eval_method="Ours"):
    
    # load model:
    model = load_model_from_config(cfg, ckpt)
    model = model.to(device)
    # get data:
    if eval_method == "Ours":
        print("Evaluating Ours Method: ========> ")
        data = instantiate_from_config(cfg.data_eval_metric)
    data.prepare_data()
    data.setup()

    if is_ddp:
        print(is_ddp)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        sampler = DistributedSampler(data.datasets['validation'], dist.get_world_size(), dist.get_rank(), shuffle=False)
        num_workers = 0
    else:
        sampler = None
        num_workers = 8
    
    dataloader = DataLoader(data.datasets['validation'],sampler=sampler, batch_size=cfg.data_eval_metric.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False) 

    return dataloader, model


def model_inference(model, batch, guidance_scale=1.0):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    print("Guidance Scale: {}".format(guidance_scale))
    model.eval()

    # labels = batch["labels"].to(model.device)

    bs = batch['spec'].shape[0]
    labels = torch.ones(bs).to(model.device)

    with torch.no_grad():
        spec = batch["spec"].to(model.device)
        video_feat = batch["video_feat"].to(model.device)
        encode_spec = model.encode_spec_z(spec)
        encode_cond = model.cond_model(video_feat)
        t = torch.tensor(0).reshape(1,).repeat(spec.shape[0]).to(model.device).long()   # Constant
        prob_logits = model.model(encode_spec, context=encode_cond, timesteps=t)
        predicted = torch.round(prob_logits)
        correct_num = ((predicted == labels.float().unsqueeze(1)).sum()).item()
    return correct_num, predicted.shape[0]



def eval_audio(gpu_id, cfg, ckpt, is_ddp, save_path=None, eval_method="Ours"):
    os.makedirs(save_path, exist_ok=True)
    print('gpu id:',gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    dataloaders, model = load_model_and_dataloaders(cfg, ckpt, device, is_ddp, eval_method)
    i = 0
    total_correct_list = []
    total_len_list = []
    # import pdb
    # pdb.set_trace()
    for batch in tqdm(dataloaders):
        i += 1
        correct_num, sample_len = model_inference(model, batch)
        print("Batch: {}  ACC: {}".format(i, correct_num / sample_len))
        total_correct_list.append(correct_num)
        total_len_list.append(sample_len)
    
    correct_list = np.array(total_correct_list)
    len_list = np.array(total_len_list)

    avg_acc = correct_list.sum() / len_list.sum()
    total_num = len_list.sum()
    return avg_acc, total_num

            

def main():
    eval_method = "Ours" # or SpecVQGAN


    save_path = "./temp_folder"


    # Eval Classifier:
    cfg_path = "./config/eval_classifier.yaml"
    ckpt = "../inference/diff_foley_ckpt/eval_classifier.ckpt"      # put the eval classifier under diff_foley_ckpt


    torch.manual_seed(0)
    local_rank = os.environ.get('LOCAL_RANK')

    if local_rank is not None:
        is_ddp = True
        local_rank = int(local_rank)
        dist.init_process_group("nccl", 'env://', datetime.timedelta(0, 300))
        print(f'WORLDSIZE {dist.get_world_size()} – RANK {dist.get_rank()}')
        if dist.get_rank() == 0:
            print('MASTER:', os.environ['MASTER_ADDR'], ':', os.environ['MASTER_PORT'])
    else:
        is_ddp = False
        local_rank = 0
    

    cfg = OmegaConf.load(cfg_path)
    avg_acc, total_num = eval_audio(local_rank, cfg, ckpt, is_ddp, save_path=save_path, eval_method=eval_method)
    print("Metric =====> Avg ACC: {}   Total Num: {}".format(avg_acc, total_num))


    with open(os.path.join(save_path, "results_metric.txt"), "w") as f:
        txt = "AVG ACC: {}   Total Num: {}".format(avg_acc, total_num)
        f.writelines(txt)
    
    print("Path:", cfg.data_eval_metric.params.validation.params.eval_dataset_path)
    


if __name__ == "__main__":
    main()



