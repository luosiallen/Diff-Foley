import csv
import os
import pickle
import sys

import numpy as np
import torch
import random


class VGGSound_audio_video_spec_fullset_Dataset_Infer(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, eval_dataset_path, split, data_dir, split_txt_path, feat_type='CAVP_feat', transforms=None, sr=22050, duration=10, truncate=220000, fps=21.5, debug_num=False, fix_frames=True):
        super().__init__()
        self.data_dir = data_dir

        if split == "train":
            self.split_path = os.path.join(self.data_dir, "Train")  # spec dir
            self.split = "Train"
        elif split == "valid" or split == 'test':
            self.split_path = os.path.join(self.data_dir, "Test")   # spec dir
            self.split = "Test"

        # Default params:
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        print("Fix Frames: {}".format(self.fix_frames))


        # Generate Data Path:
        self.eval_datset_path = eval_dataset_path
        self.data_list = os.listdir(self.eval_datset_path)
        self.audio_name_list = list(map(lambda x: "_".join(x.split("_")[:-2]), self.data_list))


        self.feat_dir = os.path.join(self.data_dir, feat_type, self.split, "contrast_feat")


        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)
    
    def load_spec_and_feat(self, spec_path, video_feat_path):
        spec_raw = np.load(spec_path).astype(np.float32)                    # channel: 1
        video_feat = np.load(video_feat_path)['feat'].astype(np.float32)    # L x C


        start_idx = 0
            
        start_frame = int(self.fps * start_idx / self.sr)
        truncate_frame = int(self.fps * self.truncate / self.sr)

        # Spec Start & Truncate: 
        spec_start = int(start_idx / 256)    # Hop_Size:
        spec_truncate = int(self.truncate / 256)

        # check spec_raw:
        if spec_raw.shape[-1] < spec_start + spec_truncate:
            repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
            spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        else:
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        
        # check video feat:
        if video_feat.shape[0] < start_frame + truncate_frame:
            repeat_num = int((start_frame + truncate_frame) // video_feat.shape[0]) + 1
            video_feat = np.tile(video_feat, (repeat_num,1))
            video_feat = video_feat[start_frame: start_frame + truncate_frame]
        else:
            video_feat = video_feat[start_frame: start_frame + truncate_frame]
        
        # Revise Spec:
        # spec_raw = spec_raw[:,:,None].repeat(3,axis=-1)    # Repeat the channel dim 1 -> 3
        spec_raw = spec_raw[None].repeat(3,axis=0)    # Repeat the channel dim 1 -> 3

        end_frame = start_frame + truncate_frame

        return spec_raw, video_feat, start_frame, end_frame


    def __getitem__(self, idx):
        audio_name = self.audio_name_list[idx]             # 
        # video_name:
        # video_name = os.path.basename(audio_name)[:-8] # remove _mel.npy
        video_feat_path = os.path.join(self.feat_dir, audio_name + ".npz")

        video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
        start_frame = 0
        truncate_frame = int(self.fps * self.truncate / self.sr) 
        video_feat = video_feat[start_frame:start_frame + truncate_frame]

        spec_path = os.path.join(self.eval_datset_path, self.data_list[idx])
        spec = np.load(spec_path).astype(np.float32)
        truncate_spec = 512
        spec = spec[:, :truncate_spec]

        data_dict = {}
        # data_dict['spec'] = audio
        data_dict['audio_name'] = audio_name
        data_dict['video_feat'] = video_feat
        data_dict["spec"] = spec[None].repeat(3, axis=0)
        data_dict["labels"] = torch.tensor(1)
        return data_dict



class VGGSound_audio_video_spec_fullset_Dataset_Train_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class VGGSound_audio_video_spec_fullset_Dataset_Valid_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class VGGSound_audio_video_spec_fullset_Dataset_Test_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)