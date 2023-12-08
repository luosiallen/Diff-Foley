import csv
import os
import pickle
import sys

import numpy as np
import torch
import random
import math


class audio_video_spec_fullset_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, dataset1, feat_type='clip', transforms=None, sr=22050, duration=10, truncate=220000, fps=21.5, debug_num=False, fix_frames=False, hop_len=256):
        super().__init__()

        if split == "train":
            self.split = "Train"
        elif split == "valid" or split == 'test':
            self.split = "Test"

        # Default params:
        self.min_duration = 2
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        print("Fix Frames: {}".format(self.fix_frames))


        # Dataset1: (VGGSound)
        assert dataset1.dataset_name == "VGGSound"
        
        # spec_dir: spectrogram path
        # feat_dir: CAVP feature path
        # video_dir: video path
        
        dataset1_spec_dir = os.path.join(dataset1.data_dir, self.split, "audio_npy_spec")
        dataset1_feat_dir = os.path.join(dataset1.data_dir, feat_type, self.split)
        dataset1_video_dir = os.path.join(dataset1.video_dir, self.split, "video_fps21.5")
        
        split_txt_path = dataset1.split_txt_path
        with open(os.path.join(split_txt_path, '{}.txt'.format(self.split)), "r") as f:
            data_list1 = f.readlines()
            data_list1 = list(map(lambda x: x.strip(), data_list1))

            spec_list1 = list(map(lambda x: os.path.join(dataset1_spec_dir, x) + "_mel.npy", data_list1))      # spec
            feat_list1 = list(map(lambda x: os.path.join(dataset1_feat_dir, x) + ".npz",     data_list1))      # feat
            video_list1 = list(map(lambda x: os.path.join(dataset1_video_dir, x) + ".mp4",   data_list1))      # video


        # Merge Data:
        self.data_list = data_list1
        self.spec_list = spec_list1 
        self.feat_list = feat_list1 
        self.video_list = video_list1

        assert len(self.data_list) == len(self.spec_list) == len(self.feat_list) == len(self.video_list)
        
        shuffle_idx = np.random.permutation(np.arange(len(self.data_list)))
        self.data_list = [self.data_list[i] for i in shuffle_idx]
        self.spec_list = [self.spec_list[i] for i in shuffle_idx]
        self.feat_list = [self.feat_list[i] for i in shuffle_idx]
        self.video_list = [self.video_list[i] for i in shuffle_idx]


        if debug_num:
            self.data_list = self.data_list[:debug_num]
            self.spec_list = self.spec_list[:debug_num]
            self.feat_list = self.feat_list[:debug_num]
            self.video_list = self.video_list[:debug_num]

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)
    

    def load_spec_and_feat(self, spec_path, video_feat_path):
        """Load audio spec and video feat"""
        spec_raw = np.load(spec_path).astype(np.float32)                    # channel: 1
        video_feat = np.load(video_feat_path)['feat'].astype(np.float32)
        
        # Padding the samples:
        spec_len = self.sr * self.duration / self.hop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, :int(spec_len)]
        
        feat_len = self.fps * self.duration
        if video_feat.shape[0] < feat_len:
            video_feat = np.tile(video_feat, (math.ceil(feat_len / video_feat.shape[0]), 1))
        video_feat = video_feat[:int(feat_len)]
        return spec_raw, video_feat


    def mix_audio_and_feat(self, spec1=None, spec2=None, video_feat1=None, video_feat2=None, video_info_dict={}, mode='single'):
        """ Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            if not self.fix_frames:
                start_idx = random.randint(0, self.sr * self.duration - self.truncate - 1)  # audio start
            else:
                start_idx = 0

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)

            # Spec Start & Truncate:
            spec_start = int(start_idx / self.hop_len)
            spec_truncate = int(self.truncate / self.hop_len)

            spec1 = spec1[:, spec_start : spec_start + spec_truncate]
            video_feat1 = video_feat1[start_frame: start_frame + truncate_frame]

            # info_dict:
            video_info_dict['video_time1'] = str(start_frame) + '_' + str(start_frame+truncate_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = ""
            return spec1, video_feat1, video_info_dict
        
        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(self.min_duration * self.sr // self.hop_len, total_spec_len - self.min_duration * self.sr // self.hop_len - 1)
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = spec_start1 + spec1_truncate_len, spec_start2 + spec2_truncate_len
            
            # concat spec:
            spec1, spec2 = spec1[:, spec_start1 : spec_end1], spec2[:, spec_start2 : spec_end2]
            concat_audio_spec = np.concatenate([spec1, spec2], axis=1)  

            # Concat Video Feat:
            start1_frame, truncate1_frame = int(self.fps * spec_start1 * self.hop_len / self.sr), int(self.fps * spec1_truncate_len * self.hop_len / self.sr)
            start2_frame, truncate2_frame = int(self.fps * spec_start2 * self.hop_len / self.sr), int(self.fps * self.truncate / self.sr) - truncate1_frame
            video_feat1, video_feat2 = video_feat1[start1_frame : start1_frame + truncate1_frame], video_feat2[start2_frame : start2_frame + truncate2_frame]
            concat_video_feat = np.concatenate([video_feat1, video_feat2])

            video_info_dict['video_time1'] = str(start1_frame) + '_' + str(start1_frame+truncate1_frame)   # Start frame, end frame
            video_info_dict['video_time2'] = str(start2_frame) + '_' + str(start2_frame+truncate2_frame)
            return concat_audio_spec, concat_video_feat, video_info_dict 



    def __getitem__(self, idx):
        
        audio_name1 = self.data_list[idx]
        spec_npy_path1 = self.spec_list[idx]
        video_feat_path1 = self.feat_list[idx]
        video_path1 = self.video_list[idx]

        # select other video:
        flag = False
        if random.uniform(0, 1) < 0.5:
            flag = True
            random_idx = idx
            while random_idx == idx:
                random_idx = random.randint(0, len(self.data_list)-1)
            audio_name2 = self.data_list[random_idx]
            spec_npy_path2 = self.spec_list[random_idx]
            video_feat_path2 = self.feat_list[random_idx]
            video_path2 = self.video_list[random_idx]

        # Load the Spec and Feat:
        spec1, video_feat1 = self.load_spec_and_feat(spec_npy_path1, video_feat_path1)

        if flag:
            spec2, video_feat2 = self.load_spec_and_feat(spec_npy_path2, video_feat_path2)
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': audio_name2, 'video_path1': video_path1, 'video_path2': video_path2}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1, spec2, video_feat1, video_feat2, video_info_dict, mode='concat')
        else:
            video_info_dict = {'audio_name1':audio_name1, 'audio_name2': "", 'video_path1': video_path1, 'video_path2': ""}
            mix_spec, mix_video_feat, mix_info = self.mix_audio_and_feat(spec1=spec1, video_feat1=video_feat1, video_info_dict=video_info_dict, mode='single')

        # print("mix spec shape:", mix_spec.shape)
        # print("mix video feat:", mix_video_feat.shape)
        data_dict = {}
        data_dict['mix_spec'] = mix_spec[None].repeat(3, axis=0)
        data_dict['mix_video_feat'] = mix_video_feat
        data_dict['mix_info_dict'] = mix_info     
        return data_dict





class audio_video_spec_fullset_Dataset_Train(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class audio_video_spec_fullset_Dataset_Valid(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class audio_video_spec_fullset_Dataset_Test(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)





