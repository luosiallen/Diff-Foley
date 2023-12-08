""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .clip_modules import ResNet3dSlowOnly, spec_encoder_resnet50, Spec_VIT, ViViT, ViViT_mean, Spec_VIT_mean, Cnn14



# Revise CLIP: Video & Spec Model
class CLIP_Video_Spec(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            video_encode,
            spec_encode,
            embed_dim: int,
            output_dict: bool = False,
            truncate_sec = 4.
    ):
        super().__init__()
        self.output_dict = output_dict

        self.video_encode = video_encode
        self.spec_encode = spec_encode


        # Spec & Video:
        if self.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)

        if self.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(truncate_sec=truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
        elif self.spec_encode == "spec_vit":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=16, heads=12, mlp_ratio=4)
            self.spec_project_head = nn.Linear(1024, embed_dim)
        
        # Project Head:
        self.video_project_head = nn.Linear(2048, embed_dim)


        # Logit Scale:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        # text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        # self.transformer = text.transformer
        # self.vocab_size = text.vocab_size
        # self.token_embedding = text.token_embedding
        # self.positional_embedding = text.positional_embedding
        # self.ln_final = text.ln_final
        # self.text_projection = text.text_projection
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable


    def encode_video(self, video, normalize: bool = True, avg: bool = True):
        # Video: B x T x 3 x H x W
        video = video.permute(0, 2, 1, 3, 4)
        video_feat = self.video_encoder(video)
        bs, c, t, _, _ = video_feat.shape
        video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
        video_feat = self.video_project_head(video_feat)        # B x T x C

        # Avg:
        if avg:
            video_feat = video_feat.mean(dim=1)

        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


    def encode_spec(self, spec, normalize: bool = True, avg: bool=True):
        # spec: B x Mel_num x T
        spec = spec.unsqueeze(1)
        spec_feat = self.spec_encoder(spec)
        bs, c, t = spec_feat.shape
        spec_feat = spec_feat.permute(0, 2, 1)
        spec_feat = self.spec_project_head(spec_feat)

        # Avg:
        if avg:
            spec_feat = spec_feat.mean(dim=1)
        
        if normalize:
            spec_feat = F.normalize(spec_feat, dim=-1)
        
        return spec_feat


    def forward(self, video, spec, output_dict=True):
        video_features = self.encode_video(video, normalize=True)
        spec_features = self.encode_spec(spec, normalize=True)
        if output_dict:
            return {
                "video_features": video_features,
                "spec_features": spec_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_features, spec_features, self.logit_scale.exp()





class CLIP_Video_Spec_Temporal(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            video_encode: str,
            spec_encode: str,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.video_encode = video_encode
        self.spec_encode = spec_encode

        # Spec & Video:
        if self.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
        
        elif self.video_encode == "vivit_base":     # 88M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=768, spatial_depth=8,  temporal_depth=4, heads=12, mlp_dim=3072)
            # self.video_cls_project_head = nn.Linear(768, embed_dim)
            self.video_temporal_project_head = nn.Linear(768, embed_dim)

        elif self.video_encode == "vivit_medium":   # 250M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=12, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)

        elif self.video_encode == "vivit_large":    # 400M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=24, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)

        # VIVIT Mean:
        elif self.video_encode == "mean_vivit_medium":   # 250M
            self.video_encoder = ViViT_mean(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=12, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)


        if self.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(self.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)

        elif self.spec_encode == "spec_vit":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=512, layers=12, heads=8, mlp_ratio=4)
            self.spec_project_head = nn.Linear(512, embed_dim)

        elif self.spec_encode == "spec_vit_base":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=768, layers=16, heads=12, mlp_ratio=4, output_dim=768)
            self.spec_project_head = nn.Linear(768, embed_dim)
        
        elif self.spec_encode == "spec_vit_large":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=24, heads=16, mlp_ratio=4, output_dim=1024)
            self.spec_project_head = nn.Linear(1024, embed_dim)
        
        # Independent Cls Token:
        elif self.spec_encode == "cls_spec_vit_base":     # 88M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=768, layers=12, heads=12, mlp_ratio=4, output_dim=768)
            # self.spec_cls_project_head = nn.Linear(768, embed_dim)
            self.spec_temporal_project_head = nn.Linear(768, embed_dim)

        elif self.spec_encode == "cls_spec_vit_medium":   # 200M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=16, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)

        elif self.spec_encode == "mean_spec_vit_medium":   # 200M
            self.spec_encoder = Spec_VIT_mean(spec_size=256, patch_size=16, width=1024, layers=16, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)
        
        elif self.spec_encode == "cls_spec_vit_large":    # 300M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=24, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)


        elif self.spec_encode == "spec_vit_mean":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=512, layers=12, heads=8, mlp_ratio=4, cls_token=False, global_average_pool=True)
            self.spec_project_head = nn.Linear(512, embed_dim)
        


        # Logit Scale:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        # text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        # self.transformer = text.transformer
        # self.vocab_size = text.vocab_size
        # self.token_embedding = text.token_embedding
        # self.positional_embedding = text.positional_embedding
        # self.ln_final = text.ln_final
        # self.text_projection = text.text_projection
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable


    def encode_video(self, video, normalize: bool = False):
        # Video: B x T x 3 x H x W
        video = video.permute(0, 2, 1, 3, 4)    # B x 3 x T x H x W
        if self.video_encode == "Slowonly":
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_temporal_feat = self.video_project_head(video_feat)

            # Semantic Mean Features:
            video_mean_feat = video_temporal_feat.mean(dim=1)     # mean features
        
        elif self.video_encode.startswith("vivit"):
            video_mean_feat, video_temporal_feat = self.video_encoder(video)
            video_mean_feat = self.video_temporal_project_head(video_mean_feat)
            video_temporal_feat = self.video_temporal_project_head(video_temporal_feat)

        elif self.video_encode.startswith("mean_vivit"):
            video_temporal_feat = self.video_encoder(video)
            video_temporal_feat = self.video_temporal_project_head(video_temporal_feat)
            video_mean_feat = video_temporal_feat.mean(dim=1)

        if normalize:
            video_temporal_feat = F.normalize(video_temporal_feat, dim=-1)
            video_mean_feat = F.normalize(video_mean_feat, dim=-1)

        return video_temporal_feat, video_mean_feat   # temporal features, mean feat


    def encode_spec(self, spec, normalize: bool = False):
        # spec: B x Mel_num x T
        if self.spec_encode == "resnet50":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_temporal_feat = self.spec_project_head(spec_feat)

            # Semantic Mean Features:
            spec_mean_feat = spec_temporal_feat.mean(dim=1)
        

        elif self.spec_encode.startswith("spec_vit"):
            # print(spec.shape)
            spec_mean_feat, spec_temporal_feat = self.spec_encoder(spec)
            spec_mean_feat = self.spec_project_head(spec_mean_feat)         # Cls Token:
            spec_temporal_feat = self.spec_project_head(spec_temporal_feat)
        
        
        elif self.spec_encode.startswith("cls_spec_vit"):
            # print(spec.shape)
            spec_mean_feat, spec_temporal_feat = self.spec_encoder(spec)
            spec_mean_feat = self.spec_temporal_project_head(spec_mean_feat)         # Cls Token:
            spec_temporal_feat = self.spec_temporal_project_head(spec_temporal_feat)


        elif self.spec_encode.startswith("mean_spec_vit"):
            # print(spec.shape)
            spec_temporal_feat = self.spec_encoder(spec)
            spec_temporal_feat = self.spec_temporal_project_head(spec_temporal_feat)         # Cls Token:
            spec_mean_feat = spec_temporal_feat.mean(dim=1)

        elif self.spec_encode == "spec_vit_mean":
            # print(spec.shape)
            spec_mean_feat, spec_temporal_feat = self.spec_encoder(spec)
            spec_temporal_feat = self.spec_project_head(spec_temporal_feat)

        if normalize:
            spec_temporal_feat = F.normalize(spec_temporal_feat, dim=-1)
            spec_mean_feat = F.normalize(spec_mean_feat, dim=-1)
        
        return spec_temporal_feat, spec_mean_feat


    def forward(self, video, spec, output_dict=True):
        video_temporal_features, video_mean_features = self.encode_video(video, normalize=True)       # B x T x C
        spec_temporal_features, spec_mean_features = self.encode_spec(spec, normalize=True)          # B x T x C
        if output_dict:
            return {
                "video_temporal_features": video_temporal_features,
                "video_mean_features": video_mean_features,
                "spec_temporal_features": spec_temporal_features,
                "spec_mean_features": spec_mean_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_temporal_features, video_mean_features, spec_temporal_features, spec_mean_features, self.logit_scale.exp()



## Version2:
class CLIP_Video_Spec_v2(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            video_encode,
            spec_encode,
            embed_dim: int,
            use_spec_aug: bool = False,
            video_pretrained: bool = False,
            audio_pretrained: bool = False,
    ):
        super().__init__()

        self.video_encode = video_encode
        self.spec_encode = spec_encode

        self.use_spec_aug = use_spec_aug

        # video_pretrained:
        self.video_pretrained = video_pretrained
        self.audio_pretrained = audio_pretrained
        print("Video Pretrained: {}  Audio Pretrained: {}".format(video_pretrained, audio_pretrained))

        if self.use_spec_aug:
            print("========> Using Spec Augmenter")
            self.spec_augmenter = SpecAugmentation(time_drop_width=25, time_stripes_num=2, freq_drop_width=16, freq_stripes_num=2)

        # Spec & Video:
        if self.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
        
        elif self.video_encode == "Slowonly_pool":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif self.video_encode == "Slowonly_pool_fps8":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_pool = nn.MaxPool1d(kernel_size=32)

        
        elif self.video_encode == "X3D_L_pool":
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/x3d-L_config.yaml"
            self.video_encoder = init_X3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif self.video_encode == "I3D_pool":   # 35M
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/i3d_config.yaml"
            self.video_encoder = init_I3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif self.video_encode == "R2plus1D_pool":
            self.video_encoder = init_R2plus()
            self.video_pool = nn.MaxPool1d(kernel_size=16)

        
        if self.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(self.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
        
        elif self.spec_encode == "resnet50_pool":
            self.spec_encoder = spec_encoder_resnet50(self.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)
            # print("===========> Audio Embed dim",embed_dim)
        
        elif self.spec_encode == "cnn10_pool":
            self.spec_encoder = Cnn10(embed_dim=2048)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)

        elif self.spec_encode == "cnn14_pool":      # Pretrained
            self.spec_encoder = Cnn14(embed_dim=512, pretrained=self.audio_pretrained)
            self.spec_project_head = torch.nn.Identity()
            self.spec_pool = nn.MaxPool1d(kernel_size=16)


        elif self.spec_encode == "spec_vit":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=512, layers=12, heads=8, mlp_ratio=4)
            self.spec_project_head = nn.Linear(512, embed_dim)
        

        # Logit Scale:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Load Pretrained:
        if self.video_pretrained:
            # Load Video pretained:
            ckpt_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/pretrained_model/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth"
            state_dict = torch.load(ckpt_path)['state_dict']
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = state_dict[key]
            self.video_encoder.load_state_dict(new_state_dict, strict=False)
            print("====> Load Pretrained Video Encoder Success from: {}".format(ckpt_path))
        
        if self.audio_pretrained:
            # Load Audio pretrained:
            ckpt_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/pretrained_model/Cnn14_16k_mAP=0.438.pth"
            state_dict = torch.load(ckpt_path)['model']
            self.spec_encoder.load_state_dict(state_dict, strict=False)
            print("====> Load Pretrained Audio Encoder Succes from: {}".format(ckpt_path))
            

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable


    def encode_video(self, video, normalize: bool = False, train=False, pool=True):

        # Video: B x T x 3 x H x W
        if self.video_encode == "Slowonly":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = self.video_project_head(video_feat)
            # Avg:
            video_feat = video_feat.mean(dim=1)
        
        elif self.video_encode == "R2plus1D_pool":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, t, c = video_feat.shape
            # Semantic Features:
            if pool:
                video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)

        elif self.video_encode == "Slowonly_pool" or self.video_encode == "Slowonly_pool_fps8":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = self.video_project_head(video_feat)
            if pool:
                video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)
        
        elif self.video_encode == "I3D_pool" or self.video_encode == "X3D_L_pool":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder([video])
            bs, t, c = video_feat.shape
            # Semantic Features:
            if pool:
                video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)

        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


    def encode_spec(self, spec, normalize: bool = False, train=False, pool=True):
        # spec: B x Mel_num x T
        if train and self.use_spec_aug:         # Augmentation:
            spec = self.spec_augmenter(spec)    

        if self.spec_encode == "resnet50":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_feat = self.spec_project_head(spec_feat)
            # Avg:
            spec_feat = spec_feat.mean(dim=1)

        elif self.spec_encode=="cnn14_pool" or self.spec_encode=="cnn10_pool":
            spec = spec.unsqueeze(1)                        # B x 1 x Mel x T
            spec = spec.permute(0, 1, 3, 2)                 # B x 1 x T x Mel
            spec_feat = self.spec_encoder(spec)             # B x T x C
            spec_feat = self.spec_project_head(spec_feat)
            if pool:
                spec_feat = self.spec_pool(spec_feat.permute(0, 2, 1)).squeeze(2)

        elif self.spec_encode == "resnet50_pool":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_feat = self.spec_project_head(spec_feat)
            if pool:
                spec_feat = self.spec_pool(spec_feat.permute(0,2,1)).squeeze(2)
            # print(spec_feat.shape)

        if normalize:
            spec_feat = F.normalize(spec_feat, dim=-1)
        
        return spec_feat


    def forward(self, video, spec, output_dict=True, train=False):
        video_features = self.encode_video(video, normalize=True, train=train)
        spec_features = self.encode_spec(spec, normalize=True, train=train)
        if output_dict:
            return {
                "video_features": video_features,
                "spec_features": spec_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_features, spec_features, self.logit_scale.exp()