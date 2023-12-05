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

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import to_2tuple

from .audio_contrastive import ResNet3dSlowOnly, spec_encoder_resnet50, Spec_VIT, ViViT, ViViT_mean, Spec_VIT_mean

from .audio_contrastive import init_X3D, init_I3D, init_R2plus, Cnn10, Cnn14

from .aug_utils import SpecAugmentation

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256 # n_queries for attentional pooler
    attn_pooler_heads: int = 8 # n heads for attentional_pooling
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed







# Revise CLIP: Video & Spec Model
class CLIP_Video_Spec(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            use_spec_aug: bool = False,
            video_pretrained: bool = False,
            audio_pretrained: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.video_encode = args.video_encode
        self.spec_encode = args.spec_encode

        self.use_spec_aug = use_spec_aug

        # video_pretrained:
        self.video_pretrained = video_pretrained
        self.audio_pretrained = audio_pretrained
        print("Video Pretrained: {}  Audio Pretrained: {}".format(video_pretrained, audio_pretrained))

        if self.use_spec_aug:
            print("========> Using Spec Augmenter")
            self.spec_augmenter = SpecAugmentation(time_drop_width=25, time_stripes_num=2, freq_drop_width=16, freq_stripes_num=2)

        # Spec & Video:
        if args.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
        
        elif args.video_encode == "Slowonly_pool":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        

        elif args.video_encode == "Slowonly_pool_fps8":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_pool = nn.MaxPool1d(kernel_size=32)

        
        elif args.video_encode == "X3D_L_pool":
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/x3d-L_config.yaml"
            self.video_encoder = init_X3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif args.video_encode == "I3D_pool":   # 35M
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/i3d_config.yaml"
            self.video_encoder = init_I3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif args.video_encode == "R2plus1D_pool":
            self.video_encoder = init_R2plus()
            self.video_pool = nn.MaxPool1d(kernel_size=16)

        
        if args.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
        
        elif args.spec_encode == "resnet50_pool":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)
            # print("===========> Audio Embed dim",embed_dim)
        
        elif args.spec_encode == "cnn10_pool":
            self.spec_encoder = Cnn10(embed_dim=2048)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)

        elif args.spec_encode == "cnn14_pool":      # Pretrained
            self.spec_encoder = Cnn14(embed_dim=512, pretrained=self.audio_pretrained)
            self.spec_project_head = torch.nn.Identity()
            self.spec_pool = nn.MaxPool1d(kernel_size=16)


        elif args.spec_encode == "spec_vit":
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


    def encode_video(self, video, normalize: bool = False, train=False):

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
            video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)

        elif self.video_encode == "Slowonly_pool" or self.video_encode == "Slowonly_pool_fps8":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = self.video_project_head(video_feat)
            video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)
        
        elif self.video_encode == "I3D_pool" or self.video_encode == "X3D_L_pool":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder([video])
            bs, t, c = video_feat.shape
            # Semantic Features:
            video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)

        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


    def encode_spec(self, spec, normalize: bool = False, train=False):
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
            spec_feat = self.spec_pool(spec_feat.permute(0, 2, 1)).squeeze(2)

        elif self.spec_encode == "resnet50_pool":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_feat = self.spec_project_head(spec_feat)
            spec_feat = self.spec_pool(spec_feat.permute(0,2,1)).squeeze(2)
            # print(spec_feat.shape)

        if normalize:
            spec_feat = F.normalize(spec_feat, dim=-1)
        
        return spec_feat


    def forward(self, video, spec, output_dict=True, train=False):
        video_features = self.encode_video(video, normalize=True, train=train)
        spec_features = self.encode_spec(spec, normalize=True, train=train)
        if self.output_dict:
            return {
                "image_features": video_features,
                "text_features": spec_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_features, spec_features, self.logit_scale.exp()





class CLIP_Video_Spec_Temporal(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            use_spec_aug: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.video_encode = args.video_encode
        self.spec_encode = args.spec_encode

        # Spec & Video:
        if args.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
        
        elif args.video_encode == "Slowonly_pool":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_pool = nn.MaxPool1d(kernel_size=16)

        elif args.video_encode == "X3D_L_pool":
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/x3d-L_config.yaml"
            self.video_encoder = init_X3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)


        elif args.video_encode == "vivit_base":     # 88M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=768, spatial_depth=8,  temporal_depth=4, heads=12, mlp_dim=3072)
            # self.video_cls_project_head = nn.Linear(768, embed_dim)
            self.video_temporal_project_head = nn.Linear(768, embed_dim)

        elif args.video_encode == "vivit_medium":   # 250M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=12, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)
        

        elif args.video_encode == "project_vivit_medium":   # 250M
            # Project VIVIT: Use different projector, cls_token_project, temporal_project
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=12, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)
            self.video_cls_project_head = nn.Linear(1024, embed_dim)

        elif args.video_encode == "vivit_large":    # 400M
            self.video_encoder = ViViT(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=24, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)

        # VIVIT Mean:
        elif args.video_encode == "mean_vivit_medium":   # 250M
            self.video_encoder = ViViT_mean(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=12, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)
        
                # VIVIT Mean:
        elif args.video_encode == "mean_vivit_large":   # 250M
            self.video_encoder = ViViT_mean(image_size=224, image_patch_size=32, frames=16, frame_patch_size=1, dim=1024, spatial_depth=24, temporal_depth=8, heads=16, mlp_dim=4096)
            # self.video_cls_project_head = nn.Linear(1024, embed_dim)
            self.video_temporal_project_head = nn.Linear(1024, embed_dim)


        if args.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
        
        elif args.spec_encode == "resnet50_pool":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)

        elif args.spec_encode == "spec_vit":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=512, layers=12, heads=8, mlp_ratio=4)
            self.spec_project_head = nn.Linear(512, embed_dim)

        elif args.spec_encode == "spec_vit_base":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=768, layers=16, heads=12, mlp_ratio=4, output_dim=768)
            self.spec_project_head = nn.Linear(768, embed_dim)
        
        elif args.spec_encode == "spec_vit_large":
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=24, heads=16, mlp_ratio=4, output_dim=1024)
            self.spec_project_head = nn.Linear(1024, embed_dim)
        
        # Independent Cls Token:
        elif args.spec_encode == "cls_spec_vit_base":     # 88M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=768, layers=12, heads=12, mlp_ratio=4, output_dim=768)
            # self.spec_cls_project_head = nn.Linear(768, embed_dim)
            self.spec_temporal_project_head = nn.Linear(768, embed_dim)

        elif args.spec_encode == "cls_spec_vit_medium":   # 200M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=16, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)
        
        elif args.spec_encode == "project_spec_vit_medium":   # 200M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=16, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)
            self.spec_cls_project_head = nn.Linear(1024, embed_dim)

        elif args.spec_encode == "mean_spec_vit_medium":   # 200M
            self.spec_encoder = Spec_VIT_mean(spec_size=256, patch_size=16, width=1024, layers=16, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)
        
        elif args.spec_encode == "mean_spec_vit_large":   # 200M
            self.spec_encoder = Spec_VIT_mean(spec_size=256, patch_size=16, width=1024, layers=24, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)
        
        elif args.spec_encode == "cls_spec_vit_large":    # 300M
            self.spec_encoder = Spec_VIT(spec_size=256, patch_size=16, width=1024, layers=24, heads=16, mlp_ratio=4, output_dim=1024)
            # self.spec_cls_project_head = nn.Linear(1024, embed_dim)
            self.spec_temporal_project_head = nn.Linear(1024, embed_dim)


        elif args.spec_encode == "spec_vit_mean":
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


    def encode_video(self, video, normalize: bool = False, train=False):
        # Video: B x T x 3 x H x W
        video = video.permute(0, 2, 1, 3, 4)    # B x 3 x T x H x W
        if self.video_encode == "Slowonly":
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_temporal_feat = self.video_project_head(video_feat)
            # Semantic Mean Features:
            video_mean_feat = video_temporal_feat.mean(dim=1)     # mean features
        
        elif self.video_encode == "X3D_L_pool":
            video_temporal_feat = self.video_encoder([video])
            bs, t, c = video_temporal_feat.shape
            # Semantic Features:
            video_mean_feat = self.video_pool(video_temporal_feat.permute(0,2,1)).squeeze(2)
        
        elif self.video_encode == "Slowonly_pool":
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_temporal_feat = self.video_project_head(video_feat)   # B x T x C
            # Semantic Mean Features:
            # video_mean_feat = video_temporal_feat.mean(dim=1)         # mean features
            video_mean_feat = self.video_pool(video_temporal_feat.permute(0,2,1)).squeeze(2)

        elif self.video_encode.startswith("vivit"):
            video_mean_feat, video_temporal_feat = self.video_encoder(video)
            video_mean_feat = self.video_temporal_project_head(video_mean_feat)
            video_temporal_feat = self.video_temporal_project_head(video_temporal_feat)

        elif self.video_encode.startswith("mean_vivit"):
            video_temporal_feat = self.video_encoder(video)
            video_temporal_feat = self.video_temporal_project_head(video_temporal_feat)
            video_mean_feat = video_temporal_feat.mean(dim=1)
        
        elif self.video_encode.startswith("project_vivit"):
            video_mean_feat, video_temporal_feat = self.video_encoder(video)
            video_temporal_feat = self.video_temporal_project_head(video_temporal_feat)
            video_mean_feat = self.video_cls_project_head(video_mean_feat)

        if normalize:
            video_temporal_feat = F.normalize(video_temporal_feat, dim=-1)
            video_mean_feat = F.normalize(video_mean_feat, dim=-1)

        return video_temporal_feat, video_mean_feat   # temporal features, mean feat


    def encode_spec(self, spec, normalize: bool = False, train=False):
        # spec: B x Mel_num x T

        if self.spec_encode == "resnet50":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_temporal_feat = self.spec_project_head(spec_feat)

            # Semantic Mean Features:
            spec_mean_feat = spec_temporal_feat.mean(dim=1)
        
        elif self.spec_encode == "resnet50_pool":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_temporal_feat = self.spec_project_head(spec_feat)
            # Semantic Pool Features:
            spec_mean_feat = self.spec_pool(spec_temporal_feat.permute(0,2,1)).squeeze(2)

        
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
        
        elif self.spec_encode.startswith("project_spec_vit"):
            # print(spec.shape)
            spec_mean_feat, spec_temporal_feat = self.spec_encoder(spec)
            spec_mean_feat = self.spec_temporal_project_head(spec_mean_feat)         # Cls Token:
            spec_temporal_feat = self.spec_cls_project_head(spec_temporal_feat)


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


    def forward(self, video, spec, output_dict=True, train=False):
        video_temporal_features, video_mean_features = self.encode_video(video, normalize=True, train=train)       # B x T x C
        spec_temporal_features, spec_mean_features = self.encode_spec(spec, normalize=True, train=train)          # B x T x C
        if output_dict:
            return {
                "video_temporal_features": video_temporal_features,
                "video_mean_features": video_mean_features,
                "spec_temporal_features": spec_temporal_features,
                "spec_mean_features": spec_mean_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_temporal_features, video_mean_features, spec_temporal_features, spec_mean_features, self.logit_scale.exp()





class CLIP_Video_Spec_Intra_Mean(nn.Module):
    """
    Use Mean Pool for Intra Contrastive
    Use Max  Pool for Extra Contrastive
    """

    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            use_spec_aug: bool = False,
            video_pretrained: bool = False,
            audio_pretrained: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.video_encode = args.video_encode
        self.spec_encode = args.spec_encode

        self.use_spec_aug = use_spec_aug

        # video_pretrained:
        self.video_pretrained = video_pretrained
        self.audio_pretrained = audio_pretrained
        print("Video Pretrained: {}  Audio Pretrained: {}".format(video_pretrained, audio_pretrained))

        if self.use_spec_aug:
            print("========> Using Spec Augmenter")
            self.spec_augmenter = SpecAugmentation(time_drop_width=25, time_stripes_num=2, freq_drop_width=16, freq_stripes_num=2)

        # Spec & Video:
        if args.video_encode == "Slowonly":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
        
        elif args.video_encode == "Slowonly_pool":
            self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
            # Project Head:
            self.video_project_head = nn.Linear(2048, embed_dim)
            self.video_max_pool = nn.MaxPool1d(kernel_size=16)
            self.video_mean_pool  =nn.AvgPool1d(kernel_size=16)
        
        elif args.video_encode == "X3D_L_pool":
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/x3d-L_config.yaml"
            self.video_encoder = init_X3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif args.video_encode == "I3D_pool":   # 35M
            cfg_path = "/public/MARS/Users/lsm/lsm_project/lsm_project/open_clip-main/src/open_clip/i3d_config.yaml"
            self.video_encoder = init_I3D(cfg_path)   # Already Project
            # Project Head:
            self.video_pool = nn.MaxPool1d(kernel_size=16)
        
        elif args.video_encode == "R2plus1D_pool":
            self.video_encoder = init_R2plus()
            self.video_pool = nn.MaxPool1d(kernel_size=16)

        
        if args.spec_encode == "resnet50":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
        
        elif args.spec_encode == "resnet50_pool":
            self.spec_encoder = spec_encoder_resnet50(args.truncate_sec)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)
            # print("===========> Audio Embed dim",embed_dim)
        
        elif args.spec_encode == "cnn10_pool":
            self.spec_encoder = Cnn10(embed_dim=2048)
            self.spec_project_head = nn.Linear(2048, embed_dim)
            self.spec_pool = nn.MaxPool1d(kernel_size=16)

        elif args.spec_encode == "cnn14_pool":      # Pretrained
            self.spec_encoder = Cnn14(embed_dim=512, pretrained=self.audio_pretrained)
            self.spec_project_head = torch.nn.Identity()
            self.spec_max_pool = nn.MaxPool1d(kernel_size=16)
            self.spec_mean_pool = nn.AvgPool1d(kernel_size=16)


        elif args.spec_encode == "spec_vit":
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


    def encode_video(self, video, normalize: bool = False, train=False):

        # Video: B x T x 3 x H x W
        if self.video_encode == "Slowonly":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = self.video_project_head(video_feat)
            # Avg:
            video_feat = video_feat.mean(dim=1)

        elif self.video_encode == "Slowonly_pool":
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = self.video_encoder(video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = self.video_project_head(video_feat)
            video_max_feat = self.video_max_pool(video_feat.permute(0,2,1)).squeeze(2)      # For Extra Contrastive
            video_mean_feat = self.video_mean_pool(video_feat.permute(0,2,1)).squeeze(2)    # For Intra Contrastive
        
        if normalize:
            video_max_feat = F.normalize(video_max_feat, dim=-1)
            video_mean_feat = F.normalize(video_mean_feat, dim=-1)

        return video_max_feat, video_mean_feat


    def encode_spec(self, spec, normalize: bool = False, train=False):
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
            spec_max_feat = self.spec_max_pool(spec_feat.permute(0, 2, 1)).squeeze(2)
            spec_mean_feat = self.spec_mean_pool(spec_feat.permute(0, 2, 1)).squeeze(2)

        elif self.spec_encode == "resnet50_pool":
            spec = spec.unsqueeze(1)
            spec_feat = self.spec_encoder(spec)
            bs, c, t = spec_feat.shape
            spec_feat = spec_feat.permute(0, 2, 1)
            spec_feat = self.spec_project_head(spec_feat)
            spec_feat = self.spec_pool(spec_feat.permute(0,2,1)).squeeze(2)
            # print(spec_feat.shape)

        if normalize:
            # spec_feat = F.normalize(spec_feat, dim=-1)
            spec_max_feat = F.normalize(spec_max_feat, dim=-1)
            spec_mean_feat = F.normalize(spec_mean_feat, dim=-1)
        
        return spec_max_feat, spec_mean_feat


    def forward(self, video, spec, output_dict=True, train=False):
        video_max_features, video_mean_features = self.encode_video(video, normalize=True, train=train)
        spec_max_features, spec_mean_features = self.encode_spec(spec, normalize=True, train=train)
        if output_dict:
            return {
                "video_max_features": video_max_features,
                "video_mean_features": video_mean_features,
                "spec_max_features": spec_max_features,
                "spec_mean_features": spec_mean_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_max_features, spec_max_features, self.logit_scale.exp()
