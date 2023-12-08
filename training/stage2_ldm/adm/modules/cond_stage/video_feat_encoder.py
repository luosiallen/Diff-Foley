

import torch.nn as nn
import torch

from ..diffusionmodules.attention_openai import SpatialTransformer_Cond

class Video_Feat_Encoder(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, latent_len=3760):
        super().__init__() 
        self.embedder = nn.Sequential(
            nn.Linear(origin_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)        # B x 117 x C
        # x = torch.randn(x.shape[0], 3760, 128).to(x.device)
        
        return x





class Video_Feat_Encoder_simple(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, latent_len=3760):
        super().__init__() 
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))
    
    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)        # B x 117 x C
        # x = torch.randn(x.shape[0], 3760, 128).to(x.device)
        return x





class Video_Feat_Encoder_Posembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__() 
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        # Revise the shape here:
        bs, seq_len, c = x.shape
        x = self.embedder(x)        # B x 117 x C
        pos_embedding = self.pos_emb(torch.arange(seq_len, device=x.device).reshape(1,-1)).repeat(bs, 1, 1)
        x = x + pos_embedding
        return x



class FusionNet(nn.Module):
    def __init__(self, hidden_dim, embed_dim, depth, heads=8, d_head=64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.fusion_module = SpatialTransformer_Cond(in_channels=hidden_dim, n_heads=heads, d_head=d_head, depth=depth)
        self.proj_out = nn.Sequential(nn.Linear(hidden_dim, embed_dim))


    def forward(self, video_feat, spec_feat):
        """
        Input:
            video_feat: B x L x C
            spec_feat: B x C x H x W
        Output:
            B x L x C
        """
        bs, c, h, w = spec_feat.shape
        spec_feat = spec_feat.permute(0, 2, 3, 1).reshape(bs, -1, c)
        fusion_features = self.fusion_module(video_feat, spec_feat)
        fusion_features = self.proj_out(fusion_features)
        return fusion_features


class Video_Feat_Encoder_Posembed_AR(nn.Module):
    """ Transform the video feat encoder"""
    """
        Input:
            Data Dict: 
                video_feat:  B x L x C
                spec_prev_z: B x C x H x W
    """
    def __init__(self, origin_dim, hidden_dim, embed_dim, depth=2, seq_len=215):
        super().__init__() 
        self.embed_video_feat = nn.Sequential(nn.Linear(origin_dim, hidden_dim))
        self.embed_spec_feat = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=hidden_dim, kernel_size=1))
        self.fusion_net = FusionNet(hidden_dim, embed_dim, depth)
        self.pos_emb_video = nn.Embedding(seq_len, hidden_dim)
        self.pos_emb_spec = nn.Embedding(seq_len, hidden_dim)

    def forward(self, x):
        video_feat = x['video_feat']
        spec_prev_z = x['spec_prev_z']

        bs, seq_len, c = video_feat.shape
        bs, _, spec_h, spec_w = spec_prev_z.shape

        video_feat = self.embed_video_feat(video_feat)  # B x L x C
        spec_feat = self.embed_spec_feat(spec_prev_z)   # B x C' x H x W

        # Add Pos Embedding:
        pos_embed_video = self.pos_emb_video(torch.arange(seq_len, device=video_feat.device).reshape(1, -1)).repeat(bs, 1, 1)
        video_feat = video_feat + pos_embed_video

        pos_embed_spec = self.pos_emb_spec(torch.arange(spec_w, device=video_feat.device).reshape(1, -1)).permute(0, 2, 1).unsqueeze(2)     # 1 x C x W
        pos_embed_spec = pos_embed_spec.repeat(bs, 1, spec_h, 1)
        spec_feat = spec_feat + pos_embed_spec

        # Features Fusion:  Cross Attention
        fuse_features = self.fusion_net(video_feat, spec_feat)
        return fuse_features