import torch.nn as nn
import torch

class Video_Feat_Encoder_Posembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__() 
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        # Revise the shape here:
        bs, seq_len, c = x.shape
        x = self.embedder(x)       
        pos_embedding = self.pos_emb(torch.arange(seq_len, device=x.device).reshape(1,-1)).repeat(bs, 1, 1)
        x = x + pos_embedding
        return x






