import torch.nn as nn

from .vit_attention import ViTAttention


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ViTAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
