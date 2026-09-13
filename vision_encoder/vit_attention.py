import math

import torch.nn as nn
import torch.nn.functional as F


class ViTAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, num_tokens, d_model = x.shape
        qkv = self.qkv(x).view(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = F.softmax(
            q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1
        )
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, d_model
        )
        return self.out_proj(output)
