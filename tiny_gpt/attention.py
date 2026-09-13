import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.view(1, 1, seq_len, seq_len)
