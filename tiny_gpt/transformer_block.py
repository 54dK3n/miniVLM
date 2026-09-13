import torch.nn as nn

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ratio=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = d_model * ratio
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, self.hidden_dim)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        if use_cache:
            attention_output, present_kv = self.attn(
                self.ln1(x), mask=mask, past_kv=past_kv, use_cache=True
            )
        else:
            attention_output = self.attn(self.ln1(x), mask=mask)
            present_kv = None

        x = x + attention_output
        x = x + self.mlp(self.ln2(x))
        if use_cache:
            return x, present_kv
        return x
