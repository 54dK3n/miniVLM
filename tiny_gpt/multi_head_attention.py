import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        output = scaled_dot_product_attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_projection(output)
        if use_cache:
            return output, present_kv
        return output
