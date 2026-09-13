"""Q-Former-lite 视觉语言桥接（MEMORY 模块三）。

上半部分（CrossAttention / QFormLiter）是手写的 Q-Former-lite：可学习 query token
通过 cross-attention 从 image patch tokens 里抽取固定数量 Q 的视觉摘要，再投影到 LLM hidden。
下半部分（TinyVLMQFormer / build_qformer_model）是把它接进已有 TinyVLM、替换线性
visual_proj 的集成 glue。
"""

import math

import torch
import torch.nn as nn

from tiny_gpt import TinyGPT
from vision_encoder import TinyViT

from .tinyvlm_model import TinyVLM

PATCH_SIZE = 8
MODEL_DIM = 64


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.ln_q   = nn.LayerNorm(d_model)
        self.ln_img = nn.LayerNorm(d_model)
        # 手写多头 cross-attention：Q 来自 query token，K/V 来自 image token
        self.q_proj = nn.Linear(d_model, d_model)   # 只投 query
        self.k_proj = nn.Linear(d_model, d_model)   # K 来自 image
        self.v_proj = nn.Linear(d_model, d_model)   # V 来自 image
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model))

    def _cross_attention(self, query, img):
        # query: [B, Q, C]   img: [B, N, C]
        batch_size, num_query, d_model = query.shape
        num_img = img.size(1)

        q = self.q_proj(query)
        q = q.view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Q,hd]
        k = self.k_proj(img)
        k = k.view(batch_size, num_img, self.num_heads, self.head_dim).transpose(1, 2)    # [B,H,N,hd]
        v = self.v_proj(img)
        v = v.view(batch_size, num_img, self.num_heads, self.head_dim).transpose(1, 2)    # [B,H,N,hd]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # [B,H,Q,N]
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))       # 对 image token 维归一
        out = attn @ v                                                # [B,H,Q,hd]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_query, d_model)  # [B,Q,C]
        return self.out_proj(out)

    def forward(self, query_token, img_token):
        q   = self.ln_q(query_token)
        img = self.ln_img(img_token)
        # Q 来自 query，K/V 来自 image —— 这是 cross-attention 和 self-attention 的本质区别
        query_token = query_token + self._cross_attention(q, img)    # 残差
        query_token = query_token + self.ffn(self.ln_ffn(query_token))
        return query_token


class QFormLiter(nn.Module):
    def __init__(self, img_dim, d_model, num_query_token=16, num_layers=2, num_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, d_model)
        self.query_token = nn.Parameter(torch.randn(1, num_query_token, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            CrossAttention(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )
        self.final_lm = nn.LayerNorm(d_model)

    def forward(self, image_tokens):
        B = image_tokens.size(0)
        image_tokens = self.img_proj(image_tokens)
        query_tokens = self.query_token.expand(B, -1, -1)
        for block in self.blocks:
            query_tokens = block(query_tokens, image_tokens)
        visual_tokens = self.final_lm(query_tokens)
        return visual_tokens


# ---- 集成：把 Q-Former 当桥接，替换 TinyVLM 的线性 visual_proj ----

class TinyVLMQFormer(TinyVLM):
    """与 TinyVLM 完全一致，只是视觉桥接从 nn.Linear 换成 Q-Former-lite。

    TinyVLM.forward 里 image_length 是按桥接输出长度动态取的，所以把 N 个 patch 压成
    Q 个 query 后，prefix mask / label 补 -100 / 拼接都自动适配，无需改 forward。
    """

    def __init__(self, vit, gpt, vit_dim, gpt_dim, num_query_token=16, num_layers=2, num_heads=4):
        super().__init__(vit, gpt, vit_dim, gpt_dim)
        bridge = QFormLiter(
            img_dim=vit_dim, d_model=gpt_dim,
            num_query_token=num_query_token, num_layers=num_layers, num_heads=num_heads,
        )
        # 继承的 forward 里有一句 `image_tokens.size(-1) != self.visual_proj.in_features` 断言；
        # QFormLiter 不是 nn.Linear、没有 in_features，这里补上让它 drop-in 替换 visual_proj。
        bridge.in_features = vit_dim
        self.visual_proj = bridge


def build_qformer_model(image_size, max_text_len, vocab_size,
                        num_query_token=16, bridge_layers=2, bridge_heads=4,
                        tie_word_embeddings=False):
    """对照 train_vlm.build_model，但视觉桥接用 Q-Former-lite。"""
    if image_size % PATCH_SIZE != 0:
        raise ValueError(f"image_size must be divisible by {PATCH_SIZE}")
    # 关键：序列预算用 Q（query 数）而不是 N（patch 数）
    max_seq_len = num_query_token + max_text_len
    vit = TinyViT(image_size, PATCH_SIZE, 3, MODEL_DIM, 8, 2)
    gpt = TinyGPT(MODEL_DIM, 4, 2, vocab_size, max_seq_len, tie_word_embeddings=tie_word_embeddings)
    return TinyVLMQFormer(vit, gpt, MODEL_DIM, MODEL_DIM, num_query_token, bridge_layers, bridge_heads)
