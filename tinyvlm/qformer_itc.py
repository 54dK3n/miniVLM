"""Q-Former ITC（Image-Text Contrastive）预训练。

BLIP-2 里 Q-Former 的三个预训练目标之一：让 image query 输出和文本在对比目标下对齐，
query 由此学会"该从图里抽什么"。结构：
    图像塔: image -> TinyViT -> QFormLiter -> [B,Q,d] -> 每个 query 投影+L2norm
    文本塔: caption ids -> TextEncoder(双向 self-attn + masked mean) -> 投影+L2norm
    相似度: 每个 query 对每条文本算 sim，取 query 维 max（BLIP-2 ITC 做法）
    loss:   对称 InfoNCE（image->text + text->image），可学习 temperature

复用已有手写组件：TinyViT、ViTBlock、QFormLiter。对比损失手写（见 MEMORY 第四节公式）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_encoder import TinyViT, ViTBlock

from .qformer import QFormLiter


class TextEncoder(nn.Module):
    """轻量双向文本编码器：token+pos embedding -> ViTBlock 自注意力 -> masked mean pool。"""

    def __init__(self, vocab_size, d_model, max_len, num_heads=4, num_layers=2, mlp_ratio=4, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList(ViTBlock(d_model, num_heads, mlp_ratio) for _ in range(num_layers))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        # input_ids: [B, T]
        seq_len = input_ids.size(1)
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        # 对非 pad 位置做 mean pool（pad 不进句向量）
        keep = (input_ids != self.pad_id).unsqueeze(-1).float()   # [B,T,1]
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return pooled   # [B, d_model]


class QFormerITC(nn.Module):
    def __init__(self, image_size, vit_dim, d_model, vocab_size, max_text_len,
                 num_query_token=16, proj_dim=64, pad_id=0):
        super().__init__()
        self.vit = TinyViT(image_size, 8, 3, vit_dim, 8, 2)
        self.qformer = QFormLiter(img_dim=vit_dim, d_model=d_model,
                                  num_query_token=num_query_token, num_layers=2, num_heads=4)
        self.text_encoder = TextEncoder(vocab_size, d_model, max_text_len, pad_id=pad_id)
        self.img_proj = nn.Linear(d_model, proj_dim)
        self.text_proj = nn.Linear(d_model, proj_dim)
        # 可学习温度（CLIP 做法），存 log 形式
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_image(self, images):
        query_out = self.qformer(self.vit(images))     # [B, Q, d_model]
        query_emb = self.img_proj(query_out)            # [B, Q, P]
        return F.normalize(query_emb, dim=-1)           # 每个 query 单独 L2 归一

    def encode_text(self, input_ids):
        text_emb = self.text_proj(self.text_encoder(input_ids))  # [B, P]
        return F.normalize(text_emb, dim=-1)

    @staticmethod
    def _sim_i2t(img_query_emb, text_emb):
        # 每个 query 对每条文本求点积，取 query 维 max -> [B_img, B_txt]
        sim = torch.einsum("iqp,jp->iqj", img_query_emb, text_emb)
        return sim.max(dim=1).values

    def forward(self, images, input_ids):
        img_query_emb = self.encode_image(images)       # [B, Q, P]
        text_emb = self.encode_text(input_ids)          # [B, P]
        sim_i2t = self._sim_i2t(img_query_emb, text_emb)  # [B, B]
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * sim_i2t                        # [B, B]，对角线为正样本

        labels = torch.arange(images.size(0), device=images.device)
        loss_i2t = F.cross_entropy(logits, labels)      # 行：image -> 正确 text
        loss_t2i = F.cross_entropy(logits.t(), labels)  # 列：text -> 正确 image
        return (loss_i2t + loss_t2i) / 2, logits


@torch.no_grad()
def evaluate_recall(model, images, input_ids, device, ks=(1, 5)):
    """在一组（每图唯一）image-text 对上算 image<->text 检索 Recall@K。"""
    model.eval()
    img_query_emb = model.encode_image(images.to(device))   # [N, Q, P]
    text_emb = model.encode_text(input_ids.to(device))      # [N, P]
    sim = QFormerITC._sim_i2t(img_query_emb, text_emb)      # [N, N]，对角线为正确配对
    num = sim.size(0)
    targets = torch.arange(num, device=sim.device)

    out = {}
    for name, scores in (("i2t", sim), ("t2i", sim.t())):
        ranking = scores.argsort(dim=1, descending=True)    # 每行按相似度排序
        hit_rank = (ranking == targets.unsqueeze(1)).float().argmax(dim=1)  # 正确项的名次(0-based)
        for k in ks:
            out[f"{name}_R@{k}"] = round(float((hit_rank < k).float().mean()), 4)
    return out
