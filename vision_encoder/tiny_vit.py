import torch
import torch.nn as nn

from .patch_embedding import PatchEmbedding
from .vit_block import ViTBlock


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, d_model=64, num_heads=16, num_layers=2, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, d_model
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, images):
        x = self.patch_embed(images)
        batch_size, num_patches, _ = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : num_patches + 1, :]
        for block in self.blocks:
            x = block(x)
        return self.ln(x)

