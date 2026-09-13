from dataclasses import dataclass


@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 8
    in_channels: int = 3
    d_model: int = 64
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: int = 4
