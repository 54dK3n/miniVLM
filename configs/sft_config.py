"""多模态 SFT 配置。"""

from dataclasses import dataclass


@dataclass
class SFTConfig:
    # ---- 数据 ----
    sft_data_path: str = "data/processed/sft_train.json"
    max_len: int = 128                 # 字符级 tokenizer 下 64 太小，answer 会被截掉

    # ---- token id（默认与 CharTokenizer 的 SPECIAL_TOKENS 顺序一致）----
    pad_token_id: int = 0              # <PAD>
    label_pad_id: int = -100           # cross_entropy 的 ignore_index

    # ---- 训练 ----
    batch_size: int = 8
    lr: float = 1e-4
    num_epochs: int = 3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # ---- 杂项 ----
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 10
    ckpt_path: str = "checkpoints/sft.pt"
