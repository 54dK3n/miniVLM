from dataclasses import dataclass


@dataclass
class TinyGPTConfig:
    vocab_size: int = 21
    max_seq_len: int = 64
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
