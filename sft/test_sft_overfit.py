"""SFT 端到端 overfit 冒烟测试（对应 MEMORY 6.5 Test 5）。

证明：FlickerDataset -> SFTCollator -> 已有的 TinyVLM -> train_sft 这条链路打通，
且 response-only loss 的梯度能流过 visual bridge —— 在一小撮固定 image-answer 上
反复训练，loss 必须明显下降（模型记住答案）。
"""

import torch
from torch.utils.data import DataLoader

from tokenizer.tokenizer import CharTokenizer
from tinyvlm.train_vlm import build_model

from .sft_dataset import FlickerDataset
from .sft_collator import SFTCollator
from .train_sft import train_sft

IMAGE_SIZE = 32
MAX_TEXT_LEN = 64


class _ToyBase:
    """固定的 (image, caption) 对，模拟少量 VQA 样本。"""

    def __init__(self):
        torch.manual_seed(0)
        self.captions = [
            "a red car on the road.",
            "a small blue bird.",
            "two people playing football.",
            "a brown dog in the park.",
        ]
        # 每张图固定且互不相同，模型才能把图映射到各自答案
        self.images = [torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) for _ in self.captions]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.images[index], self.captions[index]


def test_sft_overfit_drops_loss():
    tokenizer = CharTokenizer()
    dataset = FlickerDataset(_ToyBase(), tokenizer, max_len=MAX_TEXT_LEN)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=SFTCollator(pad_token_id=tokenizer.pad_token_id),
    )
    model = build_model(
        image_size=IMAGE_SIZE,
        max_text_len=MAX_TEXT_LEN,
        vocab_size=tokenizer.vocab_size,
    )

    history = train_sft(
        model, loader, epochs=300, lr=1e-3, device="cpu", log_interval=100
    )

    assert len(history) >= 2
    initial_loss = history[0]
    final_loss = history[-1]
    # 在固定小数据上 overfit，loss 应当大幅下降
    assert final_loss < initial_loss * 0.3, (
        f"loss 没有明显下降：initial={initial_loss:.4f} final={final_loss:.4f}"
    )
    assert final_loss < 0.5, f"final loss 偏高，可能没记住答案：{final_loss:.4f}"
