import argparse
import csv

import numpy as np
import torch
import pytest
from PIL import Image

from tinyvlm.train_vlm import train


@pytest.mark.parametrize("tokenizer_type", ["char", "word"])
def test_train_validation_shuffle_generate_and_checkpoint(tmp_path, tokenizer_type):
    rows = []
    for index, caption in enumerate(("a cat", "a dog", "a bird", "a ball")):
        image_name = f"image_{index}.png"
        pixels = np.full((16, 16, 3), 40 + index * 50, dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / image_name)
        rows.append((image_name, caption))

    caption_file = tmp_path / "captions.csv"
    with caption_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "caption"))
        writer.writerows(rows)

    save_path = tmp_path / "best_tinyvlm.pt"
    args = argparse.Namespace(
        image_root=str(tmp_path),
        caption_file=str(caption_file),
        batch_size=2,
        epochs=1,
        lr=1e-3,
        image_size=16,
        max_text_len=8,
        val_ratio=0.5,
        seed=42,
        num_samples=1,
        max_new_tokens=4,
        num_workers=0,
        log_interval=100,
        limit_samples=None,
        save_path=str(save_path),
        no_amp=True,
        tokenizer=tokenizer_type,
        min_word_frequency=1,
        max_vocab_size=None,
    )

    model = train(args)

    assert model.training
    assert save_path.exists()
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert torch.isfinite(torch.tensor(checkpoint["validation"]["val_loss"]))
    assert torch.isfinite(
        torch.tensor(checkpoint["validation"]["val_shuffle_loss"])
    )
    assert torch.isfinite(torch.tensor(checkpoint["validation"]["val_zero_loss"]))
    assert torch.isfinite(torch.tensor(checkpoint["validation"]["val_noise_loss"]))
    assert checkpoint["tokenizer_stoi"]["<PAD>"] == 0
    assert checkpoint["tokenizer_type"] == tokenizer_type
