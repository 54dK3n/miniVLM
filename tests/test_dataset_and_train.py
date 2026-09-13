import csv

import numpy as np
import torch
from PIL import Image

from tiny_gpt import TinyGPT
from tinyvlm import TinyVLM, generate
from tinyvlm.dataset import ImageCaptionDataset, vlm_collate_fn
from tokenizer import (
    CharTokenizer,
    WordTokenizer,
    tokenizer_from_checkpoint,
)
from vision_encoder import TinyViT


def _write_image(path, value):
    pixels = np.full((12, 10, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_dataset_collate_forward_and_backward(tmp_path):
    _write_image(tmp_path / "first.png", 64)
    _write_image(tmp_path / "second.png", 192)
    caption_file = tmp_path / "captions.csv"
    with caption_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "caption"])
        writer.writerow(["first.png", "a cat"])
        writer.writerow(["second.png", "dog!"])

    tokenizer = CharTokenizer()
    dataset = ImageCaptionDataset(
        tmp_path,
        caption_file,
        tokenizer,
        image_size=16,
        max_text_len=8,
    )

    first = dataset[0]
    second = dataset[1]
    assert first["image"].shape == (3, 16, 16)
    assert first["input_ids"][0].item() == tokenizer.bos_token_id
    assert first["labels"][-1].item() == tokenizer.eos_token_id
    assert len(first["input_ids"]) == len(first["labels"])

    images, input_ids, labels = vlm_collate_fn(
        [first, second], tokenizer=tokenizer
    )
    assert images.shape == (2, 3, 16, 16)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids[1, -1].item() == tokenizer.pad_token_id
    assert labels[1, -1].item() == -100

    # 2x2 patches + CLS + at most 8 text tokens = 13 positions.
    vit = TinyViT(16, 8, 3, 16, 4, 1)
    gpt = TinyGPT(16, 4, 1, tokenizer.vocab_size, max_seq_len=13)
    model = TinyVLM(vit, gpt, vit_dim=16, gpt_dim=16)

    logits, loss = model(images, input_ids, labels)
    assert logits.shape == (2, 5 + input_ids.size(1), tokenizer.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.visual_proj.weight.grad is not None


def test_tokenizer_unknown_and_generation_shape():
    tokenizer = CharTokenizer()
    assert tokenizer.encode("A") == tokenizer.encode("a")
    assert tokenizer.encode("你") == [tokenizer.unk_token_id]

    vit = TinyViT(16, 8, 3, 16, 4, 1)
    gpt = TinyGPT(16, 4, 1, tokenizer.vocab_size, max_seq_len=10)
    model = TinyVLM(vit, gpt, vit_dim=16, gpt_dim=16)
    generated = generate(
        model,
        torch.randn(1, 3, 16, 16),
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        max_new_tokens=4,
    )
    assert generated.shape[0] == 1
    assert 2 <= generated.shape[1] <= 5


def test_word_tokenizer_build_encode_decode_and_restore():
    tokenizer = WordTokenizer.from_texts(
        ["A red circle.", "A blue circle!", "A red square."],
        min_frequency=1,
    )

    ids = tokenizer.encode("A red circle.")
    assert len(ids) == 4
    assert tokenizer.decode(ids) == "a red circle."
    assert tokenizer.encode("unseen") == [tokenizer.unk_token_id]
    assert tokenizer.stoi["a"] < tokenizer.stoi["blue"]

    restored = tokenizer_from_checkpoint(
        {
            "tokenizer_type": "word",
            "tokenizer_stoi": tokenizer.stoi,
            "config": {},
        }
    )
    assert restored.tokenizer_type == "word"
    assert restored.stoi == tokenizer.stoi
    assert restored.decode(restored.encode("A blue circle!")) == "a blue circle!"
