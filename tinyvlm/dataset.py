import csv
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_caption_file(caption_file):
    """Read image-caption rows without requiring a tokenizer instance."""
    caption_file = Path(caption_file)
    samples = []
    with caption_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row_index, row in enumerate(reader):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"row {row_index + 1} must contain image path and caption"
                )

            image_name = row[0].strip()
            caption = row[1].strip()
            if row_index == 0 and image_name.lower() in {
                "image",
                "filename",
                "image_path",
            } and caption.lower() == "caption":
                continue
            samples.append((image_name, caption))
    return samples


class ImageCaptionDataset(Dataset):
    """Load image-caption pairs from a two-column CSV file."""

    def __init__(
        self,
        image_root,
        caption_file,
        tokenizer,
        image_size=64,
        max_text_len=32,
    ):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_text_len < 1:
            raise ValueError("max_text_len must be at least 1")

        self.image_root = Path(image_root)
        self.caption_file = Path(caption_file)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.samples = self._read_caption_file()

        if not self.samples:
            raise ValueError(f"no image-caption rows found in {self.caption_file}")

    def _read_caption_file(self):
        return read_caption_file(self.caption_file)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, relative_path):
        image_path = self.image_root / relative_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0

        # [H,W,3] -> [3,H,W]
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index):
        image_name, caption = self.samples[index]
        image = self._load_image(image_name)

        # Reserve one position for BOS in input_ids and EOS in labels.
        caption_ids = self.tokenizer.encode(caption)[: self.max_text_len - 1]
        input_ids = [self.tokenizer.bos_token_id] + caption_ids
        labels = caption_ids + [self.tokenizer.eos_token_id]

        return {
            "image": image,  # [3,H,W]
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # [T]
            "labels": torch.tensor(labels, dtype=torch.long),  # [T]
            "caption": caption,
        }


def vlm_collate_fn(batch, tokenizer):
    """Stack images and right-pad variable-length caption tensors."""
    images = torch.stack([sample["image"] for sample in batch])  # [B,3,H,W]
    max_length = max(sample["input_ids"].numel() for sample in batch)

    input_ids = torch.full(
        (len(batch), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)

    for row, sample in enumerate(batch):
        length = sample["input_ids"].numel()
        input_ids[row, :length] = sample["input_ids"]
        labels[row, :length] = sample["labels"]

    return images, input_ids, labels


def make_vlm_collate_fn(tokenizer):
    """Return the tokenizer-bound function expected by DataLoader."""
    return partial(vlm_collate_fn, tokenizer=tokenizer)


def split_by_image(samples, val_ratio=0.1, seed=42):
    """Randomly split caption rows while keeping each image in only one split."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    image_names = sorted({image_name for image_name, _ in samples})
    if len(image_names) < 2:
        raise ValueError("at least two distinct images are required")

    random.Random(seed).shuffle(image_names)
    val_image_count = max(1, round(len(image_names) * val_ratio))
    val_images = set(image_names[:val_image_count])
    train_images = set(image_names[val_image_count:])
    train_indices = [
        index
        for index, (image_name, _) in enumerate(samples)
        if image_name in train_images
    ]
    val_indices = [
        index
        for index, (image_name, _) in enumerate(samples)
        if image_name in val_images
    ]
    return train_indices, val_indices, train_images, val_images
