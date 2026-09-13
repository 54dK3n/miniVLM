"""PPO prompt 数据集：只需要 (image, prompt)——response 由 policy 在线采样，没有标签。

每条样本编码成与 SFT/DPO 一致的模板前缀（保证从 SFT 模型继续时 token 分布一致）：
    prompt_ids = encode("User: {prompt}\\nAssistant:")
图像加载 transform 与 ImageCaptionDataset / DPODataset 完全一致。
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def default_image_loader(image_root, image_size):
    root = Path(image_root)

    def load(relative_path):
        with Image.open(root / relative_path) as image:
            image = image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    return load


class PPOPromptDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_root=None, image_size=32,
                 max_prompt_len=48, image_loader=None):
        with open(json_path, encoding="utf-8") as f:
            self.rows = json.load(f)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        if image_loader is not None:
            self._load_image = image_loader
        else:
            assert image_root is not None, "需要 image_root（或显式传 image_loader）"
            self._load_image = default_image_loader(image_root, image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self._load_image(row["image"])
        prompt_text = row.get("prompt", "")
        prompt_ids = self.tokenizer.encode(f"User: {prompt_text}\nAssistant:")
        if len(prompt_ids) > self.max_prompt_len:
            prompt_ids = prompt_ids[: self.max_prompt_len]
        return {
            "image": image,
            "prompt": prompt_text,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
        }
