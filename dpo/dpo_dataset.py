"""DPO chosen/rejected 偏好数据集。

每条样本对 chosen 和 rejected 各自构造 token：
    prompt_ids = encode("User: {prompt}\\nAssistant:")
    answer_ids = encode(" " + answer) + [eos]
    input_ids  = prompt_ids + answer_ids          # 完整序列（不是 full[:-1]）
    labels     = [-100]*len(prompt_ids) + answer_ids   # prompt 段 -100，answer 段保留（含 EOS）

注意 shift 约定：这里用「对齐 label」（input_ids 与 labels 等长、同位置），
next-token shift 留给将来的 dpo_loss（它自己 gather logprob，做 logits[:, :-1] vs labels[:, 1:]，
并跳过 visual prefix 的偏移）。这与 SFT FlickerDataset 的「数据侧 shift」不同，
因为 DPO 不走模型内置 cross_entropy，而是自己算 sequence logprob。

visual prefix 由模型内部加，这里只处理文本 token 和 image 加载。
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def default_image_loader(image_root, image_size):
    """与 ImageCaptionDataset 一致的图像 transform：RGB -> resize -> [3,H,W] in [0,1]。"""
    root = Path(image_root)

    def load(relative_path):
        with Image.open(root / relative_path) as image:
            image = image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    return load


class DPODataset(Dataset):
    def __init__(self, json_path, tokenizer, image_root=None, image_size=32,
                 max_len=64, image_loader=None):
        with open(json_path, encoding="utf-8") as f:
            self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        if image_loader is not None:
            self._load_image = image_loader
        else:
            assert image_root is not None, "需要 image_root（或显式传 image_loader）"
            self._load_image = default_image_loader(image_root, image_size)

    def __len__(self):
        return len(self.pairs)

    def _build(self, prompt, answer):
        # 与 SFT 一致的模板，保证 DPO 从 SFT 模型继续时 token 分布一致
        prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant:")
        answer_ids = self.tokenizer.encode(" " + answer) + [self.tokenizer.eos_token_id]

        # 截断：保证 answer 至少 1 个 token + EOS（EOS 不能被截掉）
        if len(prompt_ids) > self.max_len - 2:
            prompt_ids = prompt_ids[: self.max_len - 2]
        budget = self.max_len - len(prompt_ids)
        if len(answer_ids) > budget:
            answer_ids = answer_ids[: budget - 1] + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)   # 对齐 label：prompt 段 -100
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = self._load_image(pair["image"])
        chosen_input_ids, chosen_labels = self._build(pair["prompt"], pair["chosen"])
        rejected_input_ids, rejected_labels = self._build(pair["prompt"], pair["rejected"])
        return {
            "image": image,
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
        }
