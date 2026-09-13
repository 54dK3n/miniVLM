"""PPO collator：对 prompt 做**左 padding**，stack image。

为什么左 padding：生成是在序列右端逐 token 追加的，左 padding 让所有样本的"真实
末尾"对齐在同一列，采样循环可以整批 append。prompt_mask 标出真实 prompt token
（=1）与左侧 padding（=0）。

注意：TinyVLM 内部只建 causal+prefix mask，没有文本 padding mask，所以左侧 pad token
会被 attend 到。教学骨架里可接受；要严格正确可让同 batch 内 prompt 等长，或后续给
模型加 padding-aware attention（留作 TODO）。
"""

import torch


class PPOCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])  # [B,3,H,W]
        prompts = [item["prompt"] for item in batch]
        id_list = [item["prompt_ids"] for item in batch]

        batch_size = len(id_list)
        max_len = max(ids.size(0) for ids in id_list)
        prompt_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, ids in enumerate(id_list):
            length = ids.size(0)
            prompt_ids[i, max_len - length:] = ids       # 左 padding
            prompt_mask[i, max_len - length:] = 1
        return {
            "images": images,
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
        }
