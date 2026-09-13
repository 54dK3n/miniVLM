"""Response-only label mask collator。

把 ``FlickerDataset`` 产出的变长样本 pad 成一个 batch：

- ``input_ids``  : 右 pad 到 batch 内最长长度，pad 位填 ``pad_token_id``
- ``labels``     : 同样右 pad，但 pad 位填 ``label_pad_id``(-100)，
                   这样 padding 不参与 cross-entropy（response-only loss 的 padding 部分）
- ``attention_mask`` : 真实 token 为 1，pad 位为 0
- ``image``      : 直接 stack 成 [B, ...]

注意：prompt 段的 -100 已经在 dataset 里做好了，这里只负责把 *padding* 段也置为 -100。
"""

import torch


class SFTCollator:
    def __init__(self, pad_token_id, label_pad_id=-100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch):
        batch_size = len(batch)
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        max_len = max(ids.size(0) for ids in input_ids_list)

        # 全部预填 pad / -100 / 0，再把真实 token 写进左侧（右 padding）
        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len), self.label_pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            length = ids.size(0)
            input_ids[i, :length] = ids
            labels[i, :length] = lbl
            attention_mask[i, :length] = 1

        out = {
            "input_ids": input_ids,            # [B, max_len]
            "labels": labels,                  # [B, max_len]
            "attention_mask": attention_mask,  # [B, max_len]
            "prompt": [item["prompt"] for item in batch],
            "caption": [item["caption"] for item in batch],
        }

        # image 可能为 None（纯文本样本）；只有都是 tensor 时才 stack
        images = [item["image"] for item in batch]
        if all(isinstance(img, torch.Tensor) for img in images):
            out["image"] = torch.stack(images, dim=0)  # [B, C, H, W]
        else:
            out["image"] = images

        return out
