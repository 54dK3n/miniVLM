"""DPO collator：对 chosen / rejected 分别做文本 padding，image stack。

只负责文本 padding 和 image batching；visual prefix 由模型内部加。
chosen 和 rejected 共用同一张 image。
"""

import torch


class DPOCollator:
    def __init__(self, pad_id, label_pad=-100):
        self.pad_id = pad_id
        self.label_pad = label_pad

    def _pad_group(self, input_ids_list, labels_list):
        """右 padding：input_ids 填 pad_id，labels 填 -100，attention_mask 真实=1/pad=0。"""
        batch_size = len(input_ids_list)
        max_len = max(ids.size(0) for ids in input_ids_list)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.label_pad, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            length = ids.size(0)
            input_ids[i, :length] = ids
            labels[i, :length] = lbl
            attention_mask[i, :length] = 1
        return input_ids, labels, attention_mask

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])  # [B,3,H,W]

        c_ids, c_labels, c_attn = self._pad_group(
            [item["chosen_input_ids"] for item in batch],
            [item["chosen_labels"] for item in batch],
        )
        r_ids, r_labels, r_attn = self._pad_group(
            [item["rejected_input_ids"] for item in batch],
            [item["rejected_labels"] for item in batch],
        )

        # chosen / rejected 共用同一张 image
        return {
            "chosen": {
                "images": images,
                "input_ids": c_ids,
                "labels": c_labels,
                "attention_mask": c_attn,
            },
            "rejected": {
                "images": images,
                "input_ids": r_ids,
                "labels": r_labels,
                "attention_mask": r_attn,
            },
        }
