"""Image-instruction-answer SFT 数据集。

把 (image, caption) 的 base_dataset 包成 image-instruction-answer 监督样本：
套 instruction prompt 模板 + response-only label mask（只对 answer 算 loss）。
"""

import torch
from torch.utils.data import Dataset


class FlickerDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, max_len=64):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.instructions = [
            "Describe this image.",
            "What is happening in the image?",
            "Write a short caption for this image.",
            "What can you see in the picture?",
        ]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, caption = self.base_dataset[index]
        instruction = self.instructions[index % len(self.instructions)]
        prompt = f"User: {instruction}\nAssistant:"
        answer = " " + caption

        # prompt / answer 分开 encode，再拼 token id：避免依赖
        # encode(prompt+answer)==encode(prompt)+encode(answer) 这个脆弱的 prefix 假设。
        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer) + [self.tokenizer.eos_token_id]

        # 分段截断：先保证 prompt 不占满（给 answer 留 >=1 token + EOS），
        # 再截 answer 时把 EOS 接回末尾，保证“答案以 EOS 结尾”不丢、且 answer 段非空（不会 NaN）。
        if len(prompt_ids) > self.max_len - 2:
            prompt_ids = prompt_ids[: self.max_len - 2]
        budget = self.max_len - len(prompt_ids)
        if len(answer_ids) > budget:
            answer_ids = answer_ids[: budget - 1] + [self.tokenizer.eos_token_id]

        full = prompt_ids + answer_ids

        # 本仓库约定：模型 forward 里 cross_entropy 直接 logits[t] 对齐 labels[t]，**不做** shift
        # （见 tiny_gpt.py / tinyvlm_model.py，以及 ImageCaptionDataset 的 [BOS]+cap / cap+[EOS]）。
        # 所以 shift 在数据侧做：input_ids[t] 预测 labels[t]=full[t+1]。
        input_ids = torch.tensor(full[:-1], dtype=torch.long)
        labels = torch.tensor(full[1:], dtype=torch.long)

        # response-only：labels[t] 预测 full[t+1]，只在 answer 段(t+1 >= len(prompt_ids))算 loss，
        # 等价于把前 len(prompt_ids)-1 个 label 置 -100。
        labels[: len(prompt_ids) - 1] = -100

        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "prompt": prompt,
            "caption": caption,
        }
