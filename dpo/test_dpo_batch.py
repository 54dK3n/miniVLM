"""DPO 数据/collator 最小测试。

验证：chosen/rejected tokenize、prompt 段 -100、answer 段保留、padding 段 -100、
attention_mask、chosen/rejected 同 image、chosen != rejected、shift 后 loss 只来自 answer。
"""

import json

import torch

from tokenizer.tokenizer import CharTokenizer

from .dpo_dataset import DPODataset
from .dpo_collator import DPOCollator

IMAGE_SHAPE = (3, 8, 8)
PROMPT_0 = "User: Describe this image.\nAssistant:"
CHOSEN_0 = "a red car ."


def _make(tmp_path, max_len=64):
    pairs = [
        {"image": "a.jpg", "prompt": "Describe this image.",
         "chosen": CHOSEN_0, "rejected": "a man is riding a bike on the street ."},
        {"image": "b.jpg", "prompt": "Describe this image.",
         "chosen": "a small blue bird sitting on a long branch near the water .",
         "rejected": "two dogs ."},
    ]
    path = tmp_path / "dpo.json"
    path.write_text(json.dumps(pairs), encoding="utf-8")
    tokenizer = CharTokenizer()
    dataset = DPODataset(
        path, tokenizer, max_len=max_len,
        image_loader=lambda name: torch.zeros(IMAGE_SHAPE),
    )
    return tokenizer, dataset


def test_chosen_and_rejected_tokenize(tmp_path):
    tokenizer, dataset = _make(tmp_path)
    sample = dataset[0]
    for key in ("chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"):
        assert sample[key].ndim == 1 and sample[key].numel() > 0


def test_prompt_labels_are_ignored(tmp_path):
    tokenizer, dataset = _make(tmp_path)
    sample = dataset[0]
    prompt_len = len(tokenizer.encode(PROMPT_0))
    # 对齐 label：整个 prompt 段都是 -100
    assert torch.all(sample["chosen_labels"][:prompt_len] == -100)
    assert torch.all(sample["rejected_labels"][:prompt_len] == -100)


def test_answer_labels_kept_with_eos(tmp_path):
    tokenizer, dataset = _make(tmp_path)
    sample = dataset[0]
    prompt_len = len(tokenizer.encode(PROMPT_0))
    answer_labels = sample["chosen_labels"][prompt_len:]
    assert torch.all(answer_labels != -100)
    # answer 段 == answer_ids（含末尾 EOS）
    answer_ids = tokenizer.encode(" " + CHOSEN_0) + [tokenizer.eos_token_id]
    assert answer_labels.tolist() == answer_ids
    assert sample["chosen_input_ids"][-1].item() == tokenizer.eos_token_id


def test_padding_and_mask_after_collation(tmp_path):
    tokenizer, dataset = _make(tmp_path)
    collate = DPOCollator(pad_id=tokenizer.pad_token_id)
    batch = collate([dataset[0], dataset[1]])
    for key in ("chosen", "rejected"):
        group = batch[key]
        assert group["input_ids"].shape == group["labels"].shape == group["attention_mask"].shape
        pad = group["attention_mask"] == 0
        assert pad.any()                                   # 短样本被 pad
        assert torch.all(group["labels"][pad] == -100)     # padding label = -100
        assert torch.all(group["input_ids"][pad] == tokenizer.pad_token_id)
        real = group["attention_mask"] == 1
        assert torch.all(group["input_ids"][real] != -100)


def test_same_image_and_chosen_ne_rejected(tmp_path):
    tokenizer, dataset = _make(tmp_path)
    collate = DPOCollator(pad_id=tokenizer.pad_token_id)
    batch = collate([dataset[0], dataset[1]])
    # chosen / rejected 用同一张 image
    assert torch.equal(batch["chosen"]["images"], batch["rejected"]["images"])
    assert batch["chosen"]["images"].shape[0] == 2
    # chosen != rejected
    sample = dataset[0]
    assert not torch.equal(sample["chosen_input_ids"], sample["rejected_input_ids"])


def test_shifted_loss_only_from_answer(tmp_path):
    """next-token shift（labels[1:]）后，被监督的 token 恰好是 answer 段（含 EOS）。"""
    tokenizer, dataset = _make(tmp_path)
    sample = dataset[0]
    labels = sample["chosen_labels"]
    prompt_len = len(tokenizer.encode(PROMPT_0))

    shifted = labels[1:]                                    # 预测目标
    answer_ids = tokenizer.encode(" " + CHOSEN_0) + [tokenizer.eos_token_id]
    assert int((shifted != -100).sum()) == len(answer_ids)  # 监督数 == answer 长度
    assert torch.all(shifted[: prompt_len - 1] == -100)     # prompt 段不算
    assert torch.all(shifted[prompt_len - 1:] != -100)      # answer 段都算
