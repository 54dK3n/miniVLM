"""SFT prompt/answer/padding mask 测试。

约定：本仓库 label 在“数据里”左移一位，模型不再 shift。即
    input_ids = full[:-1],  labels = full[1:],  full = prompt_ids + answer_ids(+EOS)
所以 labels[t] 预测 full[t+1]；前 len(prompt_ids)-1 个 label 被置 -100。

对照 MEMORY.md 6.5：
  Test 1：prompt 部分 labels 必须是 -100
  Test 2：answer 部分 labels 必须保留 token id（含末尾 EOS）
  Test 3：padding 部分 labels 必须是 -100
  Test 4：loss 只来自 answer tokens
  Test 5：cross_entropy(logits, labels) 直接对齐（不再外部 shift），ignore_index 排除 prompt/padding。
"""

import torch

from tokenizer.tokenizer import CharTokenizer

from .sft_dataset import FlickerDataset
from .sft_collator import SFTCollator

# instructions[0]，对应 index=0 的样本
PROMPT_0 = "User: Describe this image.\nAssistant:"


class _FakeBase:
    """最小 base_dataset：返回 (image_tensor, caption)。"""

    def __init__(self, captions, image_shape=(3, 8, 8)):
        self.captions = captions
        self.image_shape = image_shape

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return torch.zeros(self.image_shape), self.captions[index]


def _make_dataset(captions, max_len=256):
    tokenizer = CharTokenizer()
    dataset = FlickerDataset(_FakeBase(captions), tokenizer, max_len=max_len)
    return tokenizer, dataset


def test_prompt_tokens_are_ignored():
    """Test 1：预测 prompt token 的位置 labels 全为 -100（shift 后是前 P-1 个）。"""
    tokenizer, dataset = _make_dataset(["a red car."])
    sample = dataset[0]
    prompt_len = len(tokenizer.encode(PROMPT_0))
    assert torch.all(sample["labels"][: prompt_len - 1] == -100)
    # 第 P-1 个位置开始就是第一个 answer token（由最后一个 prompt token 预测），不应再被 mask
    assert sample["labels"][prompt_len - 1] != -100


def test_answer_tokens_keep_their_ids():
    """Test 2：answer 段 labels 保留 token id（含末尾 EOS），且满足 labels=full[1:]。"""
    tokenizer, dataset = _make_dataset(["a red car."])
    sample = dataset[0]
    prompt_len = len(tokenizer.encode(PROMPT_0))

    # answer 段 = " " + caption 再加 EOS
    answer_ids = tokenizer.encode(" a red car.") + [tokenizer.eos_token_id]
    supervised = sample["labels"][prompt_len - 1:]
    assert torch.all(supervised != -100)
    assert supervised.tolist() == answer_ids        # 被监督的就是 answer_ids（含末尾 EOS）
    # EOS 是被预测的目标，落在 labels 末尾，不作为 input
    assert sample["labels"][-1].item() == tokenizer.eos_token_id
    assert sample["input_ids"][-1].item() != tokenizer.eos_token_id


def test_padding_tokens_are_ignored_after_collation():
    """Test 3：collator pad 后，padding 段 labels=-100、input=pad、attn=0。"""
    tokenizer, dataset = _make_dataset(
        ["short.", "a much longer caption with many more words right here."]
    )
    collate = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collate([dataset[0], dataset[1]])

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == labels.shape == attention_mask.shape
    pad_positions = attention_mask == 0
    assert pad_positions.any()  # 短样本确实被 pad 了
    assert torch.all(labels[pad_positions] == -100)
    assert torch.all(input_ids[pad_positions] == tokenizer.pad_token_id)
    assert batch["image"].shape[0] == 2  # image 正确 stack


def test_loss_tokens_only_from_answer():
    """Test 4：被监督的 token 数 == answer(含 EOS) 的 token 数。"""
    caption = "a red car on the road."
    tokenizer, dataset = _make_dataset([caption])
    sample = dataset[0]

    prompt_len = len(tokenizer.encode(PROMPT_0))
    supervised = int((sample["labels"] != -100).sum())

    # answer = " " + caption，再加一个 EOS
    expected = len(tokenizer.encode(" " + caption)) + 1
    assert supervised == expected
    # shift 后总长 = P+A-1；被监督的是 answer 段 A 个，= 总长 -(P-1)
    assert supervised == sample["input_ids"].shape[0] - (prompt_len - 1)


def test_only_answer_contributes_to_loss():
    """Test 5：按本仓库约定 cross_entropy(logits, labels) 直接对齐（不再外部 shift）。

    label 已在 dataset 里左移，模型 forward 不 shift。验证 ignore_index=-100 把
    prompt + padding 排除在 loss 之外（MEMORY 6.4 response-only loss 的核心断言）。
    """
    tokenizer, dataset = _make_dataset(["a red car."])
    collate = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collate([dataset[0]])

    labels = batch["labels"]                       # [1, T]，已是 shift 后的目标
    vocab_size = tokenizer.vocab_size
    torch.manual_seed(0)
    logits = torch.randn(1, labels.size(1), vocab_size)   # 与 labels 同长，直接对齐

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)

    num_supervised = int((labels != -100).sum())
    assert num_supervised > 0
    # 若某样本全被 mask（无 answer token），loss 会变 NaN —— dataset 的截断逻辑需防这种情况
    all_ignored = torch.full_like(labels, -100)
    nan_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        all_ignored.reshape(-1),
        ignore_index=-100,
    )
    assert torch.isnan(nan_loss)
