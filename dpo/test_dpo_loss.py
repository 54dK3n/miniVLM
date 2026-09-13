"""DPO logprob / loss 测试，重点验证方向（符号）与 reference 冻结。"""

import copy
import json
import math

import torch

from tokenizer.tokenizer import CharTokenizer
from tinyvlm.train_vlm import build_model

from .dpo_dataset import DPODataset
from .dpo_collator import DPOCollator
from .dpo_loss import sequence_logprob, dpo_loss, dpo_step, freeze_


def _batch(tmp_path):
    pairs = [
        {"image": "a.jpg", "prompt": "Describe this image.",
         "chosen": "a red car .", "rejected": "a man is riding a bike ."},
        {"image": "b.jpg", "prompt": "Describe this image.",
         "chosen": "a blue bird on a branch .", "rejected": "two dogs ."},
    ]
    path = tmp_path / "dpo.json"
    path.write_text(json.dumps(pairs), encoding="utf-8")
    tokenizer = CharTokenizer()
    dataset = DPODataset(path, tokenizer, max_len=48,
                         image_loader=lambda name: torch.zeros(3, 32, 32))
    batch = DPOCollator(pad_id=tokenizer.pad_token_id)([dataset[0], dataset[1]])
    return tokenizer, batch


def _model(tokenizer):
    torch.manual_seed(0)
    return build_model(image_size=32, max_text_len=48, vocab_size=tokenizer.vocab_size)


def test_sequence_logprob_shape_and_sign(tmp_path):
    tokenizer, batch = _batch(tmp_path)
    model = _model(tokenizer)
    chosen = batch["chosen"]
    logps = sequence_logprob(model, chosen["images"], chosen["input_ids"], chosen["labels"])
    assert logps.shape == (2,)
    assert torch.isfinite(logps).all()
    assert torch.all(logps <= 0)            # 一串 logprob 之和必为非正


def test_reference_is_frozen(tmp_path):
    tokenizer, _ = _batch(tmp_path)
    reference = freeze_(_model(tokenizer))
    assert all(not p.requires_grad for p in reference.parameters())


def test_loss_equals_log2_when_policy_equals_reference():
    # policy == reference -> logits=0 -> loss = -log σ(0) = log 2，reward margin = 0
    chosen = torch.tensor([-5.0, -8.0])
    rejected = torch.tensor([-7.0, -6.0])
    loss, metrics = dpo_loss(chosen, rejected, chosen.clone(), rejected.clone(), beta=0.1)
    assert abs(loss.item() - math.log(2.0)) < 1e-5
    assert abs(metrics["reward_margin"]) < 1e-6


def test_loss_sign_direction():
    # ref 固定；chosen 相对 ref 提升 > rejected -> loss 小；反过来 -> loss 大（符号不能写反）
    ref_c, ref_r = torch.tensor([-5.0]), torch.tensor([-5.0])
    good, _ = dpo_loss(torch.tensor([-3.0]), torch.tensor([-9.0]), ref_c, ref_r, beta=0.1)
    bad, _ = dpo_loss(torch.tensor([-9.0]), torch.tensor([-3.0]), ref_c, ref_r, beta=0.1)
    assert good.item() < bad.item()


def test_beta_scales_loss():
    # rejected 被偏好（坏方向）时，beta 越大惩罚越强
    ref_c, ref_r = torch.tensor([-5.0]), torch.tensor([-5.0])
    small = dpo_loss(torch.tensor([-9.0]), torch.tensor([-3.0]), ref_c, ref_r, beta=0.1)[0]
    large = dpo_loss(torch.tensor([-9.0]), torch.tensor([-3.0]), ref_c, ref_r, beta=1.0)[0]
    assert large.item() > small.item()


def test_dpo_step_grad_flows_to_policy_only(tmp_path):
    tokenizer, batch = _batch(tmp_path)
    policy = _model(tokenizer)
    reference = freeze_(copy.deepcopy(policy))

    loss, metrics = dpo_step(policy, reference, batch, beta=0.1)
    assert torch.isfinite(loss)
    assert set(metrics) >= {"loss", "reward_margin", "reward_accuracy"}

    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())     # policy 更新
    assert all(p.grad is None for p in reference.parameters())      # reference 不更新
