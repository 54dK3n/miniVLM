"""PPO 核心的规格测试（TDD spec）。

现在四个核心是 NotImplementedError，所以这些测试会红；按 ppo_core.py 里的公式实现后
应全部转绿。它们同时也是对"正确实现"的精确定义，实现时对照着写即可。

    pytest ppo/test_ppo_core.py -q
"""

import torch

from .ppo_core import (
    compute_gae,
    compute_rewards,
    ppo_policy_loss,
    ppo_value_loss,
)


def test_compute_rewards_places_score_at_last_action_token():
    # B=1, T=3，全是 action。kl = logprobs - ref = [0.1, 0.2, 0.3]
    logprobs = torch.tensor([[0.1, 0.2, 0.3]])
    ref = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    scores = torch.tensor([5.0])
    kl_coef = 0.1

    rewards, _ = compute_rewards(scores, logprobs, ref, mask, kl_coef=kl_coef)

    expected = -kl_coef * torch.tensor([[0.1, 0.2, 0.3]])
    expected[0, -1] += 5.0                      # score 只加在最后一个 action token
    assert torch.allclose(rewards, expected, atol=1e-6)


def test_compute_rewards_score_at_eos_when_response_shorter_than_T():
    # response_mask 只前两个是 action，score 应加在 index=1（最后一个 1）而非 index=2
    logprobs = torch.zeros(1, 3)
    ref = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    scores = torch.tensor([2.0])

    rewards, _ = compute_rewards(scores, logprobs, ref, mask, kl_coef=0.0)

    assert torch.isclose(rewards[0, 1], torch.tensor(2.0), atol=1e-6)
    assert torch.isclose(rewards[0, 2], torch.tensor(0.0), atol=1e-6)


def test_compute_gae_reduces_to_reward_to_go_when_values_zero_gamma_lambda_one():
    # values=0, gamma=lam=1 时：A_t = Σ_{k>=t} r_k（reward-to-go），returns==A
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)

    adv, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)

    assert torch.allclose(adv, torch.tensor([[6.0, 5.0, 3.0]]), atol=1e-5)
    assert torch.allclose(returns, adv, atol=1e-5)   # values=0 -> returns==adv


def test_ppo_policy_loss_equals_neg_mean_advantage_at_ratio_one():
    # logprobs == old_logprobs -> ratio=1 -> loss = -mean(advantage)（mask 内）
    logprobs = torch.tensor([[0.0, -1.0, -2.0]])
    old = logprobs.clone()
    adv = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.ones(1, 3)

    loss, _ = ppo_policy_loss(logprobs, old, adv, mask, clip_range=0.2)

    assert torch.isclose(loss, torch.tensor(-2.0), atol=1e-6)


def test_ppo_value_loss_is_half_mse_at_values_equal_old():
    # values == old_values -> v_clipped == values -> loss = 0.5*mean((v-returns)^2)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    old_values = values.clone()
    returns = torch.tensor([[1.0, 0.0, 3.0]])
    mask = torch.ones(1, 3)

    loss, _ = ppo_value_loss(values, old_values, returns, mask, clip_range=0.2)

    # 逐元素 (0,4,0) -> mean=4/3 -> *0.5
    assert torch.isclose(loss, torch.tensor(4.0 / 3.0 * 0.5), atol=1e-6)
