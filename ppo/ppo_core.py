"""PPO 核心：奖励整形 + GAE + clipped policy/value loss。

本文件分两类（与 dpo/dpo_loss.py 相同风格）：
  - 工程 glue（已实现）：logprob/value 的 next-token 对齐与 gather、entropy、
    masked_mean/masked_whiten、freeze_。
  - 数学核心（留给你手写，见 NotImplementedError）：
      compute_rewards / compute_gae / ppo_policy_loss / ppo_value_loss

约定：所有"逐 token"张量形状都是 [B, T]，其中 T 是**文本序列长度**（已剥掉视觉
prefix，见下方对齐说明）。response_mask[b,t]=1 表示 t 是策略采样出来的 action token
（prompt / padding / EOS 之后 = 0），loss 与统计都只在 mask=1 处生效。

============================ 核心数学公式 ============================

记号：一条 rollout = (image x, prompt, response y=(a_1..a_L))；πθ=policy，
πref=frozen reference（提供 KL 锚点），V=critic（value head）。

--- (A) next-token / 视觉 prefix 对齐（已在 glue 中实现，供理解）---

    模型输出 logits/hidden = [B, N_img+T, ·]；绝对位置 p 预测位置 p+1 的 token；
    文本第 t 个 token 在绝对位置 N_img+t，由 N_img+t-1 处输出预测。故取
        text_slice = full[:, N_img-1 : N_img+T-1]     # [B, T, ·]
    logprob:  logp_t = log_softmax(text_logits_t)[input_ids_t]
    value:    V_t    = value_head(hidden)_t（同一 shift，估计"产出 a_t 前"的状态价值）

--- (B) 奖励整形 compute_rewards（per-token reward）---

    RM 只给整条 response 一个标量 score（放在最后一个 response token 上）；
    其余每个 action token 上放 KL 惩罚，防止 policy 偏离 reference 太远：

        kl_t   = logπθ(a_t) - logπref(a_t)                    # 逐 token KL（近似）
        r_t    = -kl_coef * kl_t                              # 每步 KL 惩罚
        r_{t*} += score                                       # t* = 该样本最后一个 action token
        （所有 r_t 只在 response_mask=1 处有意义）

--- (C) 优势估计 compute_gae（Generalized Advantage Estimation）---

    从后往前递推（δ 是 TD-error，A 是 GAE，只在 response 内）：

        δ_t = r_t + γ · V_{t+1} - V_t
        A_t = δ_t + γ · λ · A_{t+1}          （序列末尾 A_{L+1}=0, V_{L+1}=0）
        return_t = A_t + V_t                 （value 回归目标）

    随后通常对 A 做 masked whiten（(A-mean)/std，只统计 mask=1）稳定训练。

--- (D) PPO clipped policy loss ---

    ratio_t = exp( logπθ(a_t) - logπθ_old(a_t) )
    L^policy = - E_t[ min( ratio_t · A_t,
                           clip(ratio_t, 1-ε, 1+ε) · A_t ) ]     # 只在 mask=1 上平均

--- (E) clipped value loss ---

    V_clip_t = V_old_t + clip(V_t - V_old_t, -ε_v, +ε_v)
    L^value  = 0.5 · E_t[ max( (V_t - return_t)^2,
                               (V_clip_t - return_t)^2 ) ]

总损失（在 train_ppo.py 里组合）：
    L = L^policy + vf_coef · L^value - ent_coef · entropy

======================================================================
"""

import copy

import torch
import torch.nn.functional as F


# ----------------------------- 工程 glue（已实现） -----------------------------

def freeze_(model):
    """把模型设为 reference：eval + 全部参数 requires_grad=False（KL 锚点，不更新）。"""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def clone_frozen(model):
    """深拷贝一份并冻结，作为 KL reference。"""
    return freeze_(copy.deepcopy(model))


def _text_slice(full, num_image_tokens, text_len):
    """公式 (A) 的对齐切片：full[B, N_img+T, ·] -> [B, T, ·]（左移一位，剥视觉 prefix）。"""
    start = num_image_tokens - 1
    return full[:, start:start + text_len]


def logprobs_from_logits(logits, input_ids, num_image_tokens):
    """逐 token log prob，返回 [B, T]。见公式 (A)。

    logits: [B, N_img+T, V]；input_ids: [B, T]（纯文本 token）。
    """
    text_len = input_ids.size(1)
    text_logits = _text_slice(logits, num_image_tokens, text_len)      # [B, T, V]
    logp = F.log_softmax(text_logits, dim=-1)
    return logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)         # [B, T]


def values_for_text(values_full, num_image_tokens, text_len):
    """把 critic 的逐位置 value [B, N_img+T] 对齐到文本 token -> [B, T]。见公式 (A)。"""
    return _text_slice(values_full, num_image_tokens, text_len)        # [B, T]


def entropy_from_logits(logits, num_image_tokens, text_len):
    """逐 token 分布熵，返回 [B, T]（用于 entropy bonus / 监控）。"""
    text_logits = _text_slice(logits, num_image_tokens, text_len)      # [B, T, V]
    logp = F.log_softmax(text_logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)                                     # [B, T]


def masked_mean(values, mask):
    """只在 mask=1 处求均值（标量）。mask 为 0/1 或 bool。"""
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def masked_whiten(values, mask, eps=1e-8):
    """masked 白化：(v-mean)/std，只用 mask=1 的元素统计（mask=0 处原样返回）。"""
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    mean = (values * mask_f).sum() / denom
    var = ((values - mean) ** 2 * mask_f).sum() / denom
    return (values - mean) / (var.sqrt() + eps)


# ----------------------------- 数学核心（留给你手写） -----------------------------

def compute_rewards(scores, logprobs, ref_logprobs, response_mask, kl_coef=0.05):
    """奖励整形：把 RM 标量 score + 逐 token KL 惩罚拼成 per-token reward。见公式 (B)。

    输入：
      scores:        [B]        每条 response 的标量奖励（reward model / 启发式给的）
      logprobs:      [B, T]     πθ_old 在采样时的逐 token logprob（rollout 记录）
      ref_logprobs:  [B, T]     πref（冻结）的逐 token logprob
      response_mask: [B, T]     action token=1，其余=0

    要求返回：
      rewards: [B, T]  逐 token 奖励（KL 惩罚遍布每个 action token，
                       score 只加在每条样本"最后一个 mask=1"的位置上）
      metrics: dict    建议含 mean_kl、mean_score（.item()，仅监控）

    实现提示：
      kl = logprobs - ref_logprobs                        # [B, T]
      rewards = -kl_coef * kl * response_mask
      last_idx = response_mask 每行最后一个 1 的列索引     # 见 torch 技巧下方
      rewards[b, last_idx[b]] += scores[b]
      # 找每行最后一个 1：idx = (response_mask * arange(T)).argmax(dim=1)
    """
    raise NotImplementedError("compute_rewards: 按公式 (B) 手写")


def compute_gae(rewards, values, response_mask, gamma=1.0, lam=0.95):
    """GAE：从后往前递推得到 advantages 与 value 回归目标 returns。见公式 (C)。

    输入：
      rewards:       [B, T]   compute_rewards 的输出
      values:        [B, T]   critic 对每个 action token 的 value 估计（old/detach）
      response_mask: [B, T]
      gamma, lam:    折扣与 GAE 系数

    要求返回：
      advantages: [B, T]
      returns:    [B, T]      （= advantages + values，作 value loss 的回归目标）

    实现提示（逐时间步反向循环，t 从 T-1 到 0）：
      next_value = values[:, t+1]（t=T-1 时取 0）
      delta = rewards[:, t] + gamma * next_value - values[:, t]
      last_gae = delta + gamma * lam * last_gae      # 初始 last_gae=0
      advantages[:, t] = last_gae
      # 用 response_mask 保证 padding / 非 action 位置不泄漏（可乘 mask 或跳过）
      returns = advantages + values
    """
    raise NotImplementedError("compute_gae: 按公式 (C) 手写")


def ppo_policy_loss(logprobs, old_logprobs, advantages, response_mask, clip_range=0.2):
    """PPO clipped surrogate policy loss。见公式 (D)。

    输入：
      logprobs:      [B, T]   当前 πθ 的逐 token logprob（带梯度）
      old_logprobs:  [B, T]   采样时 πθ_old 的 logprob（detach）
      advantages:    [B, T]   compute_gae 输出（通常已 masked_whiten）
      response_mask: [B, T]
      clip_range:    ε

    要求返回：(loss 标量, metrics dict)
      metrics 建议含 clip_frac、approx_kl（.item()）。

    实现提示：
      ratio = (logprobs - old_logprobs).exp()
      unclipped = -advantages * ratio
      clipped   = -advantages * ratio.clamp(1-clip_range, 1+clip_range)
      loss = masked_mean(torch.max(unclipped, clipped), response_mask)
      approx_kl = masked_mean(old_logprobs - logprobs, response_mask)
      clip_frac = masked_mean((ratio 超出 clip 区间).float(), response_mask)
    """
    raise NotImplementedError("ppo_policy_loss: 按公式 (D) 手写")


def ppo_value_loss(values, old_values, returns, response_mask, clip_range=0.2):
    """clipped value (critic) loss。见公式 (E)。

    输入：
      values:        [B, T]   当前 critic 输出（带梯度）
      old_values:    [B, T]   rollout 时的 value（detach）
      returns:       [B, T]   compute_gae 的回归目标
      response_mask: [B, T]
      clip_range:    ε_v

    要求返回：(loss 标量, metrics dict)（建议含 explained_var 或 value_error）。

    实现提示：
      v_clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
      loss_unclipped = (values    - returns) ** 2
      loss_clipped   = (v_clipped - returns) ** 2
      loss = 0.5 * masked_mean(torch.max(loss_unclipped, loss_clipped), response_mask)
    """
    raise NotImplementedError("ppo_value_loss: 按公式 (E) 手写")
