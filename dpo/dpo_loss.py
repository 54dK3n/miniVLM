"""DPO 核心：response 序列 log-probability + 两模型(policy/reference) + DPO loss。

本文件分两类：
  - 工程 glue（已实现）：freeze_（冻结 reference）、dpo_step（串起两模型 + 4 组 logprob）。
  - 数学核心（留给你手写，见 NotImplementedError）：sequence_logprob、dpo_loss。

============================ 核心数学公式 ============================

记号：x = (image, prompt)，y = answer = (y_1, ..., y_L)；πθ = policy，πref = reference。

--- (A) response(answer-only) 序列 log prob ---

    log π(y | x) = Σ_{t=1..L}  log π(y_t | x, y_<t)
                 = Σ_{t∈answer}  log softmax(logits_t)[y_t]

  只对 answer token 求和；prompt / padding 的 label = -100，不计入。
  注意 next-token 对齐：绝对位置 p 的 logits 预测位置 p+1 的 token；
  又因为前面拼了 N_img 个 visual prefix，文本第 t 个 token 在绝对位置 N_img+t，
  由 N_img+t-1 处的 logits 预测。

--- (B) 隐式 reward 与 DPO loss（Rafailov et al. 2023）---

    r(y) = β · ( log πθ(y|x) - log πref(y|x) )          # 隐式 reward

    L_DPO = - E[ log σ( r(y_w) - r(y_l) ) ]
          = - E[ log σ( β · ( (logπθ(y_w) - logπref(y_w))
                              - (logπθ(y_l) - logπref(y_l)) ) ) ]

  其中 y_w = chosen（好），y_l = rejected（坏），σ = sigmoid。

  关键性质：
  - policy 参数被更新；reference 冻结、不更新（提供锚点，防止 policy 跑飞）。
  - logprob 只对 answer 部分算（prompt/padding 不算）。
  - 符号方向：chosen 比 rejected 更被偏好 -> 括号 > 0 -> σ→1 -> loss→0。写反就把好坏学反。
  - β 越大，偏好信号越强（惩罚/奖励放大）。

======================================================================
"""

import torch
import torch.nn.functional as F


# ----------------------------- 工程 glue（已实现） -----------------------------

def freeze_(model):
    """把模型设为 reference：eval + 全部参数 requires_grad=False（不参与更新）。"""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def dpo_step(policy, reference, batch, beta=0.1):
    """一个 batch 的 DPO 前向：算 4 组 answer-only logprob，再算 loss。

    batch = {"chosen": {images,input_ids,labels,...}, "rejected": {...}}（DPOCollator 输出）。
    policy 带梯度；reference 在 no_grad 下、且已 freeze_。
    依赖下面两个"数学核心"——实现它们后本函数即可用。
    """
    chosen, rejected = batch["chosen"], batch["rejected"]

    # policy：需要梯度
    policy_chosen_logps = sequence_logprob(policy, chosen["images"], chosen["input_ids"], chosen["labels"])
    policy_rejected_logps = sequence_logprob(policy, rejected["images"], rejected["input_ids"], rejected["labels"])

    # reference：冻结，不建图
    with torch.no_grad():
        ref_chosen_logps = sequence_logprob(reference, chosen["images"], chosen["input_ids"], chosen["labels"])
        ref_rejected_logps = sequence_logprob(reference, rejected["images"], rejected["input_ids"], rejected["labels"])

    return dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=beta)


# ----------------------------- 数学核心（留给你手写） -----------------------------

def sequence_logprob(model, images, input_ids, labels):
    """answer-only 序列 log prob，返回 [B]。

    见公式 (A)。实现步骤（建议）：
      1. logits, _ = model(images, input_ids)        # [B, N_img+T, V]
      2. T = input_ids.size(1)
         N_img = logits.size(1) - T                  # visual prefix 长度，从 shape 反推
                                                       # （这样线性桥接 / Q-Former 都通用）
      3. 取"预测文本 token 0..T-1"的那段 logits（左移一位对齐）：
         text_logits = logits[:, N_img-1 : N_img+T-1, :]   # [B, T, V]
      4. logp = F.log_softmax(text_logits, dim=-1)         # [B, T, V]
      5. token_logp = logp.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
      6. mask = (labels != -100)                            # answer-only（prompt/padding 屏蔽）
      7. return (token_logp * mask).sum(dim=1)              # [B]

    shape 检查：输入 input_ids/labels = [B,T]，logits = [B,N_img+T,V]，输出 = [B]。
    """
    logits, _ = model(images, input_ids)                 # [B, N_img+T, V]
    T = input_ids.size(1)
    N_img = logits.size(1) - T                            # visual prefix 长度，从 shape 反推
    text_logits = logits[:, N_img - 1: N_img + T - 1, :]  # [B, T, V]，左移一位对齐文本
    log_p = F.log_softmax(text_logits, dim=-1)
    token_logp = log_p.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    mask = (labels != -100)                               # answer-only
    return (token_logp * mask).sum(dim=1)                 # [B]


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO loss + 监控指标。四个输入都是 [B] 的序列 logprob。

    见公式 (B)。实现步骤（建议）：
      pi_logratios  = policy_chosen_logps - policy_rejected_logps
      ref_logratios = ref_chosen_logps   - ref_rejected_logps
      logits = beta * (pi_logratios - ref_logratios)
      loss   = -F.logsigmoid(logits).mean()

      # 隐式 reward（detach，仅监控）：
      chosen_reward   = beta * (policy_chosen_logps   - ref_chosen_logps).detach()
      rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
      metrics = {loss, chosen_reward.mean, rejected_reward.mean,
                 reward_margin = (chosen_reward - rejected_reward).mean,
                 reward_accuracy = (chosen_reward > rejected_reward).float().mean}

    返回 (loss 标量 tensor, metrics dict)。注意符号别写反（见公式 B 的"符号方向"）。
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    # 隐式 reward（detach，仅监控）；metrics 存 float，方便日志/JSON
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
        "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
    }
    return loss, metrics
