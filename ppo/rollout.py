"""Rollout：用当前 policy 在线采样 response，并记录 PPO 更新所需的"旧"量。

产出一个 buffer（dict），字段都对齐到**文本序列** [B, T_full]（T_full = prompt 长 +
max_new_tokens），T_full 内 response_mask=1 的位置才是 action：
    images        [B,3,H,W]
    input_ids     [B, T_full]   prompt ++ 采样出来的 response（纯文本 token）
    response_mask [B, T_full]   action token=1；prompt / EOS 之后 = 0
    old_logprobs  [B, T_full]   采样策略 πθ_old 的逐 token logprob（detach）
    ref_logprobs  [B, T_full]   冻结 reference πref 的逐 token logprob（detach）
    old_values    [B, T_full]   critic 的逐 token value（detach）
    scores        [B]           reward model 给整条 response 的标量分
    num_image_tokens (int)      视觉 prefix 长度 N_img（对齐用）
    responses     list[str]     解码后的 response 文本（打分 / 调试用）

采样用无 KV-cache 的整批 loop（教学清晰）；logprob/value 用采样结束后的一次前向重算
（因果模型下与边采样边记录数值相同，见 ppo_core 公式 (A) 对齐）。
"""

import torch

from .ppo_core import logprobs_from_logits, values_for_text


@torch.no_grad()
def _sample_tokens(actor_critic, images, prompt_ids, bos_or_none,
                   eos_token_id, max_new_tokens, temperature, do_sample):
    """从 prompt 逐 token 采样，返回 full_ids [B, Tp+max_new_tokens]。"""
    model = actor_critic.model
    model.eval()
    cur = prompt_ids
    finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        logits, _ = model(images, cur)                 # prefill，无 cache
        next_logits = logits[:, -1, :] / temperature
        if do_sample:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        # 已结束的行固定吐 EOS（后续会被 response_mask 屏蔽）
        next_token = torch.where(finished.unsqueeze(1),
                                 torch.full_like(next_token, eos_token_id),
                                 next_token)
        cur = torch.cat([cur, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(eos_token_id)
        if finished.all():
            # 补齐到定长，便于 batch 对齐
            pad_steps = max_new_tokens - (cur.size(1) - prompt_ids.size(1))
            if pad_steps > 0:
                pad = torch.full((cur.size(0), pad_steps), eos_token_id,
                                 dtype=cur.dtype, device=cur.device)
                cur = torch.cat([cur, pad], dim=1)
            break
    return cur


def _build_response_mask(sampled, eos_token_id):
    """response 段 [B, L] -> mask [B, L]：保留到"第一个 EOS（含）"，其后置 0。"""
    is_eos = sampled.eq(eos_token_id)
    eos_cumsum = is_eos.cumsum(dim=1)
    return ((eos_cumsum == 0) | (is_eos & (eos_cumsum == 1))).long()


@torch.no_grad()
def generate_rollout(actor_critic, reference, tokenizer, batch, reward_model,
                     max_new_tokens=32, temperature=1.0, do_sample=True):
    """跑一整批 rollout，返回上文所述 buffer（所有张量 detach，供 PPO 多轮更新复用）。"""
    images = batch["images"]
    prompt_ids = batch["prompt_ids"]
    prompt_len = prompt_ids.size(1)

    full_ids = _sample_tokens(
        actor_critic, images, prompt_ids,
        bos_or_none=None, eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample,
    )
    total_len = full_ids.size(1)

    # 采样结束后一次性重算 old logprob / value / ref logprob（都对齐到文本序列）
    logits, values_full = actor_critic(images, full_ids)
    num_image_tokens = logits.size(1) - total_len
    old_logprobs = logprobs_from_logits(logits, full_ids, num_image_tokens)
    old_values = values_for_text(values_full, num_image_tokens, total_len)

    ref_logits, _ = reference(images, full_ids)
    ref_logprobs = logprobs_from_logits(ref_logits, full_ids, num_image_tokens)

    # response_mask：prompt 段全 0；response 段保留到第一个 EOS
    sampled = full_ids[:, prompt_len:]
    resp_mask_sampled = _build_response_mask(sampled, tokenizer.eos_token_id)
    response_mask = torch.zeros_like(full_ids)
    response_mask[:, prompt_len:] = resp_mask_sampled

    # 解码 response 文本（截到第一个 EOS 前）用于打分
    responses = []
    for row_idx in range(sampled.size(0)):
        keep = resp_mask_sampled[row_idx].bool()
        ids = sampled[row_idx][keep].tolist()
        if ids and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
        responses.append(tokenizer.decode(ids))

    scores = reward_model.score_batch(images, batch["prompts"], responses)

    return {
        "images": images,
        "input_ids": full_ids,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "old_values": old_values,
        "scores": scores,
        "num_image_tokens": num_image_tokens,
        "responses": responses,
    }
