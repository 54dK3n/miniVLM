"""PPO 训练循环（工程骨架）。

一个 iteration 的数据流：
    PPOPromptDataset --PPOCollator--> {images, prompt_ids, prompt_mask}
      --generate_rollout(policy, reference, reward_model)--> buffer（old logprob/value/reward）
      --compute_rewards--> per-token reward
      --compute_gae--> advantages / returns
      --(PPO epochs) 多轮 minibatch--> ppo_policy_loss + vf_coef·ppo_value_loss - ent_coef·entropy

四个数学核心（compute_rewards / compute_gae / ppo_policy_loss / ppo_value_loss）在
ppo/ppo_core.py 里待实现——实现后本脚本即可端到端跑。policy 从 SFT checkpoint 初始化；
reference = policy 的冻结副本（KL 锚点）。
"""

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tokenizer import build_tokenizer
from tinyvlm.train_vlm import build_model

from .ppo_core import (
    clone_frozen,
    compute_gae,
    compute_rewards,
    entropy_from_logits,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    ppo_policy_loss,
    ppo_value_loss,
    values_for_text,
)
from .ppo_collator import PPOCollator
from .ppo_dataset import PPOPromptDataset
from .reward import ConstantReward, LengthReward
from .rollout import generate_rollout
from .value_head import ActorCritic

REWARD_REGISTRY = {"length": LengthReward, "constant": ConstantReward}


def build_policy(image_size, max_text_len, vocab_size, sft_ckpt=None, device="cpu"):
    """policy = SFT 模型（PPO 通常从 SFT 继续）。给了 sft_ckpt 就加载权重。"""
    model = build_model(image_size, max_text_len, vocab_size)
    if sft_ckpt:
        state = torch.load(sft_ckpt, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device)


def ppo_update(actor_critic, optimizer, buffer, advantages, returns, args):
    """在一批 rollout 上做 args.ppo_epochs 轮更新。返回最后一步的 metrics。"""
    images = buffer["images"]
    input_ids = buffer["input_ids"]
    response_mask = buffer["response_mask"]
    old_logprobs = buffer["old_logprobs"]
    old_values = buffer["old_values"]
    n_img = buffer["num_image_tokens"]
    total_len = input_ids.size(1)

    last_metrics = {}
    for _ in range(args.ppo_epochs):
        logits, values_full = actor_critic(images, input_ids)
        logprobs = logprobs_from_logits(logits, input_ids, n_img)
        values = values_for_text(values_full, n_img, total_len)
        entropy = masked_mean(
            entropy_from_logits(logits, n_img, total_len), response_mask
        )

        policy_loss, pg_metrics = ppo_policy_loss(
            logprobs, old_logprobs, advantages, response_mask, args.clip_range
        )
        value_loss, vf_metrics = ppo_value_loss(
            values, old_values, returns, response_mask, args.clip_range
        )
        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.grad_clip)
        optimizer.step()

        last_metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            **pg_metrics,
            **vf_metrics,
        }
    return last_metrics


def train_ppo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(args.tokenizer)
    dataset = PPOPromptDataset(
        args.ppo_data, tokenizer,
        image_root=args.image_root, image_size=args.image_size,
        max_prompt_len=args.max_prompt_len,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=PPOCollator(pad_id=tokenizer.pad_token_id),
    )

    policy = build_policy(args.image_size, args.max_text_len, tokenizer.vocab_size,
                          sft_ckpt=args.sft_ckpt, device=device)
    actor_critic = ActorCritic(policy).to(device)
    reference = clone_frozen(policy).to(device)        # KL 锚点
    reward_model = REWARD_REGISTRY[args.reward]()

    optimizer = AdamW(actor_critic.parameters(), lr=args.lr)

    print(f"device={device} prompts={len(dataset)} iters/epoch={len(loader)} "
          f"reward={args.reward} kl_coef={args.kl_coef}")
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch["images"] = batch["images"].to(device)
            batch["prompt_ids"] = batch["prompt_ids"].to(device)
            batch["prompt_mask"] = batch["prompt_mask"].to(device)

            # 1) rollout（no_grad，采样 + 记录 old 量）
            buffer = generate_rollout(
                actor_critic, reference, tokenizer, batch, reward_model,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                do_sample=True,
            )
            # 2) 奖励整形 + GAE（用 old 量，no_grad）
            with torch.no_grad():
                rewards, reward_metrics = compute_rewards(
                    buffer["scores"], buffer["old_logprobs"], buffer["ref_logprobs"],
                    buffer["response_mask"], kl_coef=args.kl_coef,
                )
                advantages, returns = compute_gae(
                    rewards, buffer["old_values"], buffer["response_mask"],
                    gamma=args.gamma, lam=args.lam,
                )
                advantages = masked_whiten(advantages, buffer["response_mask"])
            # 3) PPO 多轮更新
            metrics = ppo_update(actor_critic, optimizer, buffer, advantages, returns, args)

            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} "
                      f"score={buffer['scores'].mean().item():+.3f} "
                      f"kl={reward_metrics.get('mean_kl', float('nan')):.4f} "
                      f"loss={metrics.get('loss', float('nan')):.4f} "
                      f"pg={metrics.get('policy_loss', float('nan')):.4f} "
                      f"vf={metrics.get('value_loss', float('nan')):.4f}")

    # TODO: 保存 checkpoint（logs/ppo/, checkpoints/ppo/）；PPO 前后奖励/幻觉对比评测
    return actor_critic


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal PPO / RLHF training (skeleton)")
    p.add_argument("--ppo_data", default="data/processed/ppo_prompts.json")
    p.add_argument("--image_root", default="data/raw/flickr8k/Images")
    p.add_argument("--sft_ckpt", default=None, help="SFT checkpoint，做 policy 初始化")
    p.add_argument("--reward", choices=tuple(REWARD_REGISTRY), default="length")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--max_text_len", type=int, default=96)
    p.add_argument("--max_prompt_len", type=int, default=48)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ppo_epochs", type=int, default=4, help="每批 rollout 的更新轮数")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.05)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer", choices=("char", "word"), default="char")
    return p.parse_args()


if __name__ == "__main__":
    train_ppo(parse_args())
