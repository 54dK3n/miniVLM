"""DPO 训练循环（工程骨架）。

工程部分已搭好：数据加载、policy/reference 两模型、优化器、训练循环、日志、checkpoint。
核心数学（sequence_logprob / dpo_loss）在 dpo/dpo_loss.py 里留作待实现——
实现它们之后本脚本即可端到端跑。

数据流：
    dpo_train.json --DPODataset--> {image, chosen/rejected input_ids+labels}
      --DPOCollator--> {chosen:{...}, rejected:{...}}
      --dpo_step(policy, reference)--> loss
    policy 从 SFT checkpoint 初始化；reference = deepcopy(policy) 后 freeze_。
"""

import argparse
import copy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tokenizer import build_tokenizer
from tinyvlm.train_vlm import build_model

from .dpo_dataset import DPODataset
from .dpo_collator import DPOCollator
from .dpo_loss import dpo_step, freeze_


def build_policy(image_size, max_text_len, vocab_size, sft_ckpt=None, device="cpu"):
    """policy = SFT 模型。给了 sft_ckpt 就从 SFT 权重初始化（DPO 通常从 SFT 继续）。"""
    model = build_model(image_size, max_text_len, vocab_size)
    if sft_ckpt:
        state = torch.load(sft_ckpt, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device)


def train_dpo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(args.tokenizer)
    dataset = DPODataset(
        args.dpo_data, tokenizer,
        image_root=args.image_root, image_size=args.image_size, max_len=args.max_text_len,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=DPOCollator(pad_id=tokenizer.pad_token_id),
    )

    # 两个模型：policy（可训练，从 SFT 初始化）+ reference（policy 的冻结副本）
    policy = build_policy(args.image_size, args.max_text_len, tokenizer.vocab_size,
                          sft_ckpt=args.sft_ckpt, device=device)
    reference = freeze_(copy.deepcopy(policy)).to(device)

    optimizer = AdamW(policy.parameters(), lr=args.lr)   # 只优化 policy

    print(f"device={device} pairs={len(dataset)} steps/epoch={len(loader)} beta={args.beta}")
    policy.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            # batch 里的 tensor 搬到 device
            for group in ("chosen", "rejected"):
                for key in ("images", "input_ids", "labels", "attention_mask"):
                    batch[group][key] = batch[group][key].to(device)

            loss, metrics = dpo_step(policy, reference, batch, beta=args.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_interval == 0:
                print(f"epoch {epoch} step {step} loss={metrics['loss']:.4f} "
                      f"acc={metrics['reward_accuracy']:.2f} margin={metrics['reward_margin']:+.3f}")

    # TODO: 保存 checkpoint（logs/dpo/, checkpoints/dpo/）；DPO 前后幻觉/偏好对比评测
    return policy


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal DPO training (skeleton)")
    p.add_argument("--dpo_data", default="data/processed/dpo_train.json")
    p.add_argument("--image_root", default="data/raw/flickr8k/Images")
    p.add_argument("--sft_ckpt", default=None, help="SFT checkpoint，做 policy 初始化")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--tokenizer", choices=("char", "word"), default="char")
    return p.parse_args()


if __name__ == "__main__":
    train_dpo(parse_args())
