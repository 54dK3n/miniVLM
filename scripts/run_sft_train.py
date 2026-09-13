"""SFT 训练入口（带日志/checkpoint 持久化）。

链路：
    caption CSV --(复用 ImageCaptionDataset 的图像加载)--> (image, caption)
      --> FlickerDataset(instruction 模板 + response-only mask)
      --> SFTCollator(变长 padding)
      --> 已有的 tinyvlm.TinyVLM
      --> sft.train_sft

训练后落盘：
    checkpoints/sft/<name>.pt   模型权重 + tokenizer
    logs/sft/<name>.json        loss 曲线 + 图像消融 + 生成样例

用法：
    python scripts/run_sft_train.py --image_root data/raw/flickr8k/Images \
        --caption_file data/raw/flickr8k/captions.txt --name flickr8k_sft
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from tokenizer import build_tokenizer
from tinyvlm.dataset import ImageCaptionDataset, read_caption_file, split_by_image
from tinyvlm.train_vlm import build_model

from sft.sft_dataset import FlickerDataset
from sft.sft_collator import SFTCollator
from sft.train_sft import train_sft, evaluate_sft, load_pretrained


class _PairView(Dataset):
    """把 ImageCaptionDataset 的 dict 适配成 FlickerDataset 需要的 (image, caption)。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        return sample["image"], sample["caption"]


@torch.no_grad()
def greedy_answer(model, image, prompt_ids, max_new, eos_id, device):
    model.eval()
    ids = prompt_ids.to(device)
    for _ in range(max_new):
        logits, _ = model(image.to(device), ids)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt) == eos_id:
            break
    return ids[:, prompt_ids.size(1):]


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal SFT for TinyVLM")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--name", default="flickr8k_sft", help="checkpoint/log 文件名")
    parser.add_argument("--pretrained", default=None, help="caption 预训练 checkpoint")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit_train", type=int, default=1500)
    parser.add_argument("--limit_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--tokenizer", choices=("char", "word"), default="char")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = read_caption_file(args.caption_file)
    train_idx, val_idx, _, _ = split_by_image(samples, val_ratio=0.1, seed=args.seed)
    if args.limit_train:
        train_idx = train_idx[: args.limit_train]
    if args.limit_val:
        val_idx = val_idx[: args.limit_val]

    tokenizer = build_tokenizer(
        args.tokenizer, texts=(samples[i][1] for i in train_idx)
    )
    base = ImageCaptionDataset(
        args.image_root, args.caption_file, tokenizer,
        image_size=args.image_size, max_text_len=args.max_text_len,
    )
    collate = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    train_ds = FlickerDataset(_PairView(Subset(base, train_idx)), tokenizer, max_len=args.max_text_len)
    val_ds = FlickerDataset(_PairView(Subset(base, val_idx)), tokenizer, max_len=args.max_text_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = build_model(args.image_size, args.max_text_len, tokenizer.vocab_size)
    if args.pretrained:
        load_pretrained(model, args.pretrained, device=device, strict=False)

    print(f"device={device} vocab={tokenizer.vocab_size} train_rows={len(train_ds)} "
          f"val_rows={len(val_ds)} steps/epoch={len(train_loader)}")

    t0 = time.time()
    history = train_sft(
        model, train_loader, epochs=args.epochs, lr=args.lr,
        grad_clip=args.grad_clip, device=device, log_interval=args.log_interval,
    )
    train_time = time.time() - t0

    # 图像消融：normal / shuffled / zero，验证模型是否真的用图像
    shuffle_tf = lambda x: x[torch.randperm(x.size(0), device=x.device)]
    zero_tf = lambda x: torch.zeros_like(x)
    ablation = {}
    for split, loader in (("train", train_loader), ("val", val_loader)):
        normal = evaluate_sft(model, loader, device)
        shuffled = evaluate_sft(model, loader, device, image_transform=shuffle_tf)
        zeroed = evaluate_sft(model, loader, device, image_transform=zero_tf)
        ablation[split] = {
            "normal_loss": normal, "shuffled_loss": shuffled, "zero_loss": zeroed,
            "shuffle_gap": shuffled - normal, "zero_gap": zeroed - normal,
        }
        print(f"[{split}] normal={normal:.4f} shuffled={shuffled:.4f} zero={zeroed:.4f} "
              f"(shuffle_gap={shuffled-normal:+.4f})")

    # 生成样例
    prompt = f"User: {train_ds.instructions[0]}\nAssistant:"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    num_img = (args.image_size // 8) ** 2 + 1
    budget = model.gpt.max_seq_len - num_img - prompt_ids.size(1) - 1
    samples_out = []
    seen = set()
    for i in range(len(val_ds)):
        img, cap = val_ds.base_dataset[i]
        if cap[:20] in seen:
            continue
        seen.add(cap[:20])
        gen = tokenizer.decode(
            greedy_answer(model, img.unsqueeze(0), prompt_ids, budget, tokenizer.eos_token_id, device)[0].tolist()
        )
        samples_out.append({"target": cap.lower(), "generated": gen})
        if len(samples_out) >= 4:
            break

    # 落盘 checkpoint + log
    ckpt_path = Path("checkpoints/sft") / f"{args.name}.pt"
    log_path = Path("logs/sft") / f"{args.name}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_type": tokenizer.tokenizer_type,
            "config": vars(args),
        },
        ckpt_path,
    )
    metrics = {
        "dataset": "Flickr8k (SFT, instruction->caption)",
        "checkpoint": str(ckpt_path),
        "config": vars(args),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "train_time_sec": round(train_time, 1),
        "loss_curve": [round(history[min(len(history) - 1, (e + 1) * max(1, len(history) // args.epochs) - 1)], 4)
                       for e in range(args.epochs)],
        "first_loss": round(history[0], 4),
        "last_loss": round(history[-1], 4),
        "image_ablation": ablation,
        "samples": samples_out,
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"saved checkpoint -> {ckpt_path}")
    print(f"saved metrics    -> {log_path}")


if __name__ == "__main__":
    main()
