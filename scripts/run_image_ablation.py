import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.dataset import ImageCaptionDataset, make_vlm_collate_fn, split_by_image
from tinyvlm.train_vlm import build_model, validate
from tokenizer import tokenizer_from_checkpoint


def evaluate_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", args.image_size)
    max_text_len = config.get("max_text_len", args.max_text_len)
    val_ratio = config.get("val_ratio", args.val_ratio)
    seed = config.get("seed", args.seed)

    tokenizer = tokenizer_from_checkpoint(checkpoint)
    dataset = ImageCaptionDataset(
        args.image_root,
        args.caption_file,
        tokenizer,
        image_size=image_size,
        max_text_len=max_text_len,
    )
    _, val_indices, _, val_images = split_by_image(
        dataset.samples, val_ratio=val_ratio, seed=seed
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_vlm_collate_fn(tokenizer),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        image_size,
        max_text_len,
        tokenizer.vocab_size,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    val_loss, shuffle_loss, zero_loss, noise_loss = validate(
        model, val_loader, device, amp_enabled, amp_dtype
    )

    report = {
        "dataset": args.dataset_name,
        "checkpoint": str(args.checkpoint),
        "validation_images": len(val_images),
        "val_loss": val_loss,
        "val_shuffle_loss": shuffle_loss,
        "val_zero_loss": zero_loss,
        "val_noise_loss": noise_loss,
        "shuffle_gap": shuffle_loss - val_loss,
        "zero_gap": zero_loss - val_loss,
        "noise_gap": noise_loss - val_loss,
        "noise_distribution": "uniform[0,1]",
    }
    print(f"checkpoint:       {args.checkpoint}")
    print(f"validation images:{len(val_images)}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_shuffle_loss: {shuffle_loss:.6f}")
    print(f"val_zero_loss:    {zero_loss:.6f}")
    print(f"val_noise_loss:   {noise_loss:.6f}")
    print(f"shuffle_gap:      {shuffle_loss - val_loss:+.6f}")
    print(f"zero_gap:         {zero_loss - val_loss:+.6f}")
    print(f"noise_gap:        {noise_loss - val_loss:+.6f}")
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report:            {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyVLM image ablations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--max_text_len", type=int, default=96)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="dataset")
    parser.add_argument("--report_path")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_ablation(parse_args())
