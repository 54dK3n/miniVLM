import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from tokenizer import build_tokenizer

from .dataset import (
    ImageCaptionDataset,
    make_vlm_collate_fn,
    read_caption_file,
    split_by_image,
)
from .multimodal_generate import generate
from .train_vlm import build_model


@torch.no_grad()
def evaluate_teacher_forcing(model, data_loader, device, amp_enabled, amp_dtype):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for images, input_ids, labels in data_loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits, loss = model(images, input_ids, labels)

        valid = labels.ne(-100)
        valid_count = valid.sum().item()
        text_logits = logits[:, -labels.size(1) :, :]  # [B,T_text,V]
        predictions = text_logits.argmax(dim=-1)
        correct_tokens += (predictions.eq(labels) & valid).sum().item()
        total_loss += loss.item() * valid_count
        total_tokens += valid_count

    mean_loss = total_loss / total_tokens
    return {
        "val_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "token_accuracy": correct_tokens / total_tokens,
    }


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(tokens, order):
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def corpus_bleu(hypotheses, references, max_order=4):
    """Compute smoothed corpus BLEU without adding an external dependency."""
    clipped = [0] * max_order
    totals = [0] * max_order
    candidate_length = 0
    reference_length = 0

    for hypothesis, reference_group in zip(hypotheses, references):
        hypothesis_tokens = _words(hypothesis)
        reference_tokens = [_words(reference) for reference in reference_group]
        candidate_length += len(hypothesis_tokens)
        reference_length += min(
            (len(tokens) for tokens in reference_tokens),
            key=lambda length: (abs(length - len(hypothesis_tokens)), length),
        )

        for order in range(1, max_order + 1):
            hypothesis_counts = _ngrams(hypothesis_tokens, order)
            max_reference_counts = Counter()
            for tokens in reference_tokens:
                reference_counts = _ngrams(tokens, order)
                for ngram, count in reference_counts.items():
                    max_reference_counts[ngram] = max(
                        max_reference_counts[ngram], count
                    )
            clipped[order - 1] += sum(
                min(count, max_reference_counts[ngram])
                for ngram, count in hypothesis_counts.items()
            )
            totals[order - 1] += sum(hypothesis_counts.values())

    if candidate_length == 0:
        return {"bleu1": 0.0, "bleu4": 0.0}

    # Add-one smoothing keeps BLEU-4 defined for this deliberately tiny model.
    precisions = [
        (match_count + 1) / (total_count + 1)
        for match_count, total_count in zip(clipped, totals)
    ]
    brevity_penalty = (
        1.0
        if candidate_length > reference_length
        else math.exp(1 - reference_length / candidate_length)
    )
    bleu1 = brevity_penalty * precisions[0]
    bleu4 = brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )
    return {"bleu1": bleu1, "bleu4": bleu4}


class ValidationImageDataset(Dataset):
    def __init__(self, caption_dataset, image_names):
        self.caption_dataset = caption_dataset
        self.image_names = sorted(image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        return self.caption_dataset._load_image(image_name), image_name


@torch.no_grad()
def evaluate_generation(
    model,
    caption_dataset,
    val_images,
    tokenizer,
    device,
    batch_size,
    num_workers,
    max_text_len,
    eval_images=0,
):
    references_by_image = defaultdict(list)
    for image_name, caption in caption_dataset.samples:
        if image_name in val_images:
            references_by_image[image_name].append(caption.lower())

    image_names = sorted(val_images)
    if eval_images > 0:
        image_names = image_names[:eval_images]
    image_dataset = ValidationImageDataset(caption_dataset, image_names)
    data_loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model.eval()
    hypotheses = []
    reference_groups = []
    examples = []
    for images, names in data_loader:
        images = images.to(device, non_blocking=True)
        generated_ids = generate(
            model,
            images,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            max_new_tokens=max_text_len - 1,
        )
        for token_ids, image_name in zip(generated_ids.cpu(), names):
            hypothesis = tokenizer.decode(token_ids.tolist())
            references = references_by_image[image_name]
            hypotheses.append(hypothesis)
            reference_groups.append(references)
            if len(examples) < 5:
                examples.append(
                    {
                        "image": image_name,
                        "prediction": hypothesis,
                        "references": references,
                    }
                )

    metrics = corpus_bleu(hypotheses, reference_groups)
    metrics["generated_images"] = len(hypotheses)
    return metrics, examples


def save_best_checkpoint(path, model, optimizer, tokenizer, args, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_type": tokenizer.tokenizer_type,
            "config": vars(args),
            "epoch": epoch,
            "validation": metrics,
        },
        path,
    )


def train_with_validation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    samples = read_caption_file(args.caption_file)
    train_indices, val_indices, train_images, val_images = split_by_image(
        samples, val_ratio=args.val_fraction, seed=args.seed
    )
    tokenizer = build_tokenizer(
        getattr(args, "tokenizer", "char"),
        texts=(samples[index][1] for index in train_indices),
        min_word_frequency=getattr(args, "min_word_frequency", 1),
        max_vocab_size=getattr(args, "max_vocab_size", None),
    )
    dataset = ImageCaptionDataset(
        args.image_root,
        args.caption_file,
        tokenizer,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
    )
    collate_fn = make_vlm_collate_fn(tokenizer)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_options
    )

    model = build_model(
        args.image_size,
        args.max_text_len,
        tokenizer.vocab_size,
        tie_word_embeddings=getattr(args, "tie_word_embeddings", False),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(
        f"device={device} train_images={len(train_images)} "
        f"val_images={len(val_images)} train_captions={len(train_indices)} "
        f"val_captions={len(val_indices)} amp={amp_enabled} "
        f"tokenizer={tokenizer.tokenizer_type} vocab_size={tokenizer.vocab_size}"
    )
    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, input_ids, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                _, loss = model(images, input_ids, labels)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if step % args.log_interval == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"train_loss={loss.item():.6f}"
                )

        train_loss = running_loss / len(train_loader)
        metrics = evaluate_teacher_forcing(
            model, val_loader, device, amp_enabled, amp_dtype
        )
        metrics.update({"epoch": epoch, "train_loss": train_loss})
        history.append(metrics)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"ppl={metrics['perplexity']:.3f} "
            f"token_acc={metrics['token_accuracy']:.3%}"
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            save_best_checkpoint(
                args.save_path, model, optimizer, tokenizer, args, epoch, metrics
            )
            print(f"saved new best checkpoint: {args.save_path}")

    checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    generation_metrics, examples = evaluate_generation(
        model,
        dataset,
        val_images,
        tokenizer,
        device,
        args.eval_batch_size,
        args.num_workers,
        args.max_text_len,
        args.eval_images,
    )
    report = {
        "split": {
            "seed": args.seed,
            "train_images": len(train_images),
            "val_images": len(val_images),
            "train_captions": len(train_indices),
            "val_captions": len(val_indices),
        },
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "generation": generation_metrics,
        "examples": examples,
        "history": history,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"final BLEU-1={generation_metrics['bleu1']:.4f} "
        f"BLEU-4={generation_metrics['bleu4']:.4f} "
        f"images={generation_metrics['generated_images']}"
    )
    for example in examples:
        print(f"image={example['image']} prediction={example['prediction']!r}")
        print(f"reference={example['references'][0]!r}")
    print(f"report saved: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TinyVLM with an image-level train/validation split"
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--max_text_len", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_images", type=int, default=0)
    parser.add_argument("--save_path", default="checkpoints/flickr8k_cv_best.pt")
    parser.add_argument("--report_path", default="logs/flickr8k_cv_metrics.json")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--tokenizer", choices=("char", "word"), default="char")
    parser.add_argument("--min_word_frequency", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int)
    parser.add_argument("--tie_word_embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_with_validation(parse_args())
