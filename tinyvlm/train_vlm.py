import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from tiny_gpt import TinyGPT
from tokenizer import build_tokenizer
from vision_encoder import TinyViT

from .dataset import (
    ImageCaptionDataset,
    make_vlm_collate_fn,
    read_caption_file,
    split_by_image,
)
from .multimodal_generate import generate
from .tinyvlm_model import TinyVLM


PATCH_SIZE = 8
MODEL_DIM = 64


def build_model(
    image_size, max_text_len, vocab_size, tie_word_embeddings=False
):
    if image_size % PATCH_SIZE != 0:
        raise ValueError(f"image_size must be divisible by {PATCH_SIZE}")

    num_image_tokens = (image_size // PATCH_SIZE) ** 2 + 1
    max_seq_len = num_image_tokens + max_text_len
    vit = TinyViT(image_size, PATCH_SIZE, 3, MODEL_DIM, 8, 2)
    gpt = TinyGPT(
        MODEL_DIM,
        4,
        2,
        vocab_size,
        max_seq_len,
        tie_word_embeddings=tie_word_embeddings,
    )
    return TinyVLM(vit, gpt, vit_dim=MODEL_DIM, gpt_dim=MODEL_DIM)


def save_checkpoint(path, model, optimizer, tokenizer, args, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tokenizer_stoi": tokenizer.stoi,
            "tokenizer_type": tokenizer.tokenizer_type,
            "config": vars(args),
            "validation": metrics,
        },
        path,
    )


@torch.no_grad()
def validate(
    model,
    data_loader,
    device,
    amp_enabled=False,
    amp_dtype=torch.float16,
    ablation_seed=0,
):
    """Calculate normal, shuffled, zero, and uniform-noise image losses."""
    model.eval()
    total_loss = 0.0
    total_shuffle_loss = 0.0
    total_zero_loss = 0.0
    total_noise_loss = 0.0
    total_tokens = 0
    generator = torch.Generator(device=device).manual_seed(ablation_seed)

    for images, input_ids, labels in data_loader:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid_tokens = labels.ne(-100).sum().item()

        permutation = torch.randperm(
            images.size(0), device=device, generator=generator
        )
        if images.size(0) > 1 and torch.equal(
            permutation, torch.arange(images.size(0), device=device)
        ):
            permutation = permutation.roll(1)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            _, loss = model(images, input_ids, labels)
            _, shuffle_loss = model(images[permutation], input_ids, labels)
            _, zero_loss = model(torch.zeros_like(images), input_ids, labels)
            noise_images = torch.rand(
                images.shape,
                dtype=images.dtype,
                device=device,
                generator=generator,
            )
            _, noise_loss = model(noise_images, input_ids, labels)

        total_loss += loss.item() * valid_tokens
        total_shuffle_loss += shuffle_loss.item() * valid_tokens
        total_zero_loss += zero_loss.item() * valid_tokens
        total_noise_loss += noise_loss.item() * valid_tokens
        total_tokens += valid_tokens

    return (
        total_loss / total_tokens,
        total_shuffle_loss / total_tokens,
        total_zero_loss / total_tokens,
        total_noise_loss / total_tokens,
    )


@torch.no_grad()
def print_validation_samples(
    model,
    dataset,
    val_indices,
    tokenizer,
    device,
    num_samples,
    max_new_tokens,
):
    """Greedily caption distinct validation images and print their targets."""
    model.eval()
    selected_indices = []
    seen_images = set()
    for index in val_indices:
        image_name, _ = dataset.samples[index]
        if image_name not in seen_images:
            selected_indices.append(index)
            seen_images.add(image_name)
        if len(selected_indices) >= num_samples:
            break

    for sample_number, index in enumerate(selected_indices, start=1):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        generated_ids = generate(
            model,
            image,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_caption = tokenizer.decode(generated_ids[0].tolist())
        print(f"Sample {sample_number}")
        print(f"  target:    {sample['caption'].lower()}")
        print(f"  generated: {generated_caption}")


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    train_indices, val_indices, _, _ = split_by_image(
        samples, val_ratio=args.val_ratio, seed=args.seed
    )
    if args.limit_samples is not None:
        if args.limit_samples < 1:
            raise ValueError("limit_samples must be positive")
        train_indices = train_indices[: args.limit_samples]
        val_indices = val_indices[: max(1, round(args.limit_samples * args.val_ratio))]

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

    active_train_images = {dataset.samples[index][0] for index in train_indices}
    active_val_images = {dataset.samples[index][0] for index in val_indices}

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": make_vlm_collate_fn(tokenizer),
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=generator,
        **loader_options,
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
    best_val_loss = float("inf")

    print(
        f"device={device} train_images={len(active_train_images)} "
        f"val_images={len(active_val_images)} train_captions={len(train_indices)} "
        f"val_captions={len(val_indices)} tokenizer={tokenizer.tokenizer_type} "
        f"vocab_size={tokenizer.vocab_size}"
    )
    max_new_tokens = min(args.max_new_tokens, args.max_text_len - 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_tokens = 0

        for step, (images, input_ids, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid_tokens = labels.ne(-100).sum().item()
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

            total_train_loss += loss.item() * valid_tokens
            total_train_tokens += valid_tokens
            if step % args.log_interval == 0:
                print(f"  step {step}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss = total_train_loss / total_train_tokens
        val_loss, val_shuffle_loss, val_zero_loss, val_noise_loss = validate(
            model, val_loader, device, amp_enabled, amp_dtype
        )
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  train_loss:       {train_loss:.4f}")
        print(f"  val_loss:         {val_loss:.4f}")
        print(f"  val_shuffle_loss: {val_shuffle_loss:.4f}")
        print(f"  val_zero_loss:    {val_zero_loss:.4f}")
        print(f"  val_noise_loss:   {val_noise_loss:.4f}")

        print_validation_samples(
            model,
            dataset,
            val_indices,
            tokenizer,
            device,
            args.num_samples,
            max_new_tokens,
        )
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_shuffle_loss": val_shuffle_loss,
            "val_zero_loss": val_zero_loss,
            "val_noise_loss": val_noise_loss,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.save_path, model, optimizer, tokenizer, args, epoch, metrics
            )
            print(f"  saved best checkpoint: {args.save_path}")

        # generate() switches to eval mode; explicitly restore training mode.
        model.train()

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TinyVLM with validation and image-shuffle ablation"
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--max_text_len", type=int, default=96)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--save_path", default="best_tinyvlm.pt")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--tokenizer", choices=("char", "word"), default="char")
    parser.add_argument("--min_word_frequency", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int)
    parser.add_argument("--tie_word_embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
