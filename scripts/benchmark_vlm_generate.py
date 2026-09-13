import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.dataset import ImageCaptionDataset
from tinyvlm.multimodal_generate import generate
from tinyvlm.train_vlm import build_model
from tokenizer import tokenizer_from_checkpoint


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU generation benchmark")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    model = build_model(
        config["image_size"],
        config["max_text_len"],
        tokenizer.vocab_size,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = ImageCaptionDataset(
        args.image_root,
        args.caption_file,
        tokenizer,
        config["image_size"],
        config["max_text_len"],
    )
    base_image = dataset[0]["image"].unsqueeze(0).to(device)

    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters())}")
    for use_kv_cache in (False, True):
        mode = "kv_cache" if use_kv_cache else "no_cache"
        for batch_size in args.batch_sizes:
            images = base_image.repeat(batch_size, 1, 1, 1)
            generated = generate(
                model,
                images,
                tokenizer.bos_token_id,
                -1,
                max_new_tokens=8,
                use_kv_cache=use_kv_cache,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.repeats):
                generated = generate(
                    model,
                    images,
                    tokenizer.bos_token_id,
                    -1,
                    max_new_tokens=args.max_new_tokens,
                    use_kv_cache=use_kv_cache,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            token_count = batch_size * args.max_new_tokens * args.repeats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(
                f"mode={mode} batch={batch_size} shape={tuple(generated.shape)} "
                f"tokens/s={token_count / elapsed:.2f} "
                f"peak_vram_mb={peak_memory:.2f} elapsed={elapsed:.4f}s"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TinyVLM generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--caption_file", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
