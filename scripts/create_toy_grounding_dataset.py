import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.toy_grounding import create_toy_grounding_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Create toy visual grounding data")
    parser.add_argument("--output_root", default="data/toy/visual_grounding")
    parser.add_argument("--samples_per_combination", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_root, caption_file, rows = create_toy_grounding_dataset(
        args.output_root,
        args.samples_per_combination,
        args.image_size,
        args.seed,
    )
    print(f"images: {image_root}")
    print(f"captions: {caption_file}")
    print(f"samples: {len(rows)}")
