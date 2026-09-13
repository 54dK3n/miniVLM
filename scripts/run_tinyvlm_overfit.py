import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.overfit_demo import run_overfit_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Run TinyVLM single-pair overfit check")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_overfit_demo(steps=args.steps, lr=args.lr, device=args.device)
