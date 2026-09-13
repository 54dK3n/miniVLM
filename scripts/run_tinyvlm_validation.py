import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinyvlm.train_with_validation import parse_args, train_with_validation


if __name__ == "__main__":
    train_with_validation(parse_args())
