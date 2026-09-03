import argparse
from pathlib import Path

import torch

from .config import LACGNNConfig
from .training import train_many


def main():
    parser = argparse.ArgumentParser(description="Train the canonical LACGNN")
    parser.add_argument(
        "--data-dir", default="datasets",
        help="directory containing the five CSV files (default: ./datasets)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument(
        "--log-file",
        help="clean TXT log path (default: <output-dir>/train.txt)",
    )
    args = parser.parse_args()
    if args.window_stride < 1:
        parser.error("--window-stride must be positive")
    log_file = args.log_file or Path(args.output_dir) / "train.txt"
    train_many(
        args.data_dir, args.output_dir,
        LACGNNConfig(window_stride=args.window_stride),
        range(1, args.seeds + 1), torch.device(args.device), log_file)


if __name__ == "__main__":
    main()
