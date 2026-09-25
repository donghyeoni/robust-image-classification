from __future__ import annotations

import argparse


def build_common_parser(description: str = "",
                        default_batch_size: int = 40) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing Train/ and Test/.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser
