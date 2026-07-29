"""Shared command-line configuration for experiment scripts.

Provides a common argument parser so that machine-specific values (dataset
root, checkpoint directory, hyper-parameters) are passed in at run time rather
than hardcoded, as they were in the original notebooks.
"""

from __future__ import annotations

import argparse


def build_common_parser(description: str = "",
                        default_epochs: int = 50,
                        default_batch_size: int = 40) -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-populated with the common options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root", required=True,
        help="Path to the dataset root containing Train/ and Test/ subfolders.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory to write model checkpoints into.",
    )
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Checkpoint frequency in epochs.")
    parser.add_argument("--num-classes", type=int, default=4)
    return parser
