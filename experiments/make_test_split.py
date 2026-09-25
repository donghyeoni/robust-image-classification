import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Class-balanced Test split.")
    parser.add_argument("--source", required=True,
                        help="Dataset root containing Train/ and Test/.")
    parser.add_argument("--target", required=True,
                        help="New dataset root (Train/ and Test/ as symlinks).")
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", default="results/test_split.txt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.target, exist_ok=True)
    train = os.path.join(args.target, "Train")
    if not os.path.exists(train):
        os.symlink(os.path.abspath(os.path.join(args.source, "Train")), train)

    chosen = []
    test = os.path.join(args.source, "Test")
    for cls in sorted(os.listdir(test)):
        files = sorted(os.listdir(os.path.join(test, cls)))
        if len(files) > args.per_class:
            idx = np.sort(rng.choice(len(files), args.per_class, replace=False))
            files = [files[i] for i in idx]
        out = os.path.join(args.target, "Test", cls)
        os.makedirs(out, exist_ok=True)
        for f in files:
            link = os.path.join(out, f)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(os.path.join(test, cls, f)), link)
            chosen.append(f"{cls}/{f}")

    os.makedirs(os.path.dirname(args.list), exist_ok=True)
    with open(args.list, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chosen) + "\n")


if __name__ == "__main__":
    main()
