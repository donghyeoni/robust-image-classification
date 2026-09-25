import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.data import build_dataset, seed_everything
from rc.denoise import (MajorityFilter, MedianFilter, MRFDenoiser,
                        diagonal_solo, mismatch)
from rc.noise import AddNoise
from rc.preprocessing import PreprocessingBaseline
from rc.runner import NOISE_LEVELS

MRF_CANDIDATES = ((1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (1.5, 0.7))


def custom_rules(arr):
    return diagonal_solo(mismatch(torch.from_numpy(arr))).numpy()


def residual_error(denoiser, cleans, ratio, seed):
    seed_everything(seed)
    noise = AddNoise(ratio, return_type="array")
    return float(np.mean([(denoiser(noise(c)) != c).mean() for c in cleans]))


def main():
    parser = argparse.ArgumentParser(description="Pixel-level restoration study.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pre = PreprocessingBaseline(threshold=128, size=64, return_type="array")
    testset = build_dataset(args.data_root, "Test")
    cleans = [pre(Image.open(p)) for p, _ in testset.samples]

    methods = {"none": lambda a: a, "median 3x3": MedianFilter(3),
               "majority 3x3": MajorityFilter(3), "custom rules (ours)": custom_rules,
               **{f"MRF({b:g},{e:g})": MRFDenoiser(beta=b, eta=e, iters=10)
                  for b, e in MRF_CANDIDATES}}
    rows = []
    for name, d in methods.items():
        row = {"method": name, "images": len(cleans)}
        for r in NOISE_LEVELS:
            row[f"p={r:g}"] = f"{residual_error(d, cleans, r, args.seed):.4f}"
        rows.append(row)
        print(row, flush=True)

    out = os.path.join(args.results_dir, "pixel_restoration_study")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "residual_error.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
