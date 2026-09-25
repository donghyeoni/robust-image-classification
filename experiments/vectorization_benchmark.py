import argparse
import csv
import os
import time

import numpy as np
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.data import build_dataset, seed_everything
from rc.denoise import MajorityFilter, MRFDenoiser, diagonal_solo, mismatch
from rc.noise import AddNoise
from rc.preprocessing import PreprocessingBaseline


def majority_loop(img, k=3):
    h, w = img.shape
    padded = np.pad(img, k // 2, mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            white = np.sum(padded[y:y + k, x:x + k] == 255)
            out[y, x] = 255 if white > k * k - white else 0
    return out


def custom_rules_loop(img):
    t = torch.from_numpy(img)
    h, w = t.shape
    a = t.clone()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = t[i - 1:i + 2, j - 1:j + 2].flatten()
            c = t[i, j]
            if torch.all(torch.cat((patch[:4], patch[5:])) != c):
                a[i, j] = 255 - c
    b = a.clone()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            c = a[i, j]
            diag = sum(int(a[i + di, j + dj] == c)
                       for di, dj in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
            straight = sum(int(a[i + di, j + dj] != c)
                           for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            if diag == 1 and straight == 4:
                b[i, j] = 255 - c
    return b.numpy()


def mrf_loop(img, beta=1.5, eta=0.7, iters=10):
    x = np.where(img > 127, 1.0, -1.0)
    y = x.copy()
    h, w = y.shape
    for _ in range(iters):
        changed = False
        for parity in (0, 1):
            new = y.copy()
            for i in range(h):
                for j in range(w):
                    if (i + j) % 2 != parity:
                        continue
                    s = 0.0
                    if i > 0:
                        s += y[i - 1, j]
                    if i < h - 1:
                        s += y[i + 1, j]
                    if j > 0:
                        s += y[i, j - 1]
                    if j < w - 1:
                        s += y[i, j + 1]
                    f = eta * x[i, j] + beta * s
                    v = 1.0 if f > 0 else (-1.0 if f < 0 else y[i, j])
                    if v != y[i, j]:
                        changed = True
                    new[i, j] = v
            y = new
        if not changed:
            break
    return np.where(y > 0, 255, 0).astype(np.uint8)


def custom_rules_vec(img):
    return diagonal_solo(mismatch(torch.from_numpy(img))).numpy()


def timed(fn, images):
    start = time.perf_counter()
    out = [fn(im) for im in images]
    return time.perf_counter() - start, out


def main():
    parser = argparse.ArgumentParser(description="Per-pixel vs vectorized denoisers.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    seed_everything(args.seed)
    pre = PreprocessingBaseline(threshold=128, size=64, return_type="array")
    noise = AddNoise(args.ratio, return_type="array")
    trainset = build_dataset(args.data_root, "Train")
    idx = np.random.choice(len(trainset.samples), args.images, replace=False)
    images = [noise(pre(trainset.loader(trainset.samples[i][0]))) for i in idx]

    rows = []
    for name, loop, vec in (
            ("majority 3x3", majority_loop, MajorityFilter(3)),
            ("custom rules (ours)", custom_rules_loop, custom_rules_vec),
            ("MRF(1.5,0.7)", mrf_loop, MRFDenoiser(beta=1.5, eta=0.7, iters=10))):
        t_loop, out_loop = timed(loop, images)
        t_vec, out_vec = timed(vec, images)
        same = all(np.array_equal(a, b) for a, b in zip(out_loop, out_vec))
        rows.append({"method": name, "images": args.images,
                     "per_pixel_seconds": f"{t_loop:.2f}",
                     "vectorized_seconds": f"{t_vec:.4f}",
                     "identical_outputs": same})
        print(rows[-1], flush=True)

    out = os.path.join(args.results_dir, "vectorization_benchmark")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "timing.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
