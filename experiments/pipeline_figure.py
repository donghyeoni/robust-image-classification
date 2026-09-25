import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.data import build_dataset, seed_everything
from rc.denoise import MajorityFilter, MRFDenoiser, diagonal_solo, mismatch
from rc.noise import AddNoise
from rc.preprocessing import (Binarize, OurPreprocessing, PreprocessingBaseline,
                              PreprocessingWavelet, TwoChannelPreprocessing)


def tile(chw):
    c = chw.shape[0]
    if c == 2:
        return np.concatenate([chw[0], np.ones((chw.shape[1], 2)), chw[1]], axis=1)
    if c == 4:
        gap = np.ones((chw.shape[1], 2))
        top = np.concatenate([chw[0], gap, chw[1]], axis=1)
        bottom = np.concatenate([chw[2], gap, chw[3]], axis=1)
        return np.concatenate([top, np.ones((2, top.shape[1])), bottom], axis=0)
    return chw[0]


def main():
    parser = argparse.ArgumentParser(description="Inputs of experiments 01-09.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--image", default="Tiger/0")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    testset = build_dataset(args.data_root, "Test")
    cls, k = args.image.split("/")
    paths = [p for p, y in testset.samples if testset.classes[y] == cls]
    path = paths[int(k)]
    img = Image.open(path).convert("RGB")

    rgb = transforms.Resize((256, 256))(img)
    b02 = transforms.ToTensor()(Binarize(128)(transforms.Resize((64, 64))(img)))
    edge = OurPreprocessing(size=64, crop=2, method="Otsu", return_type="array")(img)
    two = TwoChannelPreprocessing(size=45, crop=2)(img).numpy()
    wav = PreprocessingWavelet(size=32, crop=2)(img).numpy()

    seed_everything(args.seed)
    base = PreprocessingBaseline(threshold=128, size=64, return_type="array")(img)
    noise = AddNoise(args.ratio, return_type="array")
    noisy = noise(base)
    noisy_edge = noise(edge)
    ours = diagonal_solo(mismatch(torch.from_numpy(noisy_edge))).numpy()

    top = [("01 RGB 256²", np.asarray(rgb)),
           ("02 binary 64²", b02[0].numpy()), ("03 edge 64²", edge / 255),
           ("04 blur | edge 45²", tile(two)), ("05 LL LH / HL HH 32²", tile(wav))]
    bottom = [("baseline binary 64²", base / 255),
              (f"06 + noise p={args.ratio:g}", noisy / 255),
              ("07 majority", MajorityFilter(3)(noisy) / 255),
              (f"08 edge + noise → ours", ours / 255),
              ("09 MRF", MRFDenoiser()(noisy) / 255)]

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for row, panels in zip(axes, (top, bottom)):
        for ax in row:
            ax.axis("off")
        for ax, (title, im) in zip(row, panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1
                      if im.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=9)
    axes[0, 0].text(-0.08, 0.5, "representation", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
    axes[1, 0].text(-0.08, 0.5, "noise + repair", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
    fig.tight_layout()
    out = os.path.join(args.results_dir, "pipeline")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "pipeline.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, "source.txt"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(os.path.relpath(path, args.data_root).replace(os.sep, "/") + "\n")


if __name__ == "__main__":
    main()
