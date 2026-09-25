import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
OUT = os.path.join(RESULTS, "summary")

REPRESENTATION = ["01_rgb_baseline", "02_binarized_input",
                  "03_edge_preprocessing", "04_twochannel_fusion",
                  "05_wavelet_subbands"]
NOISE = ["06_bitflip_robustness", "07_morphological_denoise",
         "08_custom_pixel_denoise", "09_mrf_denoise"]
LABELS = {
    "01_rgb_baseline": "01 RGB",
    "02_binarized_input": "02 binary",
    "03_edge_preprocessing": "03 edge",
    "04_twochannel_fusion": "04 blur+edge",
    "05_wavelet_subbands": "05 Haar",
    "06_bitflip_robustness": "06 no repair",
    "07_morphological_denoise": "07 majority",
    "08_custom_pixel_denoise": "08 custom rules (ours)",
    "09_mrf_denoise": "09 MRF",
}
LEVELS = ["0.05", "0.1", "0.25", "0.5"]
COLORS = ["#E45756", "#F58518", "#54A24B", "#4C78A8"]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seeds_of(name):
    return sorted(glob.glob(os.path.join(RESULTS, name, "seed*")),
                  key=lambda p: int(p.rsplit("seed", 1)[1]))


def metric(name, epoch, noise, key="accuracy"):
    vals = []
    for d in seeds_of(name):
        for r in read_csv(os.path.join(d, "metrics.csv")):
            if int(r["epoch"]) == epoch and r["test_noise"] == noise:
                n = int(r["images"])
                if key.startswith("recall_"):
                    n //= sum(k.startswith("recall_") for k in r)
                vals.append(round(float(r[key]) * n) / n)
    return np.array(vals)


def f4(x):
    return f"{x:.4f}"


def ms(v):
    return f"{f4(v.mean())} ± {f4(v.std())}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    classes = [k[len("recall_"):] for k in read_csv(os.path.join(
        seeds_of(REPRESENTATION[0])[0], "metrics.csv"))[0] if k.startswith("recall_")]
    n_seeds = len(seeds_of(REPRESENTATION[0]))
    images = read_csv(os.path.join(seeds_of(REPRESENTATION[0])[0],
                                   "metrics.csv"))[0]["images"]
    final = max(int(r["epoch"]) for r in read_csv(os.path.join(
        seeds_of(REPRESENTATION[0])[0], "metrics.csv")))
    epochs = sorted({int(r["epoch"]) for r in read_csv(os.path.join(
        seeds_of(REPRESENTATION[0])[0], "metrics.csv"))})
    sections = []

    rows = []
    for n in REPRESENTATION + NOISE:
        tr = [read_csv(os.path.join(d, "train.csv")) for d in seeds_of(n)]
        loss = np.array([float(t[-1]["loss"]) for t in tr])
        acc = np.array([float(t[-1]["accuracy"]) for t in tr])
        sec = np.array([np.mean([float(r["seconds"]) for r in t]) for t in tr])
        rows.append([LABELS[n], len(tr), len(tr[0]), ms(loss), ms(acc),
                     f"{sec.mean():.1f}"])
    sections.append(("S-a Training (last epoch, mean ± SD over seeds)", md_table(
        ["experiment", "seeds", "epochs", "train loss", "train accuracy",
         "seconds per epoch"], rows)))

    rows = []
    for n in REPRESENTATION:
        v = metric(n, final, "")
        rows.append([LABELS[n], ms(v), f"{f4(v.min())} – {f4(v.max())}"])
    sections.append((f"S-b Test accuracy, no noise, epoch {final}", md_table(
        ["experiment", "accuracy (mean ± SD)", "min – max"], rows)))

    rows = []
    for n in NOISE:
        rows.append([LABELS[n], *[ms(metric(n, final, r)) for r in LEVELS]])
    sections.append((f"S-c Test accuracy under s&p noise, epoch {final} "
                     "(mean ± SD over seeds)",
                     md_table(["experiment", *[f"p = {r}" for r in LEVELS]], rows)))

    rows = []
    for n in REPRESENTATION:
        rows.append([LABELS[n], "–", *[f4(metric(n, final, "", f"recall_{c}").mean())
                                      for c in classes]])
    for n in NOISE:
        for r in LEVELS:
            rows.append([LABELS[n], r, *[f4(metric(n, final, r, f"recall_{c}").mean())
                                         for c in classes]])
    sections.append((f"S-d Per-class recall, epoch {final} (mean over seeds)",
                     md_table(["experiment", "p", *classes], rows)))

    rows = []
    for n in REPRESENTATION:
        rows.append([LABELS[n], "–", *[f4(metric(n, e, "").mean()) for e in epochs]])
    for n in NOISE:
        for r in LEVELS:
            rows.append([LABELS[n], r, *[f4(metric(n, e, r).mean()) for e in epochs]])
    sections.append(("S-e Test accuracy of each saved checkpoint (mean over seeds)",
                     md_table(["experiment", "p", *[f"epoch {e}" for e in epochs]],
                              rows)))

    pix = read_csv(os.path.join(RESULTS, "pixel_restoration_study",
                                "residual_error.csv"))
    sections.append((f"S-f Residual pixel error rate after repair "
                     f"({pix[0]['images']} Test images)", md_table(
        ["method", *[f"p = {r}" for r in LEVELS]],
        [[r["method"], *[r[f"p={l}"] for l in LEVELS]] for r in pix])))

    bench = read_csv(os.path.join(RESULTS, "vectorization_benchmark", "timing.csv"))
    rows = []
    for r in bench:
        a, b = float(r["per_pixel_seconds"]), float(r["vectorized_seconds"])
        rows.append([r["method"], r["images"], r["per_pixel_seconds"],
                     r["vectorized_seconds"], f"{a / b:.0f}", r["identical_outputs"]])
    sections.append(("S-g Per-pixel loops vs vectorized code (single CPU thread)",
                     md_table(["method", "images", "per-pixel (s)", "vectorized (s)",
                               "speed-up", "identical outputs"], rows)))

    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(f"Seeds: {n_seeds}. Test images per evaluation: {images}.\n\n")
        for title, table in sections:
            f.write(f"### {title}\n\n{table}\n\n")

    x = np.arange(len(REPRESENTATION))
    v = [metric(n, final, "") for n in REPRESENTATION]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(x, [a.mean() for a in v], 0.6, yerr=[a.std() for a in v],
                  capsize=4, color="#4C78A8", zorder=3)
    ax.bar_label(bars, [f4(a.mean()) for a in v], fontsize=7, padding=6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n].replace(" ", "\n", 1) for n in REPRESENTATION],
                       fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"test accuracy ({images} images)")
    ax.set_title(f"Representation (no noise), mean ± SD over {n_seeds} seeds",
                 fontsize=10)
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "representation.png"), dpi=150)
    plt.close(fig)

    lv = [float(r) for r in LEVELS]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for n, c in zip(NOISE, COLORS):
        v = [metric(n, final, r) for r in LEVELS]
        ax.errorbar(lv, [a.mean() for a in v], yerr=[a.std() for a in v],
                    marker="o", ms=4, capsize=3, color=c, label=LABELS[n])
    ax.axhline(0.25, color="gray", lw=0.8, ls="--")
    ax.set_xticks(lv)
    ax.set_xticklabels(LEVELS)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("test s&p ratio p")
    ax.set_ylabel(f"test accuracy ({images} images)")
    ax.set_title(f"Noise robustness, mean ± SD over {n_seeds} seeds", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    styles = {"none": ("#0b0b0b", "--"), "median 3x3": ("#B279A2", "-"),
              "majority 3x3": ("#F58518", "-"),
              "custom rules (ours)": ("#54A24B", "-"),
              "MRF(1.5,0.7)": ("#4C78A8", "-")}
    for r in pix:
        if r["method"] in styles:
            c, ls = styles[r["method"]]
            ax.plot(lv, [float(r[f"p={l}"]) for l in LEVELS], marker="o", ms=4,
                    color=c, ls=ls, label=r["method"])
    ax.set_xticks(lv)
    ax.set_xticklabels(LEVELS)
    ax.set_xlabel("s&p ratio p")
    ax.set_ylabel("residual pixel error rate")
    ax.set_title(f"Pixel restoration ({pix[0]['images']} Test images, 64×64 binary)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pixel_error.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
