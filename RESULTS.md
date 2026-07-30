# Results

These experiments require the private 4-class **Animals** ImageFolder dataset
and a GPU, so they are **not re-run in this repository**. Instead, the
training/evaluation logs captured in the original notebook runs are preserved
under [`results/notebook_reference/`](results/notebook_reference/) — one `.log`
per experiment — and the headline numbers are summarized below.

To reproduce on your own data:

```bash
python experiments/01_rgb_baseline.py --data-root /path/to/Animals
# ... one script per experiment; see README.md
```

## Headline test accuracy (from the preserved original runs)

| Experiment | Input | Test accuracy |
| --- | --- | --- |
| 01 RGB baseline | 3ch RGB, 256² | **0.880** |
| 03 Edge preprocessing | 1ch edge, 64² | 0.726 |
| 04 Bit-flip robustness | 1ch binary | ~0.63–0.66 (across noise 0.05–0.50) |
| 05 Morphological denoise | 1ch binary | ~0.60–0.64 (by filter) |
| 07 Two-channel fusion | 2ch (blur+edge), 45² | ~0.65–0.69 |
| 08 Wavelet subbands | 4ch Haar, 32² | ~0.56–0.60 |

Notes on the preserved logs:

- **02 (binarized)** and **09 (JPEG channel budget)** logs contain training
  accuracy only (no final test line was printed / the run was stopped early —
  09 was interrupted at epoch 12/50 with train accuracy ~0.93).
- **06 (custom pixel denoise)** was interrupted before completion (the log ends
  in a `KeyboardInterrupt`).

The RGB baseline is the strongest (0.88); accuracy drops progressively as the
input is degraded to binary / edge / wavelet representations and under bit-flip
noise — the bandwidth-vs-accuracy trade-off this project studies. See the
per-experiment logs and `docs/` for the full picture.
