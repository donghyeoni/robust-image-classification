# Results

These experiments require the private 4-class **Animals** ImageFolder dataset
and a GPU, so they are **not re-run from scratch in this repository**. The
training/evaluation logs captured in the original notebook runs are preserved
under [`results/notebook_reference/`](results/notebook_reference/) — one `.log`
per experiment — and the headline numbers are summarized below.

That directory also holds the qualitative figures from the original report
notebook (`report__*.png`): the Median/Component/Majority denoise comparison,
the LoG+Sobel edge-weighting steps, bit-flip noise samples, one- vs two-stage
downscaling, the Global/Otsu/Adaptive thresholding comparison, and a GrabCut
foreground experiment. `09_jpeg_channel_budget__*.png` is the confusion matrix
from the JPEG-channel run.

To reproduce on your own data:

```bash
python experiments/01_rgb_baseline.py --data-root /path/to/Animals
# ... one script per experiment; see README.md
```

## Headline test accuracy

| Experiment | Input | Test accuracy | Source |
| --- | --- | --- | --- |
| 01 RGB baseline | 3ch RGB, 256² | **0.880** | original log |
| 03 Edge preprocessing | 1ch edge, 64² | 0.726 | original log |
| 04 Bit-flip robustness | 1ch binary | ~0.63–0.66 (across noise 0.05–0.50) | original log |
| 05 Morphological denoise | 1ch binary, 64² | ~0.60–0.64 (by filter) | original log |
| 07 Two-channel fusion | 2ch (blur+edge), 45² | ~0.65–0.69 | original log |
| 08 Wavelet subbands | 4ch Haar, 32² | ~0.56–0.60 | original log |
| 09 JPEG channel budget | 3ch RGB, 224², 64 KB + BER 1e-3 | **0.874** | checkpoint re-evaluation |

### 09 — JPEG channel budget, per checkpoint

The original notebook run was interrupted at epoch 12/50, but checkpoints from
a completed 50-epoch run survived. Re-evaluating them on the Test split (784 images,
`target_bytes=65536`, `bit_error_rate=0.001`) gives:

| Checkpoint | Test loss | Test accuracy |
| --- | --- | --- |
| epoch 10 | 0.5244 | 0.7972 |
| epoch 20 | 0.9490 | 0.7997 |
| epoch 30 | 0.8293 | 0.8520 |
| epoch 40 | 0.7225 | 0.8673 |
| epoch 50 | 0.7092 | **0.8737** |

Accuracy rises monotonically from epoch 30 onwards, ending close to the
uncompressed RGB baseline (0.880) — a 64 KB JPEG budget with a 1e-3 byte error
rate costs only about half a point of accuracy.

The confusion matrix saved by the original notebook
([`09_jpeg_channel_budget__cell07_1.png`](results/notebook_reference/09_jpeg_channel_budget__cell07_1.png))
independently corroborates this: its diagonal gives
`(116+289+144+114) / 760 = 0.872`, matching the 0.874 re-evaluation above to
within the run-to-run variation of the random byte errors. Cat↔Dog accounts for
68 of the 97 errors; Tiger and Zebra are almost never confused with anything.

### 05 — checkpoint re-evaluation cross-check

Re-running the 05 test pipeline (`PreprocessingBaseline` → `AddNoise` →
`MajorityFilter` → tensor) against the surviving epoch-50 checkpoint reproduces
the logged number: **0.602** measured vs **0.625** logged, the gap being the
run-to-run variance of the random bit-flip noise and the shuffled test order.
This confirms the archived logs and the checkpoints describe the same runs.

## Gaps

- **02 (binarized input)** — no test line was printed and no checkpoints were
  kept, so only training accuracy is available (0.988 at epoch 38/40). Filling
  this in requires re-running the experiment.
- **06 (custom pixel denoise)** — the preserved log ends in a
  `KeyboardInterrupt` and no checkpoint could be matched to this configuration.
  Candidate checkpoint sets that survived alongside the 05 ones score at chance
  level (0.22–0.45 on 4 classes) under both the `ours` and `baseline`
  binarizers, so they belong to some other configuration and cannot stand in
  for 06. This experiment needs a genuine re-run.

The RGB baseline is the strongest (0.880); accuracy drops progressively as the
input is degraded to binary / edge / wavelet representations and under bit-flip
noise — the bandwidth-vs-accuracy trade-off this project studies. The JPEG
channel (09) is the exception: it keeps a 3-channel representation and so stays
close to the baseline. See the per-experiment logs and `docs/` for the full
picture.
