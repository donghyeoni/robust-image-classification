# Experiment Log

Settings, full result tables and observations of the runs of 2026-09-26: nine
classification experiments (01–09) with 5 seeds each, a pixel-level
restoration study and a timing comparison of per-pixel and vectorized code.
The [README](../README.md) summarises the results; every result value in it
appears in the tables below.

## Conventions

- **Accuracy** is the fraction of Test images whose predicted class is the
  true class. **Recall** of a class is the fraction of its Test images
  predicted as that class. The Test split has the same number of images per
  class, so accuracy equals the mean of the four recalls.
- **Seeds.** Every classification experiment is trained with seeds 0–4.
  `mean ± SD` is over the 5 seeds, with the population SD (`numpy.std`,
  ddof = 0).
- **Precision.** Accuracies, recalls, losses and pixel error rates are shown
  with 4 decimals; times in seconds as the scripts print them.
- **s&p ratio** `p`: the fraction of pixels of a 64×64 binary image whose
  value is inverted (`v → 255 − v`). Exactly `int(p · 4096)` distinct pixels
  are inverted: 204, 409, 1024 and 2048 for `p` = 0.05, 0.1, 0.25 and 0.5.
- **Result tables.** `experiments/summarize.py` reads the per-run CSV files
  under `results/` and writes `results/summary/tables.md` and three figures;
  the tables below are copied unchanged from `tables.md`.

## Data

- The Animals set (Cat, Dog, Tiger, Zebra): Train 2,000 images per class
  (8,000); Test 150 Cat, 338 Dog, 150 Tiger and 150 Zebra images.
- **Class-balanced Test split** (`experiments/make_test_split.py`, seed 0):
  150 of the 338 Dog Test images are drawn at random without replacement; the
  other classes keep all 150. The Test split used everywhere below has 600
  images, 150 per class; the file names are listed in
  `results/test_split.txt`. Train is unchanged.
- The dataset is not in the repository.

## Common setup

- **Model** (`rc/model.py`). torchvision ResNet-18 without pretrained
  weights; `conv1` replaced by a 7×7, stride 2, padding 3 convolution with the
  experiment's number of input channels; `fc` replaced by one linear layer
  with 4 outputs.
- **Training** (`rc/runner.py`, `rc/engine.py`). 50 epochs, Adam with
  learning rate 1e-3, cross-entropy loss. Batch size 20 for 01 and 02, 40 for
  03–09. The Train loader is shuffled and drops the last incomplete batch
  (8,000 is a multiple of 20 and 40, so no image is dropped). A checkpoint is
  saved after epochs 10, 20, 30, 40 and 50.
- **Evaluation.** Every saved checkpoint is scored on all 600 Test images
  (not shuffled, no image dropped); 06–09 at each `p` in {0.05, 0.1, 0.25,
  0.5}. Accuracy is counted per image. Before each evaluation the random state
  is reset to seed 1000, so every checkpoint of every seed is scored on the
  same noisy Test images for a given `p`.
- **Reproducibility.** `seed_everything` (`rc/data.py`) seeds `random`, NumPy
  and PyTorch (CPU and CUDA) and sets cuDNN to deterministic mode; each data
  loader worker seeds `random` and NumPy from its PyTorch seed, and the
  shuffling generator is seeded. Before the runs, 02 with seed 0 was trained
  for one epoch twice; the two checkpoints were byte-identical.
- **Runs.** `run_all.sh` makes the split, runs the timing comparison alone,
  then runs the training runs with the pixel-level study at the same time, and
  draws the input figure (`experiments/pipeline_figure.py`).
  Each run writes `train.csv`, `metrics.csv` and `run.json` to
  `results/<experiment>/seed<n>/`; its console output is in `results/logs/`.

## Representation experiments (01–05)

No noise. Each experiment changes only the input representation.

| # | script | input | channels |
| --- | --- | --- | --- |
| 01 | `01_rgb_baseline.py` | RGB resized to 256×256 | 3 |
| 02 | `02_binarized_input.py` | resized to 64×64 (torchvision `Resize`), grayscale, pixel `> 128` → 255, else 0 | 1 |
| 03 | `03_edge_preprocessing.py` | edge map (below), 64×64, Otsu threshold | 1 |
| 04 | `04_twochannel_fusion.py` | channel 0: crop 2 pixels per border, grayscale, 3×3 Gaussian blur (σ = 1), bicubic resize to 45×45, Otsu threshold; channel 1: the edge map at 45×45 | 2 |
| 05 | `05_wavelet_subbands.py` | crop 2 pixels per border, grayscale, one-level Haar DWT; LL, LH, HL and HH each bicubic-resized to 32×32, min–max scaled to 0–255, Otsu threshold | 4 |

**Edge map** (`rc/preprocessing/edges.py`, also channel 1 of 04 and the
binarization of 08). Crop 2 pixels per border, grayscale; `|Laplacian|`
(3×3) and Sobel gradient magnitude (3×3), each min–max scaled to [0, 1];
weighted sum with weights proportional to the two maps' means; min–max scaled
to 0–255; 3×3 Gaussian blur (σ = 1); bicubic resize.

## Noise-robustness experiments (06–09)

64×64 binary input with s&p noise. During training each image gets a ratio
drawn at random from {0.05, 0.1, 0.25, 0.5}; the repair step is applied after
the noise, in training and in testing.

| # | script | binarization | repair after noise |
| --- | --- | --- | --- |
| 06 | `06_bitflip_robustness.py` | baseline | none |
| 07 | `07_morphological_denoise.py` | baseline | majority filter |
| 08 | `08_custom_pixel_denoise.py` | edge map (as 03) | custom rules (ours) |
| 09 | `09_mrf_denoise.py` | baseline | MRF |

- **Baseline binarization** (`PreprocessingBaseline`): the same operations
  as the input of 02 (torchvision `Resize` to 64×64, grayscale, pixel `> 128`
  → 255, else 0); on all 600 Test images the two give identical arrays.
- **Majority filter** (`MajorityFilter`): 3×3 window with zero padding; the
  pixel becomes 255 when at least 5 of the 9 window pixels are 255, else 0.
- **Custom rules (ours)** (`mismatch` then `diagonal_solo`, on interior
  pixels; the border is left unchanged). `mismatch` inverts a pixel that
  differs from all 8 neighbours. `diagonal_solo` inverts a pixel that differs
  from its 4 horizontal and vertical neighbours and equals exactly one of its
  4 diagonal neighbours. In 08 they run on the GPU on each batch.
- **MRF** (`MRFDenoiser`, β = 1.5, η = 0.7). With the observation `x` and the
  estimate `y` in {−1, +1} (`y` starts at `x`), each update sets a pixel to
  the sign of `η·x + β·(sum of its 4 neighbours in y)` (unchanged when this is
  0), for one half of a checkerboard and then the other; this is repeated
  until a sweep changes nothing, at most 10 sweeps. It locally minimises the
  Ising energy `E(y) = −η Σ yᵢxᵢ − β Σ₍ᵢ,ⱼ₎ yᵢyⱼ` (neighbouring pairs).
- All inputs are scaled to [0, 1] before the network.

## Results

**S-a Training (last epoch, mean ± SD over seeds)** (`results/summary/tables.md`)

| experiment | seeds | epochs | train loss | train accuracy | seconds per epoch |
| --- | --- | --- | --- | --- | --- |
| 01 RGB | 5 | 50 | 0.0267 ± 0.0137 | 0.9917 ± 0.0039 | 7.2 |
| 02 binary | 5 | 50 | 0.0135 ± 0.0041 | 0.9955 ± 0.0014 | 4.0 |
| 03 edge | 5 | 50 | 0.0205 ± 0.0033 | 0.9935 ± 0.0005 | 3.6 |
| 04 blur+edge | 5 | 50 | 0.0212 ± 0.0100 | 0.9924 ± 0.0036 | 4.5 |
| 05 Haar | 5 | 50 | 0.0409 ± 0.0113 | 0.9861 ± 0.0037 | 2.8 |
| 06 no repair | 5 | 50 | 0.4107 ± 0.0061 | 0.7893 ± 0.0050 | 2.2 |
| 07 majority | 5 | 50 | 0.4067 ± 0.0069 | 0.7931 ± 0.0046 | 2.4 |
| 08 custom rules (ours) | 5 | 50 | 0.4562 ± 0.0109 | 0.7725 ± 0.0042 | 3.5 |
| 09 MRF | 5 | 50 | 0.4293 ± 0.0098 | 0.7830 ± 0.0057 | 3.0 |

The times per epoch were measured while other training runs shared the
machine.

**S-b Test accuracy, no noise, epoch 50** (`results/summary/tables.md`)

| experiment | accuracy (mean ± SD) | min – max |
| --- | --- | --- |
| 01 RGB | 0.8523 ± 0.0070 | 0.8417 – 0.8600 |
| 02 binary | 0.6190 ± 0.0272 | 0.5900 – 0.6617 |
| 03 edge | 0.7097 ± 0.0123 | 0.6917 – 0.7233 |
| 04 blur+edge | 0.7233 ± 0.0071 | 0.7150 – 0.7317 |
| 05 Haar | 0.5967 ± 0.0112 | 0.5783 – 0.6133 |

**S-c Test accuracy under s&p noise, epoch 50 (mean ± SD over seeds)** (`results/summary/tables.md`)

| experiment | p = 0.05 | p = 0.1 | p = 0.25 | p = 0.5 |
| --- | --- | --- | --- | --- |
| 06 no repair | 0.5647 ± 0.0134 | 0.5613 ± 0.0109 | 0.5273 ± 0.0244 | 0.2507 ± 0.0087 |
| 07 majority | 0.5513 ± 0.0092 | 0.5553 ± 0.0143 | 0.5353 ± 0.0123 | 0.2510 ± 0.0080 |
| 08 custom rules (ours) | 0.7203 ± 0.0146 | 0.7070 ± 0.0076 | 0.6453 ± 0.0140 | 0.2480 ± 0.0065 |
| 09 MRF | 0.5533 ± 0.0130 | 0.5540 ± 0.0056 | 0.5353 ± 0.0151 | 0.2433 ± 0.0064 |

**S-d Per-class recall, epoch 50 (mean over seeds)** (`results/summary/tables.md`)

| experiment | p | Cat | Dog | Tiger | Zebra |
| --- | --- | --- | --- | --- | --- |
| 01 RGB | – | 0.7200 | 0.8373 | 0.9613 | 0.8907 |
| 02 binary | – | 0.5293 | 0.6987 | 0.7147 | 0.5333 |
| 03 edge | – | 0.5480 | 0.7413 | 0.7347 | 0.8147 |
| 04 blur+edge | – | 0.5773 | 0.6907 | 0.8213 | 0.8040 |
| 05 Haar | – | 0.5493 | 0.5760 | 0.6293 | 0.6320 |
| 06 no repair | 0.05 | 0.5133 | 0.6467 | 0.6573 | 0.4413 |
| 06 no repair | 0.1 | 0.5107 | 0.6560 | 0.6533 | 0.4253 |
| 06 no repair | 0.25 | 0.4520 | 0.5773 | 0.6467 | 0.4333 |
| 06 no repair | 0.5 | 0.4147 | 0.0440 | 0.3920 | 0.1520 |
| 07 majority | 0.05 | 0.4480 | 0.6613 | 0.6027 | 0.4933 |
| 07 majority | 0.1 | 0.4480 | 0.6613 | 0.6147 | 0.4973 |
| 07 majority | 0.25 | 0.4213 | 0.6267 | 0.5893 | 0.5040 |
| 07 majority | 0.5 | 0.3280 | 0.3787 | 0.2453 | 0.0520 |
| 08 custom rules (ours) | 0.05 | 0.5827 | 0.7373 | 0.7587 | 0.8027 |
| 08 custom rules (ours) | 0.1 | 0.5533 | 0.7320 | 0.7373 | 0.8053 |
| 08 custom rules (ours) | 0.25 | 0.4293 | 0.7307 | 0.6947 | 0.7267 |
| 08 custom rules (ours) | 0.5 | 0.3787 | 0.3973 | 0.0613 | 0.1547 |
| 09 MRF | 0.05 | 0.4493 | 0.6467 | 0.6107 | 0.5067 |
| 09 MRF | 0.1 | 0.4493 | 0.6360 | 0.6013 | 0.5293 |
| 09 MRF | 0.25 | 0.3880 | 0.5893 | 0.6147 | 0.5493 |
| 09 MRF | 0.5 | 0.4213 | 0.1387 | 0.2707 | 0.1427 |

**S-e Test accuracy of each saved checkpoint (mean over seeds)** (`results/summary/tables.md`)

| experiment | p | epoch 10 | epoch 20 | epoch 30 | epoch 40 | epoch 50 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 RGB | – | 0.7910 | 0.8487 | 0.8653 | 0.8613 | 0.8523 |
| 02 binary | – | 0.6007 | 0.6260 | 0.6270 | 0.6290 | 0.6190 |
| 03 edge | – | 0.6563 | 0.6797 | 0.6850 | 0.6770 | 0.7097 |
| 04 blur+edge | – | 0.6833 | 0.7180 | 0.7287 | 0.7257 | 0.7233 |
| 05 Haar | – | 0.5747 | 0.5887 | 0.5790 | 0.5860 | 0.5967 |
| 06 no repair | 0.05 | 0.5367 | 0.5743 | 0.5687 | 0.5760 | 0.5647 |
| 06 no repair | 0.1 | 0.5317 | 0.5600 | 0.5580 | 0.5693 | 0.5613 |
| 06 no repair | 0.25 | 0.4727 | 0.5097 | 0.5363 | 0.5427 | 0.5273 |
| 06 no repair | 0.5 | 0.2557 | 0.2597 | 0.2417 | 0.2450 | 0.2507 |
| 07 majority | 0.05 | 0.5277 | 0.5437 | 0.5523 | 0.5443 | 0.5513 |
| 07 majority | 0.1 | 0.5220 | 0.5447 | 0.5507 | 0.5470 | 0.5553 |
| 07 majority | 0.25 | 0.4863 | 0.5073 | 0.5157 | 0.5263 | 0.5353 |
| 07 majority | 0.5 | 0.2437 | 0.2400 | 0.2533 | 0.2487 | 0.2510 |
| 08 custom rules (ours) | 0.05 | 0.6347 | 0.6967 | 0.7160 | 0.7210 | 0.7203 |
| 08 custom rules (ours) | 0.1 | 0.6240 | 0.6893 | 0.6983 | 0.7073 | 0.7070 |
| 08 custom rules (ours) | 0.25 | 0.5797 | 0.6237 | 0.6447 | 0.6510 | 0.6453 |
| 08 custom rules (ours) | 0.5 | 0.2457 | 0.2570 | 0.2410 | 0.2613 | 0.2480 |
| 09 MRF | 0.05 | 0.5310 | 0.5413 | 0.5537 | 0.5567 | 0.5533 |
| 09 MRF | 0.1 | 0.5230 | 0.5410 | 0.5550 | 0.5553 | 0.5540 |
| 09 MRF | 0.25 | 0.4787 | 0.4987 | 0.5273 | 0.5383 | 0.5353 |
| 09 MRF | 0.5 | 0.2480 | 0.2537 | 0.2513 | 0.2477 | 0.2433 |

Observations:

- **Representation (S-b).** 01 has the highest mean accuracy (0.8523).
  Among the binary inputs (02–05), 04 is highest (0.7233) and 05 lowest
  (0.5967); 03 (0.7097) is above 02 (0.6190). 02 has the largest SD (0.0272).
- **Noise, p = 0.05–0.25 (S-c).** 08 has the highest mean accuracy of 06–09
  at each of the three ratios (0.7203, 0.7070, 0.6453). The means of 06, 07
  and 09 differ by at most 0.0134 at each of the three ratios.
- **Noise, p = 0.5 (S-c).** Mean accuracy of 06–09 is 0.2433–0.2510; with 4
  classes of 150 images each, predicting one class for every image gives
  0.25.
- **Epochs (S-e).** For 01 the mean accuracy is highest at epoch 30 (0.8653);
  at epoch 50 it is 0.8523.

## Pixel-level restoration study

`experiments/pixel_restoration_study.py`. No classifier. The 600 Test images
are binarized with the baseline binarization (for every method, including the
custom rules, which 08 applies to the edge map instead). For each method and
`p`, the random state is reset to seed 0, noise is added to every image, the
method is applied, and the fraction of pixels that differ from the clean
binary image is averaged over the images; every method therefore sees the
same noisy images at a given `p`. `MRF(β,η)` are four settings with at most
10 sweeps; `MRF(1.5,0.7)` is the one used in 09.

**S-f Residual pixel error rate after repair (600 Test images)** (`results/summary/tables.md`)

| method | p = 0.05 | p = 0.1 | p = 0.25 | p = 0.5 |
| --- | --- | --- | --- | --- |
| none | 0.0498 | 0.0999 | 0.2500 | 0.5000 |
| median 3x3 | 0.0412 | 0.0494 | 0.1146 | 0.4997 |
| majority 3x3 | 0.0440 | 0.0537 | 0.1227 | 0.4997 |
| custom rules (ours) | 0.0232 | 0.0571 | 0.2088 | 0.5000 |
| MRF(1,1) | 0.0303 | 0.0400 | 0.1061 | 0.5005 |
| MRF(1,0.5) | 0.0319 | 0.0409 | 0.1053 | 0.5005 |
| MRF(2,1) | 0.0319 | 0.0409 | 0.1053 | 0.5005 |
| MRF(1.5,0.7) | 0.0319 | 0.0409 | 0.1053 | 0.5005 |

Observations:

- The custom rules have the lowest error rate at `p = 0.05` (0.0232);
  `MRF(1,1)` has the lowest at `p = 0.1` (0.0400), and `MRF(1,0.5)`,
  `MRF(2,1)` and `MRF(1.5,0.7)` the lowest at `p = 0.25` (0.1053). At
  `p` = 0.05, 0.1 and 0.25 every method leaves fewer errors than no repair; at
  `p = 0.5` every method is at 0.4997–0.5005.
- `MRF(1,0.5)`, `MRF(2,1)` and `MRF(1.5,0.7)` give the same value in every
  column. The update depends only on the sign of `η·x + β·s`, where the
  neighbour sum `s` is an integer; for `η/β < 1` (0.5, 0.5 and ≈ 0.4667) a
  pixel takes the sign of `s` whenever `s ≠ 0` and the value of `x` when
  `s = 0`. `MRF(1,1)` (`η/β = 1`) differs only when `|s| = 1`, which happens
  only at border pixels with 3 neighbours.
- The no-repair rate at `p` = 0.05 and 0.1 (0.0498, 0.0999) is below `p`
  because `int(p · 4096)` rounds down (`204/4096 ≈ 0.0498`,
  `409/4096 ≈ 0.0999`).

## Per-pixel loops vs vectorized code

`experiments/vectorization_benchmark.py`, run alone before the training runs
on one CPU thread (`torch.set_num_threads(1)`): 200 Train images drawn with
seed 0, baseline binarization, s&p noise with `p = 0.1`. Each denoiser is
applied to the 200 images twice: as a per-pixel Python loop (the loop
versions are in the script) and as the array implementation in
`rc/denoise.py` (majority filter: a sliding-window sum; custom rules: shifted
views of the whole image; MRF: a whole half-checkerboard per update).
"identical outputs" compares the two outputs for all 200 images.

**S-g Per-pixel loops vs vectorized code (single CPU thread)** (`results/summary/tables.md`)

| method | images | per-pixel (s) | vectorized (s) | speed-up | identical outputs |
| --- | --- | --- | --- | --- | --- |
| majority 3x3 | 200 | 3.16 | 0.0631 | 50 | True |
| custom rules (ours) | 200 | 46.25 | 0.0488 | 948 | True |
| MRF(1.5,0.7) | 200 | 4.47 | 0.2327 | 19 | True |

Scaled to the 8,000 Train images of one epoch (× 40), the measured times of
the custom rules are ≈ 1850 s per-pixel and ≈ 1.952 s vectorized.
