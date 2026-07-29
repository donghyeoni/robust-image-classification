# Robust Image Classification under Bandwidth Constraints

Train a ResNet-18 classifier on a 4-class animal dataset and study how its
accuracy holds up under aggressive preprocessing, tight bit-budget / bandwidth
limits, and channel / bit-flip noise.

The project starts from a standard RGB baseline and progressively degrades or
transforms the input representation — binarization, edge maps, wavelet
subbands, and a simulated bandwidth-limited JPEG channel — measuring the
accuracy trade-off at each step. Where noise is involved, both classical
morphology denoisers and a set of custom pixel-repair rules are compared.

All nine experiments share a single training/evaluation engine and a single
ResNet-18 factory (the stem `conv1` is re-built for 1/2/3/4 input channels and
the classifier head is a 4-way linear layer).

## Experiments

The nine experiments fall into three themes: **input representations** (01–03),
**noise robustness and denoising** (04–06), and **multi-channel and
bandwidth-constrained inputs** (07–09).

| Script | What it does | Input channels |
| --- | --- | --- |
| `01_rgb_baseline.py` | RGB baseline reference accuracy (256x256) | 3 (RGB) |
| `02_binarized_input.py` | Grayscale + fixed-threshold binarization (64x64) | 1 (binary) |
| `03_edge_preprocessing.py` | LoG+Sobel edge fusion -> blur -> Otsu binarize (64x64) | 1 (binary edge) |
| `04_bitflip_robustness.py` | Train with random-ratio bit-flip noise, test over noise 0.05-0.50, checkpoints | 1 (binary) |
| `05_morphological_denoise.py` | Denoise noisy binary input via classical morphology (median / connected-component / majority) | 1 (binary) |
| `06_custom_pixel_denoise.py` | Custom pixel-repair rules `mismatch()` + `diagonal_solo()` inside the train loop | 1 (binary) |
| `07_twochannel_fusion.py` | 2-channel fusion: blur channel + edge channel (45x45) | 2 |
| `08_wavelet_subbands.py` | 4-channel Haar wavelet subbands (LL/LH/HL/HH), each Otsu-binarized (32x32) | 4 |
| `09_jpeg_channel_budget.py` | Bandwidth sim: JPEG to 2^16-byte budget + byte bit-error + denoise + restore, normalized RGB (224x224) | 3 (RGB) |

## Dataset

The experiments expect a generic 4-class **"Animals"** image-classification
set arranged as a standard `torchvision.datasets.ImageFolder`, with separate
`Train/` and `Test/` splits:

```
<data-root>/
    Train/
        <class_1>/  *.jpg
        <class_2>/  *.jpg
        <class_3>/  *.jpg
        <class_4>/  *.jpg
    Test/
        <class_1>/  *.jpg
        ...
```

The dataset is **not included** in this repository. Supply your own
ImageFolder-structured data (any generic 4-class animal set works; class names
are read from the subfolder names) and pass its location via `--data-root`.

## Project structure

```
robust-image-classification/
├── rc/                          # reusable package
│   ├── __init__.py
│   ├── config.py                # shared CLI argument parser
│   ├── data.py                  # ImageFolder + DataLoader builders (root is configurable)
│   ├── engine.py                # single shared train() / test() / evaluate_checkpoints()
│   ├── model.py                 # build_resnet18(in_channels, num_classes=4)
│   ├── noise.py                 # AddNoise, AddRandomNoise, AddRandomNoiseTensor (bit-flip)
│   ├── denoise.py               # MedianFilter, ComponentFilter, MajorityFilter, mismatch(), diagonal_solo()
│   └── preprocessing/
│       ├── __init__.py
│       ├── binarize.py          # Binarize, PreprocessingBaseline
│       ├── edges.py             # OurPreprocessing (LoG+Sobel fusion)
│       ├── twochannel.py        # TwoChannelPreprocessing
│       ├── wavelet.py           # PreprocessingWavelet (Haar subbands)
│       └── jpeg_channel.py      # JpegChannelPreprocessing (JPEG byte-budget channel)
├── experiments/                 # one thin script per experiment
│   ├── 01_rgb_baseline.py
│   ├── 02_binarized_input.py
│   ├── 03_edge_preprocessing.py
│   ├── 04_bitflip_robustness.py
│   ├── 05_morphological_denoise.py
│   ├── 06_custom_pixel_denoise.py
│   ├── 07_twochannel_fusion.py
│   ├── 08_wavelet_subbands.py
│   └── 09_jpeg_channel_budget.py
├── notebooks/                   # original Jupyter notebooks
├── docs/                        # project report (PDF)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically when available; otherwise the code
falls back to CPU.

## Usage

Run any experiment from the repository root (each script adds the repo root to
`sys.path`, so the `rc` package is importable):

```bash
# 1. RGB baseline
python experiments/01_rgb_baseline.py --data-root /path/to/Animals

# 2. Binarized input
python experiments/02_binarized_input.py --data-root /path/to/Animals

# 3. Edge-map preprocessing
python experiments/03_edge_preprocessing.py --data-root /path/to/Animals

# 4. Bit-flip noise-robust training + noise sweep (checkpoints saved to --checkpoint-dir)
python experiments/04_bitflip_robustness.py \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/bitflip

# 5. Morphological denoising (choose the classical filter)
python experiments/05_morphological_denoise.py \
    --data-root /path/to/Animals --denoiser majority \
    --checkpoint-dir checkpoints/morphological

# 6. Custom pixel-rule denoising
python experiments/06_custom_pixel_denoise.py \
    --data-root /path/to/Animals --binarizer ours \
    --checkpoint-dir checkpoints/custom_pixel

# 7. Two-channel fusion
python experiments/07_twochannel_fusion.py \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/twochannel

# 8. Haar wavelet subbands
python experiments/08_wavelet_subbands.py \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/wavelet

# 9. Bandwidth-constrained JPEG channel
python experiments/09_jpeg_channel_budget.py \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/jpeg_channel
```

Common options (see `rc/config.py`): `--epochs`, `--batch-size`, `--lr`,
`--num-workers`, `--save-every`, `--num-classes`, `--checkpoint-dir`. Each
script also exposes a few experiment-specific flags (e.g. `--noise-levels`,
`--denoiser`, `--binarizer`, `--channels`, `--target-bytes`,
`--bit-error-rate`). The default hyper-parameters match the original notebooks.

## Notes

- **Checkpoints and datasets are gitignored.** Trained weights (`*.pth`,
  `checkpoints/`, `*_weight/`), datasets (`data/`, `Animals/`) and
  generated image outputs (`*.png`) are not tracked — see `.gitignore`.
- The original Jupyter notebooks are preserved unchanged under `notebooks/`
  for reference; the runnable code lives in `rc/` + `experiments/`.
- The project report (`docs/Robust Image Classification under Bandwidth
  Constraints.pdf`) documents the methodology and results in detail. No
  metrics are reproduced in this README to avoid restating numbers out of
  context.
