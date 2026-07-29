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

| Script | Notebook | What it does | Input channels |
| --- | --- | --- | --- |
| `prob1_1_rgb.py` | `prob1_1` | RGB baseline reference accuracy (256x256) | 3 (RGB) |
| `prob1_2_binary.py` | `prob1_2` | Grayscale + fixed-threshold binarization (64x64) | 1 (binary) |
| `prob1_3_edge.py` | `prob1_3` | "Ours": LoG+Sobel edge fusion -> blur -> Otsu binarize (64x64) | 1 (binary edge) |
| `prob2_1_noise_robust.py` | `prob2_1` | Robustness: train with random-ratio bit-flip noise, test over noise 0.05-0.50, checkpoints | 1 (binary) |
| `prob2_2_base_denoise.py` | `prob2_2_base` | Baseline denoising of noisy binary via classical morphology (median / connected-component / majority) | 1 (binary) |
| `prob2_2_ours_denoise.py` | `prob2_2_ours` | "Ours" denoising: custom pixel rules `mismatch()` + `diagonal_solo()` inside the train loop | 1 (binary) |
| `prob2_3_1_twochannel.py` | `prob2_3_1` | 2-channel fusion: baseline-blur channel + edge channel (45x45) | 2 |
| `prob2_3_2_wavelet.py` | `prob2_3_2` | 4-channel Haar wavelet subbands (LL/LH/HL/HH), each Otsu-binarized (32x32) | 4 |
| `prob2_3_3_jpeg_channel.py` | `prob2_3_3` | Bandwidth sim: JPEG to 2^16-byte budget + byte bit-error + denoise + restore, normalized RGB (224x224) | 3 (RGB) |

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
├── experiments/                 # one thin script per problem
│   ├── prob1_1_rgb.py
│   ├── prob1_2_binary.py
│   ├── prob1_3_edge.py
│   ├── prob2_1_noise_robust.py
│   ├── prob2_2_base_denoise.py
│   ├── prob2_2_ours_denoise.py
│   ├── prob2_3_1_twochannel.py
│   ├── prob2_3_2_wavelet.py
│   └── prob2_3_3_jpeg_channel.py
├── notebooks/                   # original Jupyter notebooks (as submitted)
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

Run any experiment as a module from the repository root so that the `rc`
package is importable:

```bash
# 1. RGB baseline
python -m experiments.prob1_1_rgb --data-root /path/to/Animals

# 2. Binarized input
python -m experiments.prob1_2_binary --data-root /path/to/Animals

# 3. Edge-map ("ours") preprocessing
python -m experiments.prob1_3_edge --data-root /path/to/Animals

# 4. Noise-robust training + noise sweep (checkpoints saved to --checkpoint-dir)
python -m experiments.prob2_1_noise_robust \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/prob2_1

# 5. Baseline denoising (choose the classical filter)
python -m experiments.prob2_2_base_denoise \
    --data-root /path/to/Animals --denoiser majority \
    --checkpoint-dir checkpoints/prob2_2_base

# 6. "Ours" denoising (custom pixel rules)
python -m experiments.prob2_2_ours_denoise \
    --data-root /path/to/Animals --binarizer ours \
    --checkpoint-dir checkpoints/prob2_2_ours

# 7. Two-channel fusion
python -m experiments.prob2_3_1_twochannel \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/prob2_3_1

# 8. Haar wavelet subbands
python -m experiments.prob2_3_2_wavelet \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/prob2_3_2

# 9. Bandwidth-constrained JPEG channel
python -m experiments.prob2_3_3_jpeg_channel \
    --data-root /path/to/Animals --checkpoint-dir checkpoints/prob2_3_3
```

Common options (see `rc/config.py`): `--epochs`, `--batch-size`, `--lr`,
`--num-workers`, `--save-every`, `--num-classes`, `--checkpoint-dir`. Each
script also exposes a few experiment-specific flags (e.g. `--noise-levels`,
`--denoiser`, `--binarizer`, `--channels`, `--target-bytes`,
`--bit-error-rate`). The default hyper-parameters match the original notebooks.

## Notes

- **Checkpoints and datasets are gitignored.** Trained weights (`*.pth`,
  `checkpoints/`, `prob*_weight/`), datasets (`data/`, `Animals/`) and
  generated image outputs (`*.png`) are not tracked — see `.gitignore`.
- The original Jupyter notebooks are preserved unchanged under `notebooks/`
  for reference; the runnable code lives in `rc/` + `experiments/`.
- The project report (`docs/Robust Image Classification under Bandwidth
  Constraints.pdf`) documents the methodology and results in detail. No
  metrics are reproduced in this README to avoid restating numbers out of
  context.
