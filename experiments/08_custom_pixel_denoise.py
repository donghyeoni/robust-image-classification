from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.denoise import diagonal_solo, mismatch
from rc.noise import AddNoiseTensor, AddRandomNoiseTensor
from rc.preprocessing import OurPreprocessing
from rc.runner import NOISE_LEVELS, Pipeline, run


def repair(X, y, device):
    X = diagonal_solo(mismatch(X[:, 0].to(device)))
    return X.unsqueeze(1).float() / 255, y.to(device)


def pipeline(noise):
    return Pipeline(transforms.Compose([
        OurPreprocessing(size=64, threshold=128, crop=2, method="Otsu",
                         return_type="byte_tensor"),
        noise]), batch_fn=repair)


def main():
    args = build_common_parser("custom pixel rules (ours)").parse_args()
    run(args, "08_custom_pixel_denoise", 1, pipeline(AddRandomNoiseTensor()),
        {r: pipeline(AddNoiseTensor(r)) for r in NOISE_LEVELS})


if __name__ == "__main__":
    main()
