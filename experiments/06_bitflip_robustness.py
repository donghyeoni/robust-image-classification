from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.noise import AddNoise, AddRandomNoise
from rc.preprocessing import PreprocessingBaseline
from rc.runner import NOISE_LEVELS, Pipeline, run


def pipeline(noise):
    return Pipeline(transforms.Compose([
        PreprocessingBaseline(threshold=128, size=64, return_type="array"),
        noise, transforms.ToTensor()]))


def main():
    args = build_common_parser("noise-robust training").parse_args()
    run(args, "06_bitflip_robustness", 1, pipeline(AddRandomNoise(return_type="pil")),
        {r: pipeline(AddNoise(r, return_type="pil")) for r in NOISE_LEVELS})


if __name__ == "__main__":
    main()
