from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.denoise import ArrayToTensor, MRFDenoiser
from rc.noise import AddNoise, AddRandomNoise
from rc.preprocessing import PreprocessingBaseline
from rc.runner import NOISE_LEVELS, Pipeline, run


def pipeline(noise):
    return Pipeline(transforms.Compose([
        PreprocessingBaseline(threshold=128, size=64, return_type="array"),
        noise, MRFDenoiser(beta=1.5, eta=0.7, iters=10), ArrayToTensor()]))


def main():
    args = build_common_parser("MRF (Ising / ICM) denoising").parse_args()
    run(args, "09_mrf_denoise", 1, pipeline(AddRandomNoise(return_type="array")),
        {r: pipeline(AddNoise(r, return_type="array")) for r in NOISE_LEVELS})


if __name__ == "__main__":
    main()
