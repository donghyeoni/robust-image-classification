import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.preprocessing import PreprocessingWavelet
from rc.runner import Pipeline, run


def main():
    args = build_common_parser("Haar wavelet subbands").parse_args()
    t = PreprocessingWavelet(size=32, crop=2, use_channels=["LL", "LH", "HL", "HH"])
    run(args, "05_wavelet_subbands", 4, Pipeline(t), {None: Pipeline(t)})


if __name__ == "__main__":
    main()
