import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.preprocessing import OurPreprocessing
from rc.runner import Pipeline, run


def main():
    args = build_common_parser("edge-map preprocessing").parse_args()
    t = OurPreprocessing(size=64, threshold=128, crop=2, method="Otsu",
                         return_type="tensor")
    run(args, "03_edge_preprocessing", 1, Pipeline(t), {None: Pipeline(t)})


if __name__ == "__main__":
    main()
