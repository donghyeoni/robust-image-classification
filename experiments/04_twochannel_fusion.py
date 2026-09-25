import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.preprocessing import TwoChannelPreprocessing
from rc.runner import Pipeline, run


def main():
    args = build_common_parser("two-channel fusion").parse_args()
    t = TwoChannelPreprocessing(size=45, crop=2)
    run(args, "04_twochannel_fusion", 2, Pipeline(t), {None: Pipeline(t)})


if __name__ == "__main__":
    main()
