from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.runner import Pipeline, run


def main():
    args = build_common_parser("RGB baseline", default_batch_size=20).parse_args()
    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    run(args, "01_rgb_baseline", 3, Pipeline(t), {None: Pipeline(t)})


if __name__ == "__main__":
    main()
