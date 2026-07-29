"""'ours' edge-map preprocessing.

LoG + Sobel edge fusion -> Gaussian blur -> Otsu binarize, single channel at
64x64.
"""

import torch
import torch.nn as nn
from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.data import build_dataloaders
from rc.engine import test, train
from rc.model import build_resnet18
from rc.preprocessing import OurPreprocessing


def main():
    parser = build_common_parser("edge-map preprocessing",
                                 default_epochs=47, default_batch_size=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        OurPreprocessing(size=64, threshold=128, crop=2, method="Otsu",
                         return_type="tensor"),
    ])

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=transform, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=False)

    model = build_resnet18(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs)
    test(model, device, testloader, criterion)


if __name__ == "__main__":
    main()
