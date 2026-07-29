"""1-channel binarized input.

Grayscale + fixed-threshold binarization at 64x64, single input channel.
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
from rc.preprocessing import Binarize


def main():
    parser = build_common_parser("binarized input", default_epochs=40,
                                 default_batch_size=20)
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        Binarize(threshold=args.threshold),
        transforms.ToTensor(),
    ])

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=transform, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=True)

    model = build_resnet18(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs)
    test(model, device, testloader, criterion)


if __name__ == "__main__":
    main()
