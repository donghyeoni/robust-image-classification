"""3-channel RGB baseline.

Reference accuracy for a ResNet-18 trained on 256x256 RGB inputs.
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


def main():
    parser = build_common_parser("RGB baseline", default_epochs=40,
                                 default_batch_size=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=transform, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=True)

    model = build_resnet18(in_channels=3, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs)
    test(model, device, testloader, criterion)


if __name__ == "__main__":
    main()
