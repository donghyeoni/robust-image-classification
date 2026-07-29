"""prob2_3_2: 4-channel Haar wavelet subbands.

Haar DWT subbands LL/LH/HL/HH, each Otsu-binarized and stacked to a 4-channel
32x32 input. Checkpoints saved and evaluated.
"""

import torch
import torch.nn as nn
from torchvision import transforms

from rc.config import build_common_parser
from rc.data import build_dataloaders
from rc.engine import evaluate_checkpoints, test, train
from rc.model import build_resnet18
from rc.preprocessing import PreprocessingWavelet


def main():
    parser = build_common_parser("prob2_3_2 Haar wavelet subbands",
                                 default_epochs=50)
    parser.add_argument("--channels", nargs="+",
                        default=["LL", "LH", "HL", "HH"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        PreprocessingWavelet(size=32, crop=2, use_channels=args.channels),
    ])

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=transform, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=False)

    in_channels = len(args.channels)
    model = build_resnet18(in_channels=in_channels,
                           num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs,
          save_path=args.checkpoint_dir, save_every=args.save_every)
    test(model, device, testloader, criterion)

    checkpoint_epochs = range(args.save_every, args.epochs + 1, args.save_every)
    checkpoint_paths = [f"{args.checkpoint_dir}/weight{e}.pth"
                        for e in checkpoint_epochs]

    def model_factory():
        return build_resnet18(in_channels=in_channels,
                              num_classes=args.num_classes)

    evaluate_checkpoints(model_factory, checkpoint_paths, device, testloader,
                         criterion)


if __name__ == "__main__":
    main()
