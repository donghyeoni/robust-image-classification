"""bandwidth-constrained JPEG channel.

JPEG-compress each image to a 2**16-byte budget, inject a byte-level bit-error
channel, denoise the byte stream, restore, and feed a normalized 3-channel
224x224 input. Checkpoints saved and evaluated.
"""

import torch
import torch.nn as nn
from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.data import build_dataloaders
from rc.engine import evaluate_checkpoints, test, train
from rc.model import build_resnet18
from rc.preprocessing import JpegChannelPreprocessing


def main():
    parser = build_common_parser("JPEG bandwidth channel",
                                 default_epochs=50)
    parser.add_argument("--target-bytes", type=int, default=65536)
    parser.add_argument("--bit-error-rate", type=float, default=0.001)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        JpegChannelPreprocessing(size=224, crop=2,
                                 target_bytes=args.target_bytes,
                                 bit_error_rate=args.bit_error_rate),
    ])

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=transform, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=False)

    model = build_resnet18(in_channels=3, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs,
          save_path=args.checkpoint_dir, save_every=args.save_every)
    test(model, device, testloader, criterion)

    checkpoint_epochs = range(args.save_every, args.epochs + 1, args.save_every)
    checkpoint_paths = [f"{args.checkpoint_dir}/weight{e}.pth"
                        for e in checkpoint_epochs]

    def model_factory():
        return build_resnet18(in_channels=3, num_classes=args.num_classes)

    evaluate_checkpoints(model_factory, checkpoint_paths, device, testloader,
                         criterion)


if __name__ == "__main__":
    main()
