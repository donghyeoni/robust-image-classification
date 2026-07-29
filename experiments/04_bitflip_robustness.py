"""noise-robust training.

Train on the fixed-threshold binary baseline with random-ratio bit-flip noise
(so the model sees noise levels 0.05 / 0.10 / 0.25 / 0.50 during training),
saving checkpoints every ``save_every`` epochs. Then evaluate every checkpoint
across a sweep of fixed test-time noise levels.
"""

import torch
import torch.nn as nn
from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.data import build_dataloaders
from rc.engine import evaluate_checkpoints, train
from rc.model import build_resnet18
from rc.noise import AddNoise, AddRandomNoise
from rc.preprocessing import PreprocessingBaseline


def main():
    parser = build_common_parser("noise-robust training",
                                 default_epochs=50)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.05, 0.10, 0.25, 0.50],
                        help="Fixed test-time noise ratios to sweep.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        PreprocessingBaseline(threshold=128, size=64, return_type="array"),
        AddRandomNoise(return_type="pil"),
        transforms.ToTensor(),
    ])

    _, trainloader, _, _ = build_dataloaders(
        args.data_root, train_transform=train_transform,
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_resnet18(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs,
          save_path=args.checkpoint_dir, save_every=args.save_every)

    # Evaluate saved checkpoints across a sweep of fixed noise levels.
    checkpoint_epochs = range(args.save_every, args.epochs + 1, args.save_every)
    checkpoint_paths = [f"{args.checkpoint_dir}/weight{e}.pth"
                        for e in checkpoint_epochs]

    def model_factory():
        return build_resnet18(in_channels=1, num_classes=args.num_classes)

    for noise in args.noise_levels:
        print(f"\n===== Test noise ratio = {noise} =====")
        test_transform = transforms.Compose([
            PreprocessingBaseline(threshold=128, size=64, return_type="array"),
            AddNoise(noise_ratio=noise, return_type="pil"),
            transforms.ToTensor(),
        ])
        _, _, _, testloader = build_dataloaders(
            args.data_root, train_transform=test_transform,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle_test=True)
        evaluate_checkpoints(model_factory, checkpoint_paths, device,
                             testloader, criterion)


if __name__ == "__main__":
    main()
