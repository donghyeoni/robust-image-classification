"""prob2_2 (baseline): classical denoising of noisy binary input.

Train on the fixed-threshold binary baseline with random bit-flip noise,
denoised by a classical morphology-style filter (median / connected-component
/ majority). Checkpoints are saved and then evaluated across a noise sweep.

The ``--denoiser`` flag selects which classical filter is applied.
"""

import torch
import torch.nn as nn
from torchvision import transforms

from rc.config import build_common_parser
from rc.data import build_dataloaders
from rc.denoise import ArrayToTensor, ComponentFilter, MajorityFilter, MedianFilter
from rc.engine import evaluate_checkpoints, train
from rc.model import build_resnet18
from rc.noise import AddNoise, AddRandomNoise
from rc.preprocessing import PreprocessingBaseline

DENOISERS = {
    "median": lambda: MedianFilter(ksize=3),
    "component": lambda: ComponentFilter(min_area=30),
    "majority": lambda: MajorityFilter(ksize=3),
}


def main():
    parser = build_common_parser("prob2_2 baseline denoising", default_epochs=50)
    parser.add_argument("--denoiser", choices=list(DENOISERS), default="majority")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.05, 0.10, 0.25, 0.50])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    denoiser = DENOISERS[args.denoiser]

    train_transform = transforms.Compose([
        PreprocessingBaseline(threshold=128, size=64, return_type="array"),
        AddRandomNoise(return_type="array"),
        denoiser(),
        ArrayToTensor(),
    ])

    _, trainloader, _, _ = build_dataloaders(
        args.data_root, train_transform=train_transform,
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_resnet18(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs,
          save_path=args.checkpoint_dir, save_every=args.save_every)

    checkpoint_epochs = range(args.save_every, args.epochs + 1, args.save_every)
    checkpoint_paths = [f"{args.checkpoint_dir}/weight{e}.pth"
                        for e in checkpoint_epochs]

    def model_factory():
        return build_resnet18(in_channels=1, num_classes=args.num_classes)

    for noise in args.noise_levels:
        print(f"\n===== Test noise ratio = {noise} =====")
        test_transform = transforms.Compose([
            PreprocessingBaseline(threshold=128, size=64, return_type="array"),
            AddNoise(noise_ratio=noise, return_type="array"),
            denoiser(),
            ArrayToTensor(),
        ])
        _, _, _, testloader = build_dataloaders(
            args.data_root, train_transform=test_transform,
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle_test=True)
        evaluate_checkpoints(model_factory, checkpoint_paths, device,
                             testloader, criterion)


if __name__ == "__main__":
    main()
