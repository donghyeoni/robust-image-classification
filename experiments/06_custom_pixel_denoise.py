"""custom pixel-rule denoising inside the training loop.

Unlike the other experiments, preprocessing/denoising here runs *per image
inside* the loop rather than as a dataset transform, so the dataset yields raw
PIL images (via ``pil_collate_fn``) and a ``batch_fn`` turns each batch into a
model input tensor:

    binarize -> add bit-flip noise -> mismatch() -> diagonal_solo()

``mismatch`` and ``diagonal_solo`` are the custom rules that remove isolated /
diagonally-isolated flipped pixels. ``--binarizer`` selects the front-end
binarization (edge-map 'ours' or fixed-threshold 'baseline').
"""

import torch
import torch.nn as nn

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from rc.config import build_common_parser
from rc.data import build_dataloaders, pil_collate_fn
from rc.denoise import diagonal_solo, mismatch
from rc.engine import evaluate_checkpoints, train
from rc.model import build_resnet18
from rc.noise import AddNoise, AddNoiseTensor, AddRandomNoiseTensor
from rc.preprocessing import OurPreprocessing, PreprocessingBaseline


def make_binarizer(kind):
    if kind == "ours":
        return OurPreprocessing(size=64, threshold=128, crop=2, method="Otsu",
                                return_type="byte_tensor")
    return PreprocessingBaseline(threshold=128, size=64, return_type="byte_tensor")


def main():
    parser = build_common_parser("'ours' denoising", default_epochs=50)
    parser.add_argument("--binarizer", choices=["ours", "baseline"], default="ours")
    parser.add_argument("--test-noise", type=float, default=0.1,
                        help="Fixed test-time noise ratio.")
    parser.add_argument("--legacy-test-pipeline", action="store_true",
                        help="Reproduce the original notebook's test path, which "
                             "re-ran the binarizer over the already-binarized "
                             "noisy image. This does not match how the model was "
                             "trained and collapses accuracy to chance; kept only "
                             "for reproducing the archived numbers.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    binarizer = make_binarizer(args.binarizer)
    train_noise = AddRandomNoiseTensor()

    def train_batch_fn(images, labels, device):
        # Binarize + noise per image (cv2/PIL domain), then run both pixel rules
        # over the whole batch at once — mismatch/diagonal_solo are batched.
        batch = [train_noise(binarizer(img))[0] for img in images]
        X = torch.stack(batch).to(device)             # [B, H, W] byte
        X = diagonal_solo(mismatch(X))
        return X.unsqueeze(1).float(), labels.to(device)

    legacy_noise = AddNoise(noise_ratio=args.test_noise, return_type="pil")
    test_noise = AddNoiseTensor(noise_ratio=args.test_noise)

    def test_batch_fn(images, labels, device):
        batch = []
        for img in images:
            t = binarizer(img)                    # [1, H, W] byte
            if args.legacy_test_pipeline:
                # Original path: back to numpy, add noise, then push the
                # *already binary* image through the binarizer a second time.
                # OurPreprocessing runs LoG+Sobel+Otsu, so this produces a
                # representation the model never saw while training.
                t = binarizer(legacy_noise(t[0].cpu().numpy()))
            else:
                # Match the training path: noise stays in the binary domain.
                t = test_noise(t)
            batch.append(t[0])
        X = torch.stack(batch).to(device)          # [B, H, W] byte
        X = diagonal_solo(mismatch(X))
        return X.unsqueeze(1).float(), labels.to(device)

    _, trainloader, _, testloader = build_dataloaders(
        args.data_root, train_transform=None, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle_test=True,
        collate_fn=pil_collate_fn)

    model = build_resnet18(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, device, trainloader, optimizer, criterion, args.epochs,
          save_path=args.checkpoint_dir, save_every=args.save_every,
          batch_fn=train_batch_fn)

    checkpoint_epochs = range(args.save_every, args.epochs + 1, args.save_every)
    checkpoint_paths = [f"{args.checkpoint_dir}/weight{e}.pth"
                        for e in checkpoint_epochs]

    def model_factory():
        return build_resnet18(in_channels=1, num_classes=args.num_classes)

    evaluate_checkpoints(model_factory, checkpoint_paths, device, testloader,
                         criterion, batch_fn=test_batch_fn)


if __name__ == "__main__":
    main()
