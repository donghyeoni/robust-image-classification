"""Dataset / DataLoader builders.

The dataset root is supplied by the caller (CLI argument or config) instead of
being hardcoded to a machine-specific path as in the original notebooks
(``/home/dh/venv/dataset/Animals/...``).

Expected on-disk layout (a standard ``torchvision.datasets.ImageFolder``)::

    <root>/
        Train/
            <class_a>/ *.jpg
            <class_b>/ *.jpg
            ...
        Test/
            <class_a>/ *.jpg
            ...
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from torchvision import datasets


def pil_collate_fn(batch):
    """Collate raw ``(PIL.Image, label)`` samples without stacking images.

    Used by pipelines (e.g. the "ours" denoising experiment) that apply their
    per-image preprocessing *inside* the training loop rather than as a
    dataset transform. Returns ``(list_of_images, label_tensor)``.
    """
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def build_dataset(root: str, split: str, transform: Optional[Callable] = None):
    """Build a single ImageFolder for ``split`` in {"Train", "Test"}."""
    path = os.path.join(root, split)
    return datasets.ImageFolder(path, transform=transform)


def build_dataloader(root: str, split: str, transform: Optional[Callable] = None,
                     batch_size: int = 40, shuffle: bool = True,
                     num_workers: int = 2, drop_last: bool = True,
                     collate_fn: Optional[Callable] = None):
    """Build a DataLoader for one split."""
    dataset = build_dataset(root, split, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn,
    )
    return dataset, loader


def build_dataloaders(root: str, train_transform: Optional[Callable] = None,
                      test_transform: Optional[Callable] = None,
                      batch_size: int = 40, num_workers: int = 2,
                      shuffle_test: bool = False,
                      collate_fn: Optional[Callable] = None):
    """Build both Train and Test loaders in one call.

    Returns ``(trainset, trainloader, testset, testloader)``. When
    ``test_transform`` is ``None`` the ``train_transform`` is reused for the
    test split.
    """
    if test_transform is None:
        test_transform = train_transform

    trainset, trainloader = build_dataloader(
        root, "Train", train_transform, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
    )
    testset, testloader = build_dataloader(
        root, "Test", test_transform, batch_size=batch_size,
        shuffle=shuffle_test, num_workers=num_workers, collate_fn=collate_fn,
    )
    return trainset, trainloader, testset, testloader
