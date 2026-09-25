from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from torchvision import datasets


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def build_dataset(root: str, split: str, transform: Optional[Callable] = None):
    return datasets.ImageFolder(os.path.join(root, split), transform=transform)


def build_dataloader(root: str, split: str, transform: Optional[Callable] = None,
                     batch_size: int = 40, num_workers: int = 8, seed: int = 0,
                     collate_fn: Optional[Callable] = None):
    dataset = build_dataset(root, split, transform)
    train = split == "Train"
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=num_workers, collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )
    return dataset, loader
