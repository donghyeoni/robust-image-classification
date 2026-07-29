"""Denoising of noisy binary images.

Two families:

1. Classical morphology-style filters (the *baseline* denoisers):
   ``MedianFilter``, ``ComponentFilter`` (connected-component area filter) and
   ``MajorityFilter`` (local majority vote). These operate on a single-channel
   uint8 ``np.ndarray``.

2. Custom pixel rules (the *ours* denoisers): ``mismatch`` and
   ``diagonal_solo``. These operate on a 2-D byte ``torch.Tensor`` (``H x W``)
   and remove isolated / diagonally-isolated flipped pixels.

``ArrayToTensor`` is a small helper that turns a uint8 ``np.ndarray`` into a
CxHxW float tensor via PIL (used to close a numpy-domain baseline pipeline).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class MedianFilter:
    """Median blur (removes salt-and-pepper noise)."""

    def __init__(self, ksize: int = 3):
        self.ksize = ksize

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(img_np, self.ksize)


class ComponentFilter:
    """Drop connected components smaller than ``min_area`` pixels."""

    def __init__(self, min_area: int = 30):
        self.min_area = min_area

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            img_np, connectivity=8)
        filtered = np.zeros_like(img_np)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                filtered[labels == i] = 255
        return filtered


class MajorityFilter:
    """Set each pixel to the majority value in its ``ksize x ksize`` window."""

    def __init__(self, ksize: int = 3):
        self.ksize = ksize

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        h, w = img_np.shape
        pad = self.ksize // 2
        padded = np.pad(img_np, pad, mode="constant", constant_values=0)
        filtered = np.zeros_like(img_np)
        for y in range(h):
            for x in range(w):
                window = padded[y:y + self.ksize, x:x + self.ksize]
                count_white = np.sum(window == 255)
                count_black = self.ksize * self.ksize - count_white
                filtered[y, x] = 255 if count_white > count_black else 0
        return filtered


class ArrayToTensor:
    """Convert a uint8 ``np.ndarray`` to a float tensor via PIL + ToTensor."""

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        return transforms.ToTensor()(Image.fromarray(img_np))


def mismatch(tensor_img: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Flip a pixel whose entire neighbourhood differs from it.

    A pixel that disagrees with *all* of its 8 neighbours is treated as an
    isolated flip and inverted. Operates on a 2-D byte tensor.
    """
    h, w = tensor_img.shape
    output = tensor_img.clone()
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            patch = tensor_img[i - radius:i + radius + 1,
                               j - radius:j + radius + 1]
            center = tensor_img[i, j]
            neighbors = torch.cat((patch.flatten()[:4], patch.flatten()[5:]))
            if torch.all(neighbors != center):
                output[i, j] = 255 - center
    return output


def diagonal_solo(tensor_img: torch.Tensor) -> torch.Tensor:
    """Flip a pixel connected to the foreground only through one diagonal.

    If exactly one of the 4 diagonal neighbours matches the centre while all 4
    straight (N/S/E/W) neighbours differ, the centre is inverted. Operates on
    a 2-D byte tensor.
    """
    h, w = tensor_img.shape
    output = tensor_img.clone()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = tensor_img[i, j]
            diag = [tensor_img[i - 1, j - 1], tensor_img[i - 1, j + 1],
                    tensor_img[i + 1, j - 1], tensor_img[i + 1, j + 1]]
            straight = [tensor_img[i - 1, j], tensor_img[i + 1, j],
                        tensor_img[i, j - 1], tensor_img[i, j + 1]]
            if (sum([d == center for d in diag]) == 1
                    and sum([s != center for s in straight]) == 4):
                output[i, j] = 255 - center
    return output
