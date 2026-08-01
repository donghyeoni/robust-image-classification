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


def _neighbours(t: torch.Tensor):
    """Return the 8 shifted views aligned with the interior of ``t``.

    Order is row-major over the 3x3 window, centre excluded — the same order the
    original per-pixel implementation produced with
    ``cat((patch.flatten()[:4], patch.flatten()[5:]))``.
    """
    return (t[..., :-2, :-2], t[..., :-2, 1:-1], t[..., :-2, 2:],
            t[..., 1:-1, :-2],                   t[..., 1:-1, 2:],
            t[..., 2:, :-2],  t[..., 2:, 1:-1],  t[..., 2:, 2:])


def mismatch(tensor_img: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Flip a pixel whose entire neighbourhood differs from it.

    A pixel that disagrees with *all* of its 8 neighbours is treated as an
    isolated flip and inverted. Accepts a 2-D ``H x W`` byte tensor or any
    batched ``... x H x W`` stack; the border is left untouched.

    ``radius`` other than 1 falls back to the original per-pixel loop.
    """
    if radius != 1:
        return _mismatch_loop(tensor_img, radius)

    centre = tensor_img[..., 1:-1, 1:-1]
    all_differ = torch.ones_like(centre, dtype=torch.bool)
    for n in _neighbours(tensor_img):
        all_differ &= n != centre

    output = tensor_img.clone()
    output[..., 1:-1, 1:-1] = torch.where(all_differ, 255 - centre, centre)
    return output


def _mismatch_loop(tensor_img: torch.Tensor, radius: int) -> torch.Tensor:
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
    straight (N/S/E/W) neighbours differ, the centre is inverted. Accepts a 2-D
    ``H x W`` byte tensor or any batched ``... x H x W`` stack; the border is
    left untouched.
    """
    nw, n, ne, w_, e, sw, s, se = _neighbours(tensor_img)
    centre = tensor_img[..., 1:-1, 1:-1]

    diag_matches = ((nw == centre).int() + (ne == centre).int()
                    + (sw == centre).int() + (se == centre).int())
    straight_differ = ((n != centre).int() + (s != centre).int()
                       + (w_ != centre).int() + (e != centre).int())
    flip = (diag_matches == 1) & (straight_differ == 4)

    output = tensor_img.clone()
    output[..., 1:-1, 1:-1] = torch.where(flip, 255 - centre, centre)
    return output
