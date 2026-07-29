"""Two-channel preprocessing: baseline-blur channel + edge channel.

Stacks two 1-channel binary maps into a ``(2, H, W)`` tensor:

- channel 0 (baseline): grayscale -> Gaussian blur -> resize -> Otsu binarize.
- channel 1 (ours)    : LoG + Sobel edge fusion -> blur -> resize -> Otsu binarize.

The two internal transforms below are the size/crop-parameterised variants used
specifically for the 2-channel experiment (both return a ``(1, H, W)`` tensor).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class _BaselineBlurChannel:
    """Grayscale -> Gaussian blur -> resize -> Otsu binarize -> (1, H, W)."""

    def __init__(self, size: int, crop: int):
        self.size = size
        self.crop = crop

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img.convert("RGB"))
        if self.crop > 0:
            h, w, _ = img_np.shape
            img_np = img_np[self.crop:h - self.crop, self.crop:w - self.crop, :]

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        resized = cv2.resize(blurred, (self.size, self.size),
                             interpolation=cv2.INTER_CUBIC)
        _, binarized = cv2.threshold(resized, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return transforms.functional.to_tensor(Image.fromarray(binarized))


class _EdgeChannel:
    """LoG + Sobel fusion -> blur -> resize -> Otsu binarize -> (1, H, W)."""

    def __init__(self, size: int, crop: int):
        self.size = size
        self.crop = crop

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img.convert("RGB"))
        if self.crop > 0:
            h, w, _ = img_np.shape
            img_np = img_np[self.crop:h - self.crop, self.crop:w - self.crop, :]

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        log_edge = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edge = np.sqrt(sobelx ** 2 + sobely ** 2)

        log_norm = cv2.normalize(log_edge, None, 0, 1.0, cv2.NORM_MINMAX)
        sobel_norm = cv2.normalize(sobel_edge, None, 0, 1.0, cv2.NORM_MINMAX)
        log_weight = np.mean(log_norm)
        sobel_weight = np.mean(sobel_norm)
        total = log_weight + sobel_weight
        w1 = log_weight / total
        w2 = sobel_weight / total
        edge = w1 * log_norm + w2 * sobel_norm
        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blurred = cv2.GaussianBlur(edge, (3, 3), 1)
        resized = cv2.resize(blurred, (self.size, self.size),
                             interpolation=cv2.INTER_CUBIC)
        _, binarized = cv2.threshold(resized, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return transforms.functional.to_tensor(Image.fromarray(binarized))


class TwoChannelPreprocessing:
    """Stack the baseline-blur and edge channels into a ``(2, H, W)`` tensor."""

    def __init__(self, size: int = 45, crop: int = 2):
        self.base = _BaselineBlurChannel(size, crop)
        self.ours = _EdgeChannel(size, crop)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        base_tensor = self.base(img)   # (1, H, W)
        ours_tensor = self.ours(img)   # (1, H, W)
        return torch.cat([base_tensor, ours_tensor], dim=0)  # (2, H, W)
