from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


def _flip_pixels(img_np: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img_np.shape
    num_pixels = h * w
    num_noisy = int(num_pixels * ratio)
    if num_noisy == 0:
        return img_np
    coords = np.random.choice(num_pixels, num_noisy, replace=False)
    y, x = np.unravel_index(coords, (h, w))
    img_np[y, x] = 255 - img_np[y, x]
    return img_np


def _wrap(img_np: np.ndarray, return_type: str):
    if return_type == "pil":
        return Image.fromarray(img_np.astype(np.uint8))
    if return_type == "array":
        return img_np
    raise ValueError(f"Unknown return_type: {return_type!r}")


class AddNoise:
    def __init__(self, noise_ratio: float, return_type: str = "pil"):
        self.noise_ratio = noise_ratio
        self.return_type = return_type

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        out = _flip_pixels(img.copy(), self.noise_ratio)
        return _wrap(out, self.return_type)


class AddRandomNoise:
    def __init__(self, candidate_ratios=(0.05, 0.10, 0.25, 0.50),
                 return_type: str = "pil"):
        self.candidate_ratios = list(candidate_ratios)
        self.return_type = return_type

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        ratio = random.choice(self.candidate_ratios)
        out = _flip_pixels(img.copy(), ratio)
        return _wrap(out, self.return_type)


def _flip_pixels_tensor(tensor_img: torch.Tensor, ratio: float) -> torch.Tensor:
    img = tensor_img.clone()
    h, w = img.shape[1], img.shape[2]
    num_noisy = int(h * w * ratio)
    if num_noisy > 0:
        coords = torch.randperm(h * w)[:num_noisy]
        y = coords // w
        x = coords % w
        img[0, y, x] = 255 - img[0, y, x]
    return img


class AddRandomNoiseTensor:
    def __init__(self, candidate_ratios=(0.05, 0.10, 0.25, 0.50)):
        self.candidate_ratios = list(candidate_ratios)

    def __call__(self, tensor_img: torch.Tensor) -> torch.Tensor:
        return _flip_pixels_tensor(tensor_img, random.choice(self.candidate_ratios))


class AddNoiseTensor:
    def __init__(self, noise_ratio: float):
        self.noise_ratio = noise_ratio

    def __call__(self, tensor_img: torch.Tensor) -> torch.Tensor:
        return _flip_pixels_tensor(tensor_img, self.noise_ratio)
