from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class MedianFilter:
    def __init__(self, ksize: int = 3):
        self.ksize = ksize

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(img_np, self.ksize)


class MajorityFilter:
    def __init__(self, ksize: int = 3):
        self.ksize = ksize

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        k = self.ksize
        white = np.pad(img_np == 255, k // 2).astype(np.int32)
        windows = np.lib.stride_tricks.sliding_window_view(white, (k, k))
        count = windows.sum(axis=(-1, -2))
        return np.where(2 * count > k * k, 255, 0).astype(img_np.dtype)


class MRFDenoiser:
    def __init__(self, beta: float = 1.5, eta: float = 0.7, iters: int = 10):
        self.beta = beta
        self.eta = eta
        self.iters = iters

    @staticmethod
    def _neighbour_sum(y: np.ndarray) -> np.ndarray:
        s = np.zeros_like(y)
        s[1:, :] += y[:-1, :]
        s[:-1, :] += y[1:, :]
        s[:, 1:] += y[:, :-1]
        s[:, :-1] += y[:, 1:]
        return s

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        x = np.where(img_np > 127, 1.0, -1.0).astype(np.float32)
        y = x.copy()

        h, w = y.shape
        yy, xx = np.mgrid[0:h, 0:w]
        checker = (yy + xx) % 2

        for _ in range(self.iters):
            changed = False
            for parity in (0, 1):
                field = self.eta * x + self.beta * self._neighbour_sum(y)
                new = np.where(field > 0, 1.0, np.where(field < 0, -1.0, y))
                mask = checker == parity
                if not np.array_equal(new[mask], y[mask]):
                    changed = True
                y[mask] = new[mask]
            if not changed:
                break

        return np.where(y > 0, 255, 0).astype(np.uint8)


class ArrayToTensor:
    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        return transforms.ToTensor()(Image.fromarray(img_np))


def _neighbours(t: torch.Tensor):
    return (t[..., :-2, :-2], t[..., :-2, 1:-1], t[..., :-2, 2:],
            t[..., 1:-1, :-2],                   t[..., 1:-1, 2:],
            t[..., 2:, :-2],  t[..., 2:, 1:-1],  t[..., 2:, 2:])


def mismatch(tensor_img: torch.Tensor) -> torch.Tensor:
    centre = tensor_img[..., 1:-1, 1:-1]
    all_differ = torch.ones_like(centre, dtype=torch.bool)
    for n in _neighbours(tensor_img):
        all_differ &= n != centre

    output = tensor_img.clone()
    output[..., 1:-1, 1:-1] = torch.where(all_differ, 255 - centre, centre)
    return output


def diagonal_solo(tensor_img: torch.Tensor) -> torch.Tensor:
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
