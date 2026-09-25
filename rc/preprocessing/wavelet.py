from __future__ import annotations

import cv2
import numpy as np
import pywt
import torch
from PIL import Image
from torchvision import transforms


class PreprocessingWavelet:
    def __init__(self, size: int = 32, crop: int = 0,
                 use_channels=("LL", "LH", "HL", "HH")):
        self.size = size
        self.crop = crop
        self.use_channels = list(use_channels)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img.convert("RGB"))
        if self.crop > 0:
            h, w, _ = img_np.shape
            img_np = img_np[self.crop:h - self.crop, self.crop:w - self.crop, :]

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        coeffs2 = pywt.dwt2(gray, "haar")
        LL, (LH, HL, HH) = coeffs2

        bands = {
            "LL": cv2.resize(LL, (self.size, self.size), interpolation=cv2.INTER_CUBIC),
            "LH": cv2.resize(LH, (self.size, self.size), interpolation=cv2.INTER_CUBIC),
            "HL": cv2.resize(HL, (self.size, self.size), interpolation=cv2.INTER_CUBIC),
            "HH": cv2.resize(HH, (self.size, self.size), interpolation=cv2.INTER_CUBIC),
        }

        tensors = []
        for key in self.use_channels:
            band = bands[key]
            band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binary = cv2.threshold(band, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            tensors.append(transforms.functional.to_tensor(Image.fromarray(binary)))

        return torch.cat(tensors, dim=0)
