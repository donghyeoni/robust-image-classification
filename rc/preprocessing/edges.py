"""Edge-map ('ours') preprocessing: LoG + Sobel fusion -> blur -> binarize.

Pipeline:
1. Optional border crop, convert to grayscale.
2. Compute a Laplacian-of-Gaussian (LoG) edge magnitude and a Sobel gradient
   magnitude; normalize each to ``[0, 1]``.
3. Fuse them with automatic weights proportional to each map's mean strength
   (stronger edge response -> larger weight), then rescale to ``[0, 255]``.
4. Gaussian blur, resize to ``size x size``.
5. Binarize (Fixed / Otsu / Adaptive).

``return_type`` selects the output container so the same transform can feed
either a standard ToTensor pipeline (``"tensor"``), a numpy noise pipeline
(``"array"``), or the tensor-domain denoiser (``"byte_tensor"``).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor


class OurPreprocessing:
    def __init__(self, size: int = 64, threshold: int = 128, crop: int = 2,
                 method: str = "Otsu", return_type: str = "tensor"):
        self.size = size
        self.threshold = threshold
        self.crop = crop
        self.method = method  # 'Fixed', 'Otsu', 'Adaptive'
        self.return_type = return_type

    def _binarize(self, resized: np.ndarray) -> np.ndarray:
        if self.method == "Fixed":
            _, binarized = cv2.threshold(resized, self.threshold, 255,
                                         cv2.THRESH_BINARY)
        elif self.method == "Otsu":
            _, binarized = cv2.threshold(resized, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.method == "Adaptive":
            binarized = cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)
        else:
            raise ValueError(f"Unknown binarization method: {self.method}")
        return binarized

    def __call__(self, img: Image.Image):
        # 1. PIL -> NumPy (RGB), optional crop, grayscale.
        img_np = np.array(img.convert("RGB"))
        if self.crop > 0:
            h, w, _ = img_np.shape
            img_np = img_np[self.crop:h - self.crop, self.crop:w - self.crop, :]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 2. Edge extraction: LoG + Sobel.
        log_edge = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edge = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 3. Normalize + automatic-weight fusion.
        log_norm = cv2.normalize(log_edge, None, 0, 1.0, cv2.NORM_MINMAX)
        sobel_norm = cv2.normalize(sobel_edge, None, 0, 1.0, cv2.NORM_MINMAX)
        log_weight = np.mean(log_norm)
        sobel_weight = np.mean(sobel_norm)
        total = log_weight + sobel_weight
        w1 = log_weight / total
        w2 = sobel_weight / total
        edge = w1 * log_norm + w2 * sobel_norm
        edge = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4. Blur + resize.
        blurred = cv2.GaussianBlur(edge, (3, 3), 1)
        resized = cv2.resize(blurred, (self.size, self.size),
                             interpolation=cv2.INTER_CUBIC).astype(np.uint8)

        # 5. Binarize.
        binarized = self._binarize(resized)

        # 6. Emit requested container.
        if self.return_type == "array":
            return binarized
        if self.return_type == "pil":
            return Image.fromarray(binarized)
        if self.return_type == "tensor":
            return transforms.functional.to_tensor(Image.fromarray(binarized))
        if self.return_type == "byte_tensor":
            tensor = ToTensor()(Image.fromarray(binarized)) * 255
            return tensor.byte()
        raise ValueError(f"Unknown return_type: {self.return_type!r}")
