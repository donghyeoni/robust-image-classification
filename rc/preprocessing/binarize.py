"""Grayscale + fixed-threshold binarization pipelines.

``Binarize`` is a drop-in PIL->PIL transform: grayscale, then threshold at a
fixed value, producing a 1-channel binary image. It is meant to be followed by
``transforms.ToTensor()`` in a ``transforms.Compose``.

``PreprocessingBaseline`` is the fixed-threshold binary *baseline* used by the
noise / denoising experiments: resize to a fixed square, convert to grayscale,
and threshold. Its ``return_type`` lets it emit different containers so it can
sit at the head of pipelines that differ downstream:

- ``"array"``      : uint8 ``np.ndarray`` (feeds numpy-domain noise/filters).
- ``"pil"``        : ``PIL.Image``.
- ``"byte_tensor"``: a ``[1, H, W]`` byte tensor with values in {0, 255}
                     (feeds the tensor-domain 'ours' denoising pipeline).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class Binarize:
    """PIL -> PIL: grayscale then fixed-threshold binarize."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img_np = np.array(img)
        img_bin = (img_np > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(img_bin)


class PreprocessingBaseline:
    """Resize -> grayscale -> fixed-threshold binarize.

    This is the classical binary baseline against which the edge / wavelet /
    jpeg representations are compared.
    """

    def __init__(self, threshold: int = 128, size: int = 64,
                 return_type: str = "array"):
        self.threshold = threshold
        self.size = size
        self.return_type = return_type

    def __call__(self, img: Image.Image):
        img_np = np.array(img.convert("RGB"))
        resized = cv2.resize(img_np, (self.size, self.size))
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        _, binarized = cv2.threshold(gray, self.threshold, 255,
                                     cv2.THRESH_BINARY)

        if self.return_type == "array":
            return binarized
        if self.return_type == "pil":
            return Image.fromarray(binarized)
        if self.return_type == "byte_tensor":
            tensor = ToTensor()(Image.fromarray(binarized)) * 255
            return tensor.byte()
        raise ValueError(f"Unknown return_type: {self.return_type!r}")
