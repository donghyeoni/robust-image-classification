from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision.transforms import Resize, ToTensor


class Binarize:
    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img_np = np.array(img)
        img_bin = (img_np > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(img_bin)


class PreprocessingBaseline:
    def __init__(self, threshold: int = 128, size: int = 64,
                 return_type: str = "array"):
        self.threshold = threshold
        self.size = size
        self.return_type = return_type

    def __call__(self, img: Image.Image):
        resized = Resize((self.size, self.size))(img)
        binarized = np.array(Binarize(self.threshold)(resized))

        if self.return_type == "array":
            return binarized
        if self.return_type == "pil":
            return Image.fromarray(binarized)
        if self.return_type == "byte_tensor":
            tensor = ToTensor()(Image.fromarray(binarized)) * 255
            return tensor.byte()
        raise ValueError(f"Unknown return_type: {self.return_type!r}")
