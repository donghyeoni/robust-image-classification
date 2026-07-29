"""Bandwidth-constrained JPEG channel simulation.

Models transmitting an image over a bandwidth-limited, noisy byte channel:

1. Optional border crop.
2. JPEG-compress to fit a byte budget (``target_bytes``, e.g. 2**16). A binary
   search over the JPEG quality factor finds the highest quality whose encoded
   size fits the budget; if the result is smaller, it is zero-padded to the
   budget.
3. Inject a byte-level bit-error channel: each byte is XOR-corrupted with
   probability ``bit_error_rate``.
4. "Denoise" the byte stream with a length-3 moving average.
5. Attempt to decode the (possibly corrupted) JPEG; on failure fall back to the
   original crop.
6. Resize to ``size x size`` and apply ImageNet normalization -> ``(3, H, W)``.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from torchvision import transforms


class JpegChannelPreprocessing:
    def __init__(self, size: int = 224, crop: int = 0,
                 target_bytes: int = 65536, bit_error_rate: float = 0.001):
        self.size = size
        self.crop = crop
        self.target_bytes = target_bytes  # e.g. 2**16
        self.bit_error_rate = bit_error_rate

    def compress_to_target_size(self, img_np: np.ndarray, target_size: int) -> bytearray:
        """Binary-search the JPEG quality that best fills ``target_size`` bytes."""
        low, high = 1, 95
        best_data = None
        buffer = io.BytesIO()

        while low <= high:
            mid = (low + high) // 2
            buffer = io.BytesIO()
            Image.fromarray(img_np).save(buffer, format="JPEG", quality=mid)
            data = bytearray(buffer.getvalue())

            if len(data) > target_size:
                high = mid - 1
            else:
                best_data = data
                low = mid + 1

        if best_data is not None and len(best_data) < target_size:
            best_data += bytearray([0] * (target_size - len(best_data)))
        return best_data if best_data else bytearray(buffer.getvalue())

    def __call__(self, img: Image.Image):
        # 1. PIL -> NumPy, optional crop.
        img_np = np.array(img.convert("RGB"))
        if self.crop > 0:
            h, w, _ = img_np.shape
            img_np = img_np[self.crop:h - self.crop, self.crop:w - self.crop, :]

        # 2. JPEG-compress to the byte budget.
        jpeg_data = self.compress_to_target_size(img_np, self.target_bytes)

        # 3. Byte-level bit-error channel.
        for i in range(len(jpeg_data)):
            if np.random.rand() < self.bit_error_rate:
                jpeg_data[i] ^= np.random.randint(1, 256)

        # 4. Denoise (length-3 moving average over the byte stream).
        byte_array = np.array(jpeg_data, dtype=np.uint8)
        kernel = np.ones(3, dtype=np.uint8) / 3
        denoised_bytes = np.convolve(byte_array, kernel, mode="same").astype(np.uint8)

        # 5. Attempt to decode; fall back to the original crop on failure.
        try:
            restored_img = Image.open(io.BytesIO(denoised_bytes.tobytes())).convert("RGB")
        except Exception:
            restored_img = Image.fromarray(img_np)

        # 6. Resize + ImageNet normalization.
        resized = restored_img.resize((self.size, self.size))
        tensor = transforms.ToTensor()(resized)
        norm_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(tensor)
        return norm_tensor
