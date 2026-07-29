"""Robust Image Classification package.

Shared building blocks for training a ResNet-18 classifier on a 4-class
animal dataset and studying its robustness under aggressive preprocessing,
bit-budget / bandwidth limits, and channel / bit-flip noise.

Submodules
----------
- ``rc.data``    : ImageFolder + DataLoader builders (dataset root is configurable).
- ``rc.engine``  : single shared ``train`` / ``test`` / ``evaluate_checkpoints`` loop.
- ``rc.model``   : ``build_resnet18`` factory (re-stemmed conv1 + 4-way fc head).
- ``rc.noise``   : bit-flip noise transforms (``AddNoise``, ``AddRandomNoise``, ...).
- ``rc.denoise`` : classical denoisers + custom pixel rules.
- ``rc.preprocessing`` : the various input pipelines (binarize / edges / 2ch / wavelet / jpeg).
"""

from rc.model import build_resnet18

__all__ = ["build_resnet18"]
