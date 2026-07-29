"""Input preprocessing pipelines.

Each module implements one of the input representations studied in the project:

- ``binarize``     : grayscale + fixed-threshold binarization (``Binarize``,
                     ``PreprocessingBaseline``).
- ``edges``        : LoG + Sobel edge fusion -> blur -> binarize
                     (``OurPreprocessing``).
- ``twochannel``   : baseline-blur channel + edge channel stacked (2ch).
- ``wavelet``      : Haar wavelet subbands (LL/LH/HL/HH), each Otsu-binarized (4ch).
- ``jpeg_channel`` : JPEG byte-budget + byte bit-error + denoise + restore (3ch).
"""

from rc.preprocessing.binarize import Binarize, PreprocessingBaseline
from rc.preprocessing.edges import OurPreprocessing
from rc.preprocessing.twochannel import TwoChannelPreprocessing
from rc.preprocessing.wavelet import PreprocessingWavelet
from rc.preprocessing.jpeg_channel import JpegChannelPreprocessing

__all__ = [
    "Binarize",
    "PreprocessingBaseline",
    "OurPreprocessing",
    "TwoChannelPreprocessing",
    "PreprocessingWavelet",
    "JpegChannelPreprocessing",
]
