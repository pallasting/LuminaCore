"""
LuminaFlow SDK - 光子计算时代的 CUDA

Train once, survive the noise. Build for the speed of light.
"""

__version__ = "0.1.0-alpha"

# 为了兼容性，提供 nn 作为 layers 的别名
import sys

from . import layers
from . import layers as _layers
from . import optim, viz

sys.modules[__name__ + ".nn"] = _layers

__all__ = ["layers", "nn", "optim", "viz"]
