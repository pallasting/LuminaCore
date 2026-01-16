"""
LuminaFlow 硬件感知层模块

提供模拟光子物理特性的 PyTorch 层
"""

from .optical_linear import OpticalLinear
from .complex_linear import ComplexOpticalLinear
from .wdm_mapping import WDMChannelMapper
from .attention import OpticalAttention
from .transformer_block import OpticalTransformerBlock

__all__ = [
    "OpticalLinear", 
    "ComplexOpticalLinear",
    "WDMChannelMapper", 
    "OpticalAttention", 
    "OpticalTransformerBlock"
]
