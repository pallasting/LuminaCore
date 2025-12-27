"""
Weight Quantizer - 权重量化器

负责将连续的 PyTorch 权重映射到基于 HardwareConfig 精度的离散硬件状态。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..layers.optical_components import HardwareConfig
from ..exceptions import ValidationError

class WeightQuantizer:
    """
    WeightQuantizer 类

    将 PyTorch 权重转换为硬件可读的离散整数状态。
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化权重量化器。

        Args:
            config: 硬件配置实例，包含精度信息。
        """
        self.config = config

    def quantize_to_states(self, weights: torch.Tensor) -> torch.Tensor:
        """
        将权重映射到离散的整数状态。

        Args:
            weights: 输入权重张量 (通常归一化到 [0, 1] 或 [-1, 1])。

        Returns:
            量化后的整数状态张量。
        """
        if not isinstance(weights, torch.Tensor):
            raise ValidationError(f"Weights must be a torch.Tensor, got {type(weights)}")

        # 归一化处理：假设权重在 [0, 1] 范围内
        # 如果权重包含负数，通常在光子计算中会映射到不同的相位或差分结构
        # 这里遵循 v0.3 规范，执行线性量化
        clipped_weights = torch.clamp(weights, 0.0, 1.0)
        
        # 映射到 [0, 2^precision - 1]
        max_state = self.config.max_digital_val
        states = torch.round(clipped_weights * max_state).to(torch.int32)
        
        return states

    def generate_lut(self, weights: torch.Tensor) -> Dict[str, Any]:
        """
        生成用于硬件导出的查找表 (LUT) 数据。

        Args:
            weights: 权重张量。

        Returns:
            包含量化状态和元数据的字典。
        """
        states = self.quantize_to_states(weights)
        
        return {
            "precision": self.config.precision,
            "max_state": self.config.max_digital_val,
            "states": states.tolist(),
            "shape": list(states.shape)
        }
