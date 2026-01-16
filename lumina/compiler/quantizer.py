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
    支持 HWAQ (Hardware-Wide Adaptive Quantization)。
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化权重量化器。

        Args:
            config: 硬件配置实例，包含精度信息。
        """
        self.config = config
        self.scale = 1.0
        self.zero_point = 0
        self.is_calibrated = False

    def calibrate(self, weights: torch.Tensor, method: str = "max_abs") -> Dict[str, float]:
        """
        根据权重分布和硬件约束校准量化参数。

        Args:
            weights: 用于校准的权重张量。
            method: 校准方法 ('max_abs', 'percentile')。

        Returns:
            校准后的参数字典 (scale, zero_point)。
        """
        if method == "max_abs":
            max_val = torch.max(torch.abs(weights)).item()
            # 考虑硬件衰减，留出 10% 的 Headroom
            self.scale = (max_val / self.config.attenuation) * 1.1 if max_val > 0 else 1.0
        elif method == "percentile":
            # 使用 99.9% 分位数避免异常值影响
            flattened = torch.abs(weights).flatten()
            if flattened.numel() > 0:
                k = int(flattened.numel() * 0.999)
                max_val = torch.kthvalue(flattened, max(1, k)).values.item()
                self.scale = (max_val / self.config.attenuation) * 1.05
            else:
                self.scale = 1.0
        
        # 针对光子计算，通常 zero_point 为 0 (因为光强非负)
        # 但在差分检测架构中，我们可以支持有符号映射
        self.zero_point = 0
        self.is_calibrated = True
        
        return {"scale": self.scale, "zero_point": self.zero_point}

    def quantize_to_states(self, weights: torch.Tensor) -> torch.Tensor:
        """
        将权重映射到离散的整数状态。

        Args:
            weights: 输入权重张量。

        Returns:
            量化后的整数状态张量。
        """
        if not isinstance(weights, torch.Tensor):
            raise ValidationError(f"Weights must be a torch.Tensor, got {type(weights)}")

        # 如果未校准，默认使用简单映射
        if not self.is_calibrated:
            self.calibrate(weights)

        # 线性量化公式: Q = round(W / scale)
        # 映射到 [0, 2^precision - 1]
        normalized_weights = torch.clamp(weights / (self.scale + 1e-9), 0.0, 1.0)
        max_state = self.config.max_digital_val
        states = torch.round(normalized_weights * max_state).to(torch.int32)
        
        return states

    def dequantize(self, states: torch.Tensor) -> torch.Tensor:
        """
        从整数状态还原为模拟权重值。
        """
        max_state = self.config.max_digital_val
        return (states.to(torch.float32) / max_state) * self.scale

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
            "scale": self.scale,
            "zero_point": self.zero_point,
            "is_calibrated": self.is_calibrated,
            "states": states.tolist(),
            "shape": list(states.shape)
        }
