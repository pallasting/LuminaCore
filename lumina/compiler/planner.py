"""
WDM Planner - 波分复用规划器

负责波长分配和串扰补偿逻辑，将逻辑通道映射到物理波长。
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from ..layers.wdm_mapping import WDMChannelMapper
from ..exceptions import ValidationError

class WDMPlanner:
    """
    WDMPlanner 类

    管理光子芯片上的波长资源分配，并生成串扰补偿矩阵。
    """

    def __init__(self, num_channels: int = 16):
        """
        初始化 WDM 规划器。

        Args:
            num_channels: 最大支持的通道数。
        """
        self.num_channels = num_channels
        # 默认波长范围 (nm)
        self.wavelength_start = 450.0
        self.wavelength_end = 650.0

    def plan_wavelengths(self, strategy: str = "sequential") -> torch.Tensor:
        """
        根据策略规划波长分配。

        Args:
            strategy: 分配策略 ('sequential', 'equidistant')。

        Returns:
            分配的波长张量。
        """
        if strategy in ["sequential", "equidistant"]:
            return torch.linspace(self.wavelength_start, self.wavelength_end, self.num_channels)
        else:
            raise ValidationError(f"Unsupported WDM strategy: {strategy}")

    def generate_crosstalk_compensation(self, crosstalk_matrix: torch.Tensor) -> torch.Tensor:
        """
        生成串扰补偿矩阵。
        
        逻辑：如果 Y = C * X (C 是串扰矩阵)，那么补偿后的输入 X' = C^-1 * X
        使得输出 Y' = C * X' = C * C^-1 * X = X。

        Args:
            crosstalk_matrix: 物理测量的串扰矩阵 [num_channels, num_channels]。

        Returns:
            补偿矩阵（串扰矩阵的逆）。
        """
        if crosstalk_matrix.shape[0] != crosstalk_matrix.shape[1]:
            raise ValidationError("Crosstalk matrix must be square")
            
        # 使用伪逆以提高稳定性
        compensation_matrix = torch.linalg.pinv(crosstalk_matrix)
        return compensation_matrix

    def export_mapping_table(self, mapper: WDMChannelMapper) -> Dict[str, Any]:
        """
        从 WDMChannelMapper 导出硬件映射表。

        Args:
            mapper: 已配置的 WDMChannelMapper 实例。

        Returns:
            包含波长、增益和串扰补偿的字典。
        """
        params = mapper.get_physical_parameters()
        
        crosstalk_matrix = torch.from_numpy(params["crosstalk_matrix"]) if "crosstalk_matrix" in params else None
        compensation = None
        if crosstalk_matrix is not None:
            compensation = self.generate_crosstalk_compensation(crosstalk_matrix).tolist()

        return {
            "num_channels": mapper.num_channels,
            "wavelengths": params["wavelengths"].tolist(),
            "channel_gains": params["channel_gains"].tolist(),
            "crosstalk_compensation": compensation,
            "strategy": mapper.channel_strategy
        }
