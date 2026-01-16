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
    支持智能波长规划与通道压缩。
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
        self.wavelength_grid = torch.linspace(self.wavelength_start, self.wavelength_end, 128)

    def plan_wavelengths(self, strategy: str = "sequential") -> torch.Tensor:
        """
        根据策略规划波长分配。
        """
        if strategy in ["sequential", "equidistant"]:
            return torch.linspace(self.wavelength_start, self.wavelength_end, self.num_channels)
        else:
            raise ValidationError(f"Unsupported WDM strategy: {strategy}")

    def optimize_allocation(self, crosstalk_model: torch.Tensor) -> torch.Tensor:
        """
        串扰敏感分配 (Crosstalk-Aware Allocation)。
        
        目标：选择波长组合，使得通道间的互感最小。
        简单实现：贪心搜索，每次选择与已选波长互感之和最小的下一个波长。
        
        Args:
            crosstalk_model: 预估的波长间串扰模型矩阵 [grid_size, grid_size]。
                           c[i, j] 表示波长 i 对波长 j 的干扰强度。
        
        Returns:
            优化后的波长张量 [num_channels]。
        """
        grid_size = crosstalk_model.shape[0]
        selected_indices = [0] # 初始选择第一个波长
        
        for _ in range(1, self.num_channels):
            min_interference = float('inf')
            best_idx = -1
            
            for i in range(grid_size):
                if i in selected_indices:
                    continue
                
                # 计算当前候选波长与所有已选波长的干扰总和
                current_interference = sum(crosstalk_model[idx, i] + crosstalk_model[i, idx] for idx in selected_indices)
                
                if current_interference < min_interference:
                    min_interference = current_interference
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
        
        # 将索引映射回波长
        selected_wavelengths = self.wavelength_grid[torch.tensor(selected_indices)]
        return selected_wavelengths

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
