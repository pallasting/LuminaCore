"""
WDM Channel Mapper - 波分复用通道映射

实现完整的波分复用系统，包括：
- 波长相关的物理效应建模
- 通道间干扰和串扰
- 自适应通道分配策略
- 与OpticalLinear的深度集成
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WDMChannelMapper(nn.Module):
    """
    增强的波分复用（WDM）通道映射器

    完整实现光子芯片的波分复用系统，包括：
    - 波长相关的物理效应建模（色散、非线性）
    - 通道间干扰和串扰
    - 自适应通道分配策略
    - 与OpticalLinear的深度集成
    - 实时物理参数监控

    设计理念："空间换时间"转变为"色彩换空间"

    Args:
        num_channels: WDM 通道数（默认 3，对应 RGB）
        channel_strategy: 通道分配策略 ('rgb', 'rgbw', 'sequential', 'adaptive')
        enable_crosstalk: 是否启用通道间串扰建模
        enable_dispersion: 是否启用色散效应
        enable_nonlinearity: 是否启用非线性效应
    """

    def __init__(
        self,
        num_channels: int = 3,
        channel_strategy: str = "rgb",
        enable_crosstalk: bool = True,
        enable_dispersion: bool = True,
        enable_nonlinearity: bool = False,
    ):
        super(WDMChannelMapper, self).__init__()

        self.num_channels = num_channels
        self.channel_strategy = channel_strategy
        self.enable_crosstalk = enable_crosstalk
        self.enable_dispersion = enable_dispersion
        self.enable_nonlinearity = enable_nonlinearity

        # 基础通道参数
        self.channel_gains = nn.Parameter(torch.ones(num_channels))
        self.phase_offsets = nn.Parameter(torch.zeros(num_channels))

        # 波长配置
        if channel_strategy == "rgb":
            self.wavelengths = torch.tensor([650.0, 532.0, 450.0])  # 红、绿、蓝 (nm)
        elif channel_strategy == "rgbw":
            self.wavelengths = torch.tensor([650.0, 532.0, 450.0, 550.0])  # 加白色
        elif channel_strategy == "adaptive":
            # 自适应波长分配（根据通道数动态调整）
            center_wavelength = 550.0  # 中心波长
            wavelength_range = 200.0  # 波长范围
            start_wl = center_wavelength - wavelength_range / 2
            end_wl = center_wavelength + wavelength_range / 2
            self.wavelengths = torch.linspace(start_wl, end_wl, num_channels)
        else:
            # 顺序分配（等间距）
            self.wavelengths = torch.linspace(450.0, 650.0, num_channels)

        # 物理效应参数
        # 色散系数 (ps/(nm·km))
        self.dispersion_coeff = nn.Parameter(torch.tensor(17.0))

        # 非线性系数 (1/W·km)
        self.nonlinear_coeff = nn.Parameter(torch.tensor(1.2e-3))

        # 串扰矩阵 (对称矩阵，模拟通道间干扰)
        if enable_crosstalk:
            # 初始化串扰矩阵（对角线为1，其他元素为小的随机值）
            crosstalk_init = torch.randn(num_channels, num_channels) * 0.05
            crosstalk_init.fill_diagonal_(1.0)
            self.crosstalk_matrix = nn.Parameter(crosstalk_init)
        else:
            self.register_parameter("crosstalk_matrix", None)

        # 自适应权重（用于动态调整通道重要性）
        self.adaptive_weights = nn.Parameter(torch.ones(num_channels))

        # 物理参数监控
        self.register_buffer("total_power", torch.tensor(0.0))
        self.register_buffer("crosstalk_level", torch.tensor(0.0))
        self.register_buffer("snr_estimate", torch.tensor(30.0))  # dB

        # 将波长移到设备
        self.wavelengths = self.wavelengths.float()

    def map_to_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入映射到多个 WDM 通道（包含物理效应建模）

        Args:
            x: 输入张量 [batch_size, features]

        Returns:
            多通道张量 [batch_size, num_channels, features]
        """
        batch_size, features = x.shape

        # 基础通道映射
        x_multi = x.unsqueeze(1).expand(batch_size, self.num_channels, features)

        # 应用通道增益和相位偏移
        channel_factors = self.channel_gains.view(1, -1, 1) * torch.exp(
            1j * self.phase_offsets.view(1, -1, 1)
        )

        # 如果输入是实数，转换为复数表示
        if x_multi.is_floating_point():
            x_multi = x_multi.float()
            if not torch.is_complex(x_multi):
                x_multi = x_multi.to(torch.complex64)

        # 应用通道因子
        x_multi = x_multi * channel_factors

        # 色散效应建模
        if self.enable_dispersion:
            x_multi = self._apply_dispersion(x_multi)

        # 非线性效应建模
        if self.enable_nonlinearity:
            x_multi = self._apply_nonlinear_effects(x_multi)

        # 应用自适应权重
        adaptive_factors = torch.sigmoid(self.adaptive_weights).view(1, -1, 1)
        x_multi = x_multi * adaptive_factors

        # 串扰效应建模
        if self.enable_crosstalk and self.crosstalk_matrix is not None:
            x_multi = self._apply_crosstalk(x_multi)

        return x_multi

    def _apply_dispersion(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        应用色散效应

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            色散处理后的张量
        """
        # 模拟色散引起的脉冲展宽
        # 波长差引起的群速度差异
        wavelengths = self.wavelengths.view(1, -1, 1).to(x_multi.device)
        wavelength_diffs = wavelengths - wavelengths.mean()

        # 色散引起的相位偏移
        dispersion_phase = 0.5 * self.dispersion_coeff * (wavelength_diffs**2)

        # 应用色散相位
        x_multi = x_multi * torch.exp(1j * dispersion_phase)

        return x_multi

    def _apply_nonlinear_effects(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        应用非线性效应（自相位调制、交叉相位调制）

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            非线性处理后的张量
        """
        # 计算每个通道的功率
        power_per_channel = torch.abs(x_multi) ** 2

        # 自相位调制 (SPM)
        spm_phase = self.nonlinear_coeff * power_per_channel

        # 交叉相位调制 (XPM) - 简化版本
        other_channels_power = (
            power_per_channel.sum(dim=1, keepdim=True) - power_per_channel
        )
        xpm_phase = 2 * self.nonlinear_coeff * other_channels_power

        # 总相位调制
        total_phase = spm_phase + xpm_phase

        # 应用相位调制
        x_multi = x_multi * torch.exp(1j * total_phase)

        return x_multi

    def _apply_crosstalk(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        应用通道间串扰

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            串扰处理后的张量
        """
        # 确保串扰矩阵与输入张量类型匹配
        if self.crosstalk_matrix is not None:
            if torch.is_complex(x_multi) and not torch.is_complex(self.crosstalk_matrix):
                # 如果输入是复数，确保串扰矩阵也是复数
                crosstalk_matrix = self.crosstalk_matrix.to(x_multi.dtype)
            else:
                crosstalk_matrix = self.crosstalk_matrix
        else:
            return x_multi

        # 串扰矩阵与通道数据相乘
        batch_size, num_channels, features = x_multi.shape
        x_reshaped = x_multi.transpose(
            1, 2
        ).contiguous()  # [batch_size, features, num_channels]

        # 应用串扰矩阵
        x_crosstalk = torch.bmm(
            x_reshaped, crosstalk_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        )
        x_crosstalk = x_crosstalk.transpose(
            1, 2
        ).contiguous()  # [batch_size, num_channels, features]

        return x_crosstalk

    def combine_channels(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        合并多个 WDM 通道的输出（包含解复用和信号恢复）

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            合并后的张量 [batch_size, features]
        """
        # 更新物理参数监控
        self._update_monitoring_params(x_multi)

        # 如果包含复数成分，提取实部
        if torch.is_complex(x_multi):
            x_multi = torch.real(x_multi)

        # 自适应合并策略
        if self.channel_strategy == "adaptive":
            return self._adaptive_channel_combination(x_multi)
        else:
            return self._weighted_channel_combination(x_multi)

    def _weighted_channel_combination(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        加权通道合并

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            合并后的张量
        """
        # 计算权重
        weights = torch.softmax(self.adaptive_weights, dim=0).view(1, -1, 1)

        # 加权合并
        combined = torch.sum(x_multi * weights, dim=1)

        return combined

    def _adaptive_channel_combination(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        自适应通道合并（根据信号质量和通道状况动态调整）

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            合并后的张量
        """
        # 计算每个通道的信噪比估计
        channel_snr = self._estimate_channel_snr(x_multi)

        # 基于SNR的自适应权重
        snr_weights = torch.softmax(channel_snr, dim=1)

        # 应用权重并合并
        weights = snr_weights.unsqueeze(-1)
        combined = torch.sum(x_multi * weights, dim=1)

        return combined

    def _estimate_channel_snr(self, x_multi: torch.Tensor) -> torch.Tensor:
        """
        估计每个通道的信噪比

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]

        Returns:
            SNR估计 [batch_size, num_channels]
        """
        # 计算信号功率
        signal_power = torch.mean(x_multi**2, dim=2)  # [batch_size, num_channels]

        # 估计噪声功率（基于串扰水平）
        noise_power = self.crosstalk_level * signal_power.mean(dim=1, keepdim=True)

        # 计算SNR
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        return snr.real  # 确保返回实数

    def _update_monitoring_params(self, x_multi: torch.Tensor):
        """
        更新物理参数监控

        Args:
            x_multi: 多通道张量 [batch_size, num_channels, features]
        """
        # 更新总功率
        total_power_now = torch.mean(torch.abs(x_multi) ** 2).item()
        self.total_power = 0.9 * self.total_power + 0.1 * torch.tensor(total_power_now)

        # 更新串扰水平
        if self.crosstalk_matrix is not None:
            # 计算非对角线元素的平均值作为串扰水平
            crosstalk_matrix = self.crosstalk_matrix.detach()
            mask = ~torch.eye(
                self.num_channels, device=crosstalk_matrix.device, dtype=torch.bool
            )
            crosstalk_level_now = torch.mean(torch.abs(crosstalk_matrix[mask])).item()
            self.crosstalk_level = 0.9 * self.crosstalk_level + 0.1 * torch.tensor(
                crosstalk_level_now
            )

        # 更新SNR估计
        snr_estimate_now = torch.mean(self._estimate_channel_snr(x_multi)).item()
        self.snr_estimate = 0.9 * self.snr_estimate + 0.1 * torch.tensor(
            snr_estimate_now.real  # 确保返回实数部分
        )

    def forward(self, x: torch.Tensor, mode: str = "both") -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, features]
            mode: 'map' - 映射到多通道, 'combine' - 合并通道, 'both' - 完整WDM处理

        Returns:
            处理后的张量
        """
        if mode == "map":
            return self.map_to_channels(x)
        elif mode == "combine":
            return self.combine_channels(x)
        elif mode == "both":
            mapped = self.map_to_channels(x)
            return self.combine_channels(mapped)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'map', 'combine', or 'both'")

    def forward_integrated(
        self, x: torch.Tensor, optical_linear_module
    ) -> torch.Tensor:
        """
        与OpticalLinear集成的WDM前向传播

        Args:
            x: 输入张量 [batch_size, features]
            optical_linear_module: OpticalLinear模块实例

        Returns:
            集成处理后的张量
        """
        # WDM映射到多通道
        x_wdm = self.map_to_channels(x)

        # 为每个通道应用光学线性变换
        batch_size, num_channels, features = x_wdm.shape
        x_wdm_flat = x_wdm.view(-1, features)  # [batch_size * num_channels, features]

        # 应用光学线性变换
        optical_output_flat = optical_linear_module(x_wdm_flat)
        optical_output = optical_output_flat.view(batch_size, num_channels, -1)

        # WDM合并
        combined_output = self.combine_channels(optical_output)

        return combined_output

    def optimize_channel_allocation(
        self, input_data: torch.Tensor, target_snr: float = 20.0
    ):
        """
        优化通道分配策略

        Args:
            input_data: 输入数据用于分析
            target_snr: 目标信噪比 (dB)
        """
        with torch.no_grad():
            # 分析输入数据的特征
            input_power = torch.var(input_data, dim=1)  # [batch_size]

            # 根据输入功率分布调整通道权重
            # 将批次中的高功率样本映射到更好的通道
            num_channels = len(self.channel_gains)
            
            # 计算每个通道应该分配多少个高功率样本
            samples_per_channel = len(input_power) // num_channels
            
            # 为每个通道分配高功率样本
            for channel_idx in range(num_channels):
                start_idx = channel_idx * samples_per_channel
                end_idx = start_idx + samples_per_channel
                if channel_idx == num_channels - 1:  # 最后一个通道处理剩余样本
                    end_idx = len(input_power)
                
                # 获取该通道的高功率样本
                channel_samples = input_power[start_idx:end_idx]
                if len(channel_samples) > 0:
                    # 高功率通道给予更高增益
                    gain_factor = 1.0 + 0.2 * (1.0 - channel_idx / num_channels)
                    self.channel_gains[channel_idx] = gain_factor

    def get_physical_parameters(self) -> Dict[str, Any]:
        """
        获取当前物理参数

        Returns:
            物理参数字典
        """
        return {
            "wavelengths": self.wavelengths.cpu().numpy(),
            "channel_gains": self.channel_gains.detach().cpu().numpy(),
            "phase_offsets": self.phase_offsets.detach().cpu().numpy(),
            "dispersion_coeff": self.dispersion_coeff.item(),
            "nonlinear_coeff": self.nonlinear_coeff.item(),
            "total_power": self.total_power.item(),
            "crosstalk_level": self.crosstalk_level.item(),
            "snr_estimate": self.snr_estimate.item(),
            "adaptive_weights": torch.sigmoid(self.adaptive_weights)
            .detach()
            .cpu()
            .numpy(),
        }

    def reset_physical_parameters(self):
        """
        重置物理参数到默认值
        """
        self.channel_gains.data.fill_(1.0)
        self.phase_offsets.data.fill_(0.0)
        self.adaptive_weights.data.fill_(0.0)
        if self.crosstalk_matrix is not None:
            crosstalk_init = torch.randn(self.num_channels, self.num_channels) * 0.05
            crosstalk_init.fill_diagonal_(1.0)
            self.crosstalk_matrix.data.copy_(crosstalk_init)

        # 重置监控参数
        self.total_power.fill_(0.0)
        self.crosstalk_level.fill_(0.0)
        self.snr_estimate.fill_(30.0)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, "
            f"strategy={self.channel_strategy}, "
            f"crosstalk={self.enable_crosstalk}, "
            f"dispersion={self.enable_dispersion}, "
            f"nonlinearity={self.enable_nonlinearity}"
        )
