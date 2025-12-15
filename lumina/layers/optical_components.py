"""
Optical Components - 光子计算组件

包含噪声模型、量化器和硬件配置等组件类
"""

from dataclasses import dataclass
from typing import Dict, Union
import torch
import torch.nn as nn

from ..exceptions import InvalidParameterError, ValidationError, BoundaryError


# 硬件配置预设（类级别定义）
HARDWARE_PROFILES = {
    "lumina_nano_v1": {
        "noise_level": 0.15,  # 15% 光路噪声
        "precision": 4,  # 4-bit DAC/ADC
        "temp_drift": 0.05,  # 5% 温度漂移
        "attenuation": 0.85,  # 15% 链路损耗
    },
    "lumina_micro_v1": {
        "noise_level": 0.10,  # 10% 光路噪声
        "precision": 8,  # 8-bit DAC/ADC
        "temp_drift": 0.03,  # 3% 温度漂移
        "attenuation": 0.90,  # 10% 链路损耗
    },
    "edge_ultra_low_power": {
        "noise_level": 0.20,  # 20% 光路噪声
        "precision": 2,  # 2-bit DAC/ADC
        "temp_drift": 0.10,  # 10% 温度漂移
        "attenuation": 0.75,  # 25% 链路损耗
    },
    "datacenter_high_precision": {
        "noise_level": 0.05,  # 5% 光路噪声
        "precision": 12,  # 12-bit DAC/ADC
        "temp_drift": 0.01,  # 1% 温度漂移
        "attenuation": 0.95,  # 5% 链路损耗
    },
    "custom": {
        "noise_level": 0.10,
        "precision": 8,
        "temp_drift": 0.0,
        "attenuation": 1.0,
    },
}


@dataclass
class HardwareConfig:
    """
    硬件配置数据类

    管理光子芯片的硬件参数配置
    """
    noise_level: float
    precision: int
    temp_drift: float
    attenuation: float

    @classmethod
    def from_profile(cls, profile_name: str, **overrides) -> 'HardwareConfig':
        """
        从预设配置创建HardwareConfig实例

        Args:
            profile_name: 预设配置名称
            **overrides: 覆盖的参数

        Returns:
            HardwareConfig实例
        """
        if profile_name not in HARDWARE_PROFILES:
            raise InvalidParameterError(
                f"Unknown hardware profile: {profile_name}. "
                f"Available: {list(HARDWARE_PROFILES.keys())}"
            )

        config_dict = HARDWARE_PROFILES[profile_name].copy()
        config_dict.update(overrides)

        return cls(**config_dict)

    @property
    def max_digital_val(self) -> int:
        """最大数字值"""
        return 2**self.precision - 1

    @property
    def quantization_step(self) -> float:
        """量化步长"""
        return 1.0 / self.max_digital_val


class Quantizer(nn.Module):
    """
    量化器类

    处理DAC/ADC量化逻辑
    """

    def __init__(self, config: HardwareConfig):
        super(Quantizer, self).__init__()
        self.config = config

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 DAC/ADC 量化过程

        Args:
            x: 输入张量 (范围通常在 0-1 或归一化后)
                 支持实数和复数张量，复数张量将按幅度进行量化

        Returns:
            量化后的张量（与输入相同类型）
        """
        # 输入验证
        if not isinstance(x, torch.Tensor):
            raise ValidationError(f"Input must be a torch.Tensor, got {type(x)}")

        if torch.isnan(x).any():
            raise ValidationError("Input tensor contains NaN values")

        if torch.isinf(x).any():
            raise ValidationError("Input tensor contains infinite values")

        # 处理复数张量：按幅度量化，然后保持相位信息
        if torch.is_complex(x):
            # 计算幅度和相位
            magnitude = torch.abs(x)
            phase = x / (magnitude + 1e-8)  # 避免除零

            # 对幅度进行量化
            magnitude_clipped = torch.clamp(magnitude, 0, 1)
            magnitude_quantized = (
                torch.round(magnitude_clipped / self.config.quantization_step)
                * self.config.quantization_step
            )

            # 重建复数结果
            return magnitude_quantized * phase
        else:
            # 处理实数张量
            x_clipped = torch.clamp(x, 0, 1)
            x_quantized = (
                torch.round(x_clipped / self.config.quantization_step) * self.config.quantization_step
            )
            return x_quantized

    def dac_convert(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 DAC 转换：数字信号 -> 光强信号

        Args:
            x: 数字输入

        Returns:
            模拟光强信号
        """
        return self.quantize(x)

    def adc_convert(self, analog_signal: torch.Tensor) -> torch.Tensor:
        """
        模拟 ADC 转换：光强信号 -> 数字信号

        Args:
            analog_signal: 模拟光强信号
                支持实数和复数张量，复数张量将按幅度进行处理

        Returns:
            数字信号
        """
        # 处理复数张量：按幅度进行饱和处理
        if torch.is_complex(analog_signal):
            # 计算幅度并进行饱和处理
            magnitude = torch.abs(analog_signal)
            magnitude_clipped = torch.clamp(magnitude, 0, 1.5)

            # 保持相位信息，只对幅度进行量化
            phase = analog_signal / (magnitude + 1e-8)  # 避免除零
            clipped_signal = magnitude_clipped * phase

            # 量化
            return self.quantize(clipped_signal)
        else:
            # 处理实数张量
            clipped = torch.clamp(analog_signal, 0, 1.5)
            return self.quantize(clipped)


class NoiseModel(nn.Module):
    """
    噪声模型类

    处理各种物理噪声注入逻辑
    """

    def __init__(self, config: HardwareConfig, hardware_profile: str):
        super(NoiseModel, self).__init__()
        self.config = config
        self.hardware_profile = hardware_profile

    def apply_noise(self, signal: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        应用噪声到信号

        Args:
            signal: 输入信号
            training: 是否在训练模式

        Returns:
            带噪声的信号
        """
        if not training:
            # 推理时只应用衰减
            return signal * self.config.attenuation

        # 训练时注入物理噪声
        noisy_signal = self._apply_temperature_drift(signal)
        noisy_signal = self._apply_shot_noise(noisy_signal)
        noisy_signal = self._apply_thermal_noise(noisy_signal)

        # 应用链路衰减
        return noisy_signal * self.config.attenuation

    def _apply_temperature_drift(self, signal: torch.Tensor) -> torch.Tensor:
        """
        应用温度漂移效应

        Args:
            signal: 输入信号

        Returns:
            带温度漂移的信号
        """
        if "edge" in self.hardware_profile:
            return self._edge_temperature_drift(signal)
        else:
            # 标准温度漂移
            drift_loss = 1.0 - (self.config.temp_drift * 0.8)
            drift_crosstalk = torch.mean(signal) * self.config.temp_drift * 0.5
            return signal * drift_loss + drift_crosstalk

    def _edge_temperature_drift(self, signal: torch.Tensor) -> torch.Tensor:
        """
        针对边缘端增强温度漂移模型

        Args:
            signal: 输入信号

        Returns:
            带增强温度漂移的信号
        """
        # 模拟温度波动
        temp_variation = torch.randn_like(signal) * 0.1

        # 折射率变化导致的相位漂移噪声
        phase_noise = temp_variation * self.config.temp_drift * 2.0

        # 热膨胀导致的几何畸变
        geometric_distortion = torch.randn_like(signal) * self.config.temp_drift * 0.5

        # 综合衰减
        drift_loss = 1.0 - (self.config.temp_drift * 0.8 + torch.abs(temp_variation) * 0.3)

        # 串扰增强
        drift_crosstalk = (
            (torch.mean(signal) + phase_noise.mean()) * self.config.temp_drift * 0.7
        )

        return signal * drift_loss + drift_crosstalk + geometric_distortion

    def _apply_shot_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
        应用光源散粒噪声

        Args:
            signal: 输入信号

        Returns:
            带散粒噪声的信号
        """
        shot_noise_std = self.config.noise_level * torch.sqrt(torch.abs(signal) + 1e-6)
        shot_noise = torch.randn_like(signal) * shot_noise_std
        return signal + shot_noise

    def _apply_thermal_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
        应用探测器热噪声

        Args:
            signal: 输入信号

        Returns:
            带热噪声的信号
        """
        if "datacenter" in self.hardware_profile:
            return signal + self._datacenter_thermal_noise(signal)
        else:
            # 标准热噪声
            detector_noise = torch.randn_like(signal) * 0.005
            return signal + detector_noise

    def _datacenter_thermal_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
        针对数据中心细化热噪声模型

        Args:
            signal: 当前信号强度

        Returns:
            热噪声张量
        """
        batch_size, out_features = signal.shape

        # 模拟阵列尺寸
        array_size = int(out_features ** 0.5)

        # 创建热分布矩阵
        x = torch.linspace(-1, 1, array_size, device=signal.device)
        y = torch.linspace(-1, 1, array_size, device=signal.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # 热分布：中心热点，边缘低
        thermal_profile = torch.exp(-(X**2 + Y**2) / 0.5)
        thermal_profile = thermal_profile / thermal_profile.max()

        # 展平为向量
        thermal_profile = thermal_profile.flatten()

        # 扩展到完整特征维度
        if len(thermal_profile) < out_features:
            repeats = (out_features + len(thermal_profile) - 1) // len(thermal_profile)
            thermal_profile = thermal_profile.repeat(repeats)[:out_features]
        elif len(thermal_profile) > out_features:
            thermal_profile = thermal_profile[:out_features]

        # 热噪声强度与温度相关
        thermal_noise_std = 0.005 * (1 + thermal_profile * 2.0)

        # 生成位置相关的热噪声
        thermal_noise = torch.randn_like(signal) * thermal_noise_std.unsqueeze(0)

        # 添加局部热点噪声
        if torch.is_complex(signal):
            hotspot_mask_real = torch.rand(signal.shape, device=signal.device) < 0.05
            hotspot_mask_imag = torch.rand(signal.shape, device=signal.device) < 0.05
            hotspot_noise_real = torch.randn(signal.shape, device=signal.device) * 0.02 * hotspot_mask_real.float()
            hotspot_noise_imag = torch.randn(signal.shape, device=signal.device) * 0.02 * hotspot_mask_imag.float()
            hotspot_noise = torch.complex(hotspot_noise_real, hotspot_noise_imag)
        else:
            hotspot_mask = torch.rand_like(signal) < 0.05
            hotspot_noise = torch.randn_like(signal) * 0.02 * hotspot_mask.float()

        return thermal_noise + hotspot_noise