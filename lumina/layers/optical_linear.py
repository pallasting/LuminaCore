"""
OpticalLinear - 光子全连接层

模拟 LuminaCore 芯片的光学矩阵乘法，包含噪声、量化等物理特性
重构版本：使用分离的组件架构
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal

from .optical_components import HardwareConfig, Quantizer, NoiseModel
from ..exceptions import InvalidParameterError, ValidationError, BoundaryError

# 尝试导入 Rust 后端
_RUST_BACKEND_AVAILABLE = False
try:
    import lumina_kernel
    _RUST_BACKEND_AVAILABLE = True
except ImportError:
    lumina_kernel = None

# 全局配置：是否使用 Rust 后端
# 可以通过环境变量 LUMINA_USE_RUST=1 启用
USE_RUST_BACKEND = os.environ.get("LUMINA_USE_RUST", "0") == "1" and _RUST_BACKEND_AVAILABLE


class OpticalLinearFunction(torch.autograd.Function):
    """
    自定义 Autograd 函数，桥接 Rust 算子的前向和反向传播
    """
    @staticmethod
    def forward(ctx, input, weight, bias, noise_level, precision, training):
        ctx.save_for_backward(input, weight, bias)
        
        input_np = input.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        bias_np = bias.detach().cpu().numpy() if bias is not None else None
        
        if training:
            output_np = lumina_kernel.optical_linear_fused(
                input_np, weight_np, bias_np,
                noise_std=noise_level,
                bits=precision,
                seed=np.random.randint(0, 2**32)
            )
        else:
            output_np = lumina_kernel.optical_linear_infer(
                input_np, weight_np, bias_np,
                bits=precision
            )
            
        return torch.from_numpy(output_np).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        
        # 转换为 NumPy
        grad_output_np = grad_output.detach().cpu().numpy()
        input_np = input.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        
        # 调用 Rust 反向传播内核
        grad_input_np, grad_weight_np = lumina_kernel.optical_linear_backward_kernel(
            grad_output_np, input_np, weight_np
        )
        
        grad_input = torch.from_numpy(grad_input_np).to(input.device)
        grad_weight = torch.from_numpy(grad_weight_np).to(weight.device)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
            
        return grad_input, grad_weight, grad_bias, None, None, None


class OpticalLinear(nn.Module):
    """
    模拟光子芯片的光学全连接层

    特性：
    - 硬件感知的噪声注入（光路噪声、探测器噪声）
    - 可配置的量化精度（DAC/ADC 位数）
    - 支持 WDM（波分复用）通道映射
    - 温度漂移模拟
    - 轻量级训练模式（边缘端优化）
    - 批量推理加速（数据中心优化）

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        hardware_profile: 硬件配置预设 ('lumina_nano_v1', 'lumina_micro_v1', 'edge_ultra_low_power', 'datacenter_high_precision', 'custom')
        precision: DAC/ADC 精度位数 (默认 4-bit)
        noise_level: 光路噪声水平 (0.0-1.0, 默认根据 hardware_profile)
        enable_wdm: 是否启用波分复用 (默认 True)
        bias: 是否使用偏置项 (默认 False，光子芯片通常无偏置)
        temp_drift: 温度漂移系数 (0.0-1.0, 默认根据 hardware_profile)
    """

    # 保持向后兼容的硬件配置预设
    HARDWARE_PROFILES: Dict[str, Dict[str, Union[int, float]]] = {
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

    # 类属性类型注解
    in_features: int
    out_features: int
    hardware_profile: str
    enable_wdm: bool
    noise_level: float
    precision: int
    temp_drift: float
    attenuation: float
    max_digital_val: int
    quantization_step: float
    weight: nn.Parameter
    bias: Optional[nn.Parameter]

    def _validate_init_parameters(
        self,
        in_features: int,
        out_features: int,
        precision: Optional[int],
        noise_level: Optional[float],
        temp_drift: Optional[float]
    ) -> None:
        """验证初始化参数"""
        if not isinstance(in_features, int) or in_features <= 0:
            raise InvalidParameterError(f"in_features must be a positive integer, got {in_features}")

        if not isinstance(out_features, int) or out_features <= 0:
            raise InvalidParameterError(f"out_features must be a positive integer, got {out_features}")

        if precision is not None:
            if not isinstance(precision, int) or precision <= 0:
                raise InvalidParameterError(f"precision must be a positive integer, got {precision}")
            if precision > 32:
                raise BoundaryError(f"precision too high: {precision}, maximum supported is 32 bits")

        if noise_level is not None:
            if not isinstance(noise_level, (int, float)) or not (0.0 <= noise_level <= 1.0):
                raise InvalidParameterError(f"noise_level must be a float between 0.0 and 1.0, got {noise_level}")

        if temp_drift is not None:
            if not isinstance(temp_drift, (int, float)) or not (0.0 <= temp_drift <= 1.0):
                raise InvalidParameterError(f"temp_drift must be a float between 0.0 and 1.0, got {temp_drift}")

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        """验证前向传播输入"""
        if not isinstance(x, torch.Tensor):
            raise ValidationError(f"Input must be a torch.Tensor, got {type(x)}")

        if x.dim() < 2:
            raise ValidationError(f"Input tensor must be 2-dimensional, got shape {x.shape}")

        if x.shape[-1] != self.in_features:
            raise ValidationError(f"Input feature dimension {x.shape[-1]} does not match expected {self.in_features}")

        if torch.isnan(x).any():
            raise ValidationError("Input tensor contains NaN values")

        if torch.isinf(x).any():
            raise ValidationError("Input tensor contains infinite values")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hardware_profile: Literal[
            "lumina_nano_v1",
            "lumina_micro_v1",
            "edge_ultra_low_power",
            "datacenter_high_precision",
            "custom",
        ] = "lumina_nano_v1",
        precision: Optional[int] = None,
        noise_level: Optional[float] = None,
        enable_wdm: bool = True,
        bias: bool = False,
        temp_drift: Optional[float] = None,
    ) -> None:
        super(OpticalLinear, self).__init__()

        # 参数验证
        self._validate_init_parameters(in_features, out_features, precision, noise_level, temp_drift)

        self.in_features = in_features
        self.out_features = out_features
        self.hardware_profile = hardware_profile
        self.enable_wdm = enable_wdm

        # 验证硬件配置
        if hardware_profile not in self.HARDWARE_PROFILES:
            raise InvalidParameterError(
                f"Unknown hardware profile: {hardware_profile}. "
                f"Available: {list(self.HARDWARE_PROFILES.keys())}"
            )

        # 创建硬件配置
        profile = self.HARDWARE_PROFILES[hardware_profile]
        config_dict = {
            "noise_level": noise_level if noise_level is not None else profile["noise_level"],
            "precision": precision if precision is not None else profile["precision"],
            "temp_drift": temp_drift if temp_drift is not None else profile["temp_drift"],
            "attenuation": profile["attenuation"]
        }
        
        self.hardware_config = HardwareConfig(**config_dict)
        
        # 向后兼容性属性
        self.noise_level = self.hardware_config.noise_level
        self.precision = self.hardware_config.precision
        self.temp_drift = self.hardware_config.temp_drift
        self.attenuation = self.hardware_config.attenuation
        self.max_digital_val = self.hardware_config.max_digital_val
        self.quantization_step = self.hardware_config.quantization_step

        # 创建组件
        self.quantizer = Quantizer(self.hardware_config)
        self.noise_model = NoiseModel(self.hardware_config, hardware_profile)

        # 权重参数（使用标准 Linear 层作为基础）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # 向后兼容的方法 - 委托给Quantizer
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 DAC/ADC 量化过程
        
        向后兼容方法，委托给Quantizer实例
        """
        return self.quantizer.quantize(x)

    def dac_convert(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 DAC 转换：数字信号 -> 光强信号
        
        向后兼容方法，委托给Quantizer实例
        """
        return self.quantizer.dac_convert(x)

    def adc_convert(self, analog_signal: torch.Tensor) -> torch.Tensor:
        """
        模拟 ADC 转换：光强信号 -> 数字信号
        
        向后兼容方法，委托给Quantizer实例
        """
        return self.quantizer.adc_convert(analog_signal)

    def optical_matrix_multiply(
        self, input_vec: torch.Tensor, weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        模拟光学矩阵乘法（包含物理缺陷）

        物理过程：
        1. 理想矩阵乘法（光干涉叠加）
        2. 温度漂移导致的信号衰减和串扰
        3. 光源散粒噪声（与信号强度相关）
        4. 探测器热噪声（固定底噪）

        Args:
            input_vec: 输入向量 [batch_size, in_features]
            weight_matrix: 权重矩阵 [out_features, in_features]

        Returns:
            带噪声的光学计算结果
        """
        # 理想矩阵乘法
        # 处理复数输入和实数权重的情况
        if torch.is_complex(input_vec) and not torch.is_complex(weight_matrix):
            # 复数输入 × 实数权重 = 复数输出
            real_part = F.linear(input_vec.real, weight_matrix, None)
            imag_part = F.linear(input_vec.imag, weight_matrix, None)
            ideal_result = torch.complex(real_part, imag_part)
        else:
            # 标准情况：都是实数
            ideal_result = F.linear(input_vec, weight_matrix, None)

        # 应用噪声模型
        return self.noise_model.apply_noise(ideal_result, self.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [..., in_features]

        Returns:
            输出张量 [..., out_features]
        """
        # 输入验证
        self._validate_forward_input(x)

        # 处理多维输入
        is_multidim = x.dim() > 2
        if is_multidim:
            original_shape = x.shape
            x = x.reshape(-1, self.in_features)

        # Rust 后端快速路径（如果启用且可用）
        if USE_RUST_BACKEND and not torch.is_complex(x):
            # 使用自定义 Autograd 函数桥接 Rust 后端（支持梯度回传）
            return OpticalLinearFunction.apply(
                x, self.weight, self.bias, 
                self.noise_level, self.precision, self.training
            )
        else:
            # PyTorch 标准路径
            # Step 1: DAC 转换（输入量化）
            opt_input = self.dac_convert(x)

            # Step 2: 光学矩阵乘法（包含噪声）
            opt_output = self.optical_matrix_multiply(opt_input, self.weight)

            # Step 3: ADC 转换（输出量化）
            digital_output = self.adc_convert(opt_output)

            # Step 4: 添加偏置（如果有）
            if self.bias is not None:
                digital_output = digital_output + self.bias
            
            output = digital_output

        # 恢复形状
        if is_multidim:
            output_shape = original_shape[:-1] + (self.out_features,)
            output = output.reshape(output_shape)

        return output
    
    def _forward_rust(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 Rust 后端的前向传播（融合算子）
        
        将 矩阵乘法 + 噪声注入 + 量化 融合为一个 Rust 算子
        """
        # 转换为 NumPy（零拷贝）
        input_np = x.detach().cpu().numpy()
        weight_np = self.weight.detach().cpu().numpy()
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None
        
        # 调用 Rust 融合算子
        if self.training:
            # 训练模式：包含噪声
            output_np = lumina_kernel.optical_linear_fused(
                input_np,
                weight_np,
                bias_np,
                noise_std=self.noise_level,
                bits=self.precision,
                seed=np.random.randint(0, 2**32)
            )
        else:
            # 推理模式：无噪声
            output_np = lumina_kernel.optical_linear_infer(
                input_np,
                weight_np,
                bias_np,
                bits=self.precision
            )
        
        # 转换回 PyTorch
        output = torch.from_numpy(output_np).to(x.device)
        
        # 修复：Rust 路径目前不支持自动微分，必须确保在训练模式下保持计算图
        # 如果需要支持训练，应回退到 PyTorch 路径，除非实现了 Rust 自定义算子的 backward
        if x.requires_grad or self.weight.requires_grad:
            # 这种方式只是占位，实际梯度无法流向输入和权重
            # 在 NAT 训练中，由于调用了 _forward_rust，梯度流断裂
            pass
            
        return output

    def forward_optimized(
        self, x: torch.Tensor, batch_size_threshold: int = 64
    ) -> torch.Tensor:
        """
        优化后的前向传播（数据中心批量处理加速）

        针对大规模批量推理进行优化，减少量化计算开销

        Args:
            x: 输入张量 [batch_size, in_features]
            batch_size_threshold: 批量大小阈值，超过此值启用优化

        Returns:
            输出张量 [batch_size, out_features]
        """
        batch_size = x.shape[0]

        # 小批量：使用标准处理流程
        if batch_size < batch_size_threshold:
            return self.forward(x)

        # 大批量：启用批量优化
        # Step 1: 批量DAC转换（一次性处理整个批次）
        if "datacenter" in self.hardware_profile:
            # 数据中心模式：使用更高效的批量量化
            opt_input = self._batch_dac_convert_optimized(x)
        else:
            opt_input = self.dac_convert(x)

        # Step 2: 批量光学矩阵乘法
        if "datacenter" in self.hardware_profile:
            # 数据中心：使用优化的矩阵乘法
            opt_output = self._batch_optical_matrix_multiply_optimized(
                opt_input, self.weight
            )
        else:
            opt_output = F.linear(opt_input, self.weight, None)

        # Step 3: 批量ADC转换
        if "datacenter" in self.hardware_profile:
            digital_output = self._batch_adc_convert_optimized(opt_output)
        else:
            digital_output = self.adc_convert(opt_output)

        # Step 4: 添加偏置（如果有）
        if self.bias is not None:
            digital_output = digital_output + self.bias

        return digital_output

    def _batch_dac_convert_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        优化的批量DAC转换（数据中心专用）

        使用向量化操作减少计算开销

        Args:
            x: 输入张量

        Returns:
            量化后的张量
        """
        # 使用统一的quantize方法处理实数和复数
        return self.quantize(x)

    def _batch_optical_matrix_multiply_optimized(
        self, input_vec: torch.Tensor, weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        优化的批量光学矩阵乘法（数据中心专用）

        简化物理噪声模型，专注计算性能

        Args:
            input_vec: 输入向量
            weight_matrix: 权重矩阵

        Returns:
            光学计算结果
        """
        # 处理复数输入和实数权重的情况
        if torch.is_complex(input_vec) and not torch.is_complex(weight_matrix):
            # 复数输入 × 实数权重 = 复数输出
            real_part = F.linear(input_vec.real, weight_matrix, None)
            imag_part = F.linear(input_vec.imag, weight_matrix, None)
            ideal_result = torch.complex(real_part, imag_part)
        else:
            # 标准情况：都是实数
            ideal_result = F.linear(input_vec, weight_matrix, None)

        # 应用固定的链路衰减
        return ideal_result * self.attenuation

    def _batch_adc_convert_optimized(self, analog_signal: torch.Tensor) -> torch.Tensor:
        """
        优化的批量ADC转换（数据中心专用）

        使用快速量化算法

        Args:
            analog_signal: 模拟信号

        Returns:
            数字信号
        """
        # 快速饱和和量化
        clipped = torch.clamp(analog_signal, 0, 1.5)
        return torch.round(clipped / self.quantization_step) * self.quantization_step

    def forward_smart(
        self, x: torch.Tensor, batch_size_threshold: int = 64
    ) -> torch.Tensor:
        """
        智能前向传播（自动选择最优实现）

        根据硬件配置和批量大小自动选择最佳的前向传播策略：
        - 数据中心配置 + 大批量：使用批量优化版本
        - 其他情况：使用标准版本

        Args:
            x: 输入张量 [batch_size, in_features]
            batch_size_threshold: 批量大小阈值

        Returns:
            输出张量 [batch_size, out_features]
        """
        batch_size = x.shape[0]

        # 数据中心配置且批量大小超过阈值：使用优化版本
        if "datacenter" in self.hardware_profile and batch_size >= batch_size_threshold:
            return self.forward_optimized(x, batch_size_threshold)
        else:
            return self.forward(x)

    # 新增：组件访问方法
    def get_hardware_config(self) -> HardwareConfig:
        """
        获取硬件配置对象

        Returns:
            HardwareConfig实例
        """
        return self.hardware_config

    def get_quantizer(self) -> Quantizer:
        """
        获取量化器对象

        Returns:
            Quantizer实例
        """
        return self.quantizer

    def get_noise_model(self) -> NoiseModel:
        """
        获取噪声模型对象

        Returns:
            NoiseModel实例
        """
        return self.noise_model

    def extra_repr(self) -> str:
        """返回模块的字符串表示"""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"hardware_profile={self.hardware_profile}, "
            f"precision={self.precision}-bit, "
            f"noise_level={self.noise_level:.2%}, "
            f"wdm={self.enable_wdm}"
        )

    def compile_to_hardware(self) -> Dict[str, Any]:
        """
        将当前层编译为硬件可执行格式。
        
        包含：
        - 权重量化 LUT
        - WDM 映射表（如果启用）
        - 硬件配置元数据
        """
        from ..compiler.quantizer import WeightQuantizer
        from ..compiler.planner import WDMPlanner
        from .wdm_mapping import WDMChannelMapper

        # 1. 权重量化
        quantizer = WeightQuantizer(self.hardware_config)
        lut = quantizer.generate_lut(self.weight.data)

        # 2. WDM 规划 (如果启用)
        wdm_data = None
        if self.enable_wdm:
            mapper = WDMChannelMapper(num_channels=min(self.in_features, 16))
            planner = WDMPlanner(mapper.num_channels)
            wdm_data = planner.export_mapping_table(mapper)

        return {
            "type": "OpticalLinear",
            "layer_id": id(self),
            "hardware_profile": self.hardware_profile,
            "lut": lut,
            "wdm": wdm_data,
            "config": {
                "in_features": self.in_features,
                "out_features": self.out_features,
                "precision": self.precision,
                "noise_level": self.noise_level
            }
        }
