"""
ComplexOpticalLinear - 复数域光子全连接层

支持相干光学计算 (Coherent Computing)，处理复数权重和输入。
包装了 Rust 后端的 complex_matmul 算子。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Union
from typing_extensions import Literal

from .optical_components import HardwareConfig, Quantizer, NoiseModel
from ..exceptions import ValidationError

# 尝试导入 Rust 后端
_RUST_BACKEND_AVAILABLE = False
try:
    import lumina_kernel
    _RUST_BACKEND_AVAILABLE = True
except ImportError:
    lumina_kernel = None

class ComplexOpticalLinearFunction(torch.autograd.Function):
    """
    针对复数矩阵乘法的自定义 Autograd 函数
    """
    @staticmethod
    def forward(ctx, input, weight, use_rust):
        ctx.save_for_backward(input, weight)
        ctx.use_rust = use_rust
        
        if use_rust and _RUST_BACKEND_AVAILABLE:
            input_np = input.detach().cpu().numpy()
            weight_np = weight.detach().cpu().numpy()
            output_np = lumina_kernel.complex_matmul(input_np, weight_np)
            return torch.from_numpy(output_np).to(input.device)
        else:
            # PyTorch fallback (支持梯度)
            return torch.matmul(input, weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        # 复数反向传播逻辑
        # Y = X * W^T
        # dL/dX = dL/dY * W
        # dL/dW = (dL/dY)^T * X
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t().conj(), input)
        
        return grad_input, grad_weight, None

class ComplexOpticalLinear(nn.Module):
    """
    复数域光学线性层
    
    Args:
        in_features: 输入维度
        out_features: 输出维度
        hardware_profile: 硬件配置
        use_rust: 是否强制使用 Rust 加速
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hardware_profile: str = "lumina_nano_v1",
        use_rust: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hardware_profile = hardware_profile
        self.use_rust = use_rust
        
        # 复数权重初始化 (实部和虚部独立初始化)
        self.weight = nn.Parameter(torch.complex(
            torch.randn(out_features, in_features) * 0.1,
            torch.randn(out_features, in_features) * 0.1
        ))
        
        self.hardware_config = HardwareConfig.from_profile(hardware_profile)
        self.quantizer = Quantizer(self.hardware_config)
        self.noise_model = NoiseModel(self.hardware_config, hardware_profile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = x.to(torch.complex64)
            
        # 量化处理 (复数量化)
        q_input = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.weight)
        
        # 计算
        out = ComplexOpticalLinearFunction.apply(q_input, q_weight, self.use_rust)
        
        # 噪声模拟
        if self.training:
            out = self.noise_model.apply_noise(out, training=True)
            
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, complex=True"
