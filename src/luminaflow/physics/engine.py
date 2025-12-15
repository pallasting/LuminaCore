import torch
import torch.nn as nn

class HardwareProfile:
    """定义芯片的物理规格"""
    def __init__(self, name="Lumina_Nano_v1", noise_std=0.15, precision_bits=4):
        self.name = name
        self.noise_std = noise_std       # 光路噪声标准差 (e.g., 0.15 = 15%)
        self.precision_bits = precision_bits # DAC/ADC 精度

    @staticmethod
    def get_default():
        return HardwareProfile()

class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) 量化函数
    前向传播：执行离散化 (模拟 DAC/ADC 台阶)
    反向传播：直接传递梯度 (欺骗 PyTorch，使其认为该过程可导)
    """
    @staticmethod
    def forward(ctx, input, bits):
        scale = 2 ** bits - 1
        # 1. 归一化并钳位
        x_min, x_max = input.min(), input.max()
        input_norm = (input - x_min) / (x_max - x_min + 1e-8)
        
        # 2. 量化
        output = torch.round(input_norm * scale) / scale
        
        # 3. 反归一化
        output = output * (x_max - x_min + 1e-8) + x_min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 直通梯度：不计算量化的导数
        return grad_output, None

class PhysicsEngine:
    @staticmethod
    def simulate_noise(signal, noise_std, device):
        """
        模拟散粒噪声与热噪声：噪声强度与信号强度相关 (Signal-Dependent)
        """
        if noise_std <= 0:
            return signal
        
        # 生成高斯噪声，幅度基于信号的标准差
        noise = torch.randn_like(signal, device=device) * noise_std * signal.std().detach()
        return signal + noise

    @staticmethod
    def simulate_quantization(signal, bits):
        """模拟 DAC/ADC 转换"""
        if bits >= 32: # FP32 模式不量化
            return signal
        return STEQuantize.apply(signal, bits)