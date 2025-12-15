#!/usr/bin/env python3
"""
演示重构后的OpticalLinear架构
"""

import torch
import torch.nn as nn
from lumina.layers.optical_linear import OpticalLinear
from lumina.layers.optical_components import HardwareConfig, Quantizer, NoiseModel

def demo_separation_of_concerns():
    """演示关注点分离"""
    print("=== 重构演示：关注点分离 ===")
    
    # 1. 硬件配置管理
    print("\n1. 硬件配置管理")
    config = HardwareConfig.from_profile("lumina_nano_v1")
    print(f"配置：{config}")
    
    # 2. 量化逻辑
    print("\n2. 量化逻辑")
    quantizer = Quantizer(config)
    x = torch.tensor([0.1, 0.5, 0.9, 1.2])
    x_quantized = quantizer.quantize(x)
    print(f"原始：{x}")
    print(f"量化：{x_quantized}")
    
    # 3. 噪声模型
    print("\n3. 噪声模型")
    noise_model = NoiseModel(config, "lumina_nano_v1")
    signal = torch.randn(2, 4) * 0.5  # 较大的信号以产生可见噪声
    noisy_signal = noise_model.apply_noise(signal, training=True)
    print(f"原始信号范数：{torch.norm(signal):.4f}")
    print(f"噪声信号范数：{torch.norm(noisy_signal):.4f}")
    
def demo_backward_compatibility():
    """演示向后兼容性"""
    print("\n=== 向后兼容性演示 ===")
    
    # 使用旧API创建OpticalLinear
    layer = OpticalLinear(4, 2, hardware_profile="lumina_micro_v1")
    
    # 测试旧方法
    x = torch.randn(1, 4)
    
    # 这些方法应该仍然工作
    y1 = layer(x)  # forward
    y2 = layer.dac_convert(x)  # dac转换
    y3 = layer.adc_convert(x)  # adc转换
    y4 = layer.optical_matrix_multiply(x, layer.weight)  # 光学矩阵乘法
    
    print(f"输入形状：{x.shape}")
    print(f"forward输出形状：{y1.shape}")
    print(f"dac_convert输出形状：{y2.shape}")
    print(f"adc_convert输出形状：{y3.shape}")
    print(f"optical_matrix_multiply输出形状：{y4.shape}")
    
    # 测试新API
    config = layer.get_hardware_config()
    quantizer = layer.get_quantizer()
    noise_model = layer.get_noise_model()
    
    print(f"\n新API获取的组件：")
    print(f"硬件配置：{type(config).__name__}")
    print(f"量化器：{type(quantizer).__name__}")
    print(f"噪声模型：{type(noise_model).__name__}")

def demo_architecture_improvements():
    """演示架构改进"""
    print("\n=== 架构改进演示 ===")
    
    # 创建不同的硬件配置
    configs = [
        HardwareConfig.from_profile("lumina_nano_v1"),
        HardwareConfig.from_profile("datacenter_high_precision"),
        HardwareConfig.from_profile("edge_ultra_low_power")
    ]
    
    print("不同硬件配置的量化器：")
    for i, config in enumerate(configs):
        quantizer = Quantizer(config)
        test_input = torch.tensor([0.5])
        quantized = quantizer.quantize(test_input)
        print(f"配置{i+1}: {quantized.item():.4f} (量化步长: {config.quantization_step:.4f})")
    
    print("\n不同硬件配置的噪声模型：")
    for i, config in enumerate(configs):
        noise_model = NoiseModel(config, config.__dict__)
        signal = torch.ones(1, 4) * 0.8  # 较大信号
        noisy = noise_model.apply_noise(signal, training=True)
        noise_level = torch.norm(noisy - signal).item()
        print(f"配置{i+1}: 噪声水平 {noise_level:.4f}")

if __name__ == "__main__":
    print("重构后的OpticalLinear架构演示")
    print("=" * 50)
    
    demo_separation_of_concerns()
    demo_backward_compatibility()
    demo_architecture_improvements()
    
    print("\n" + "=" * 50)
    print("重构完成！所有功能正常工作。")