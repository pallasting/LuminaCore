#!/usr/bin/env python3
"""
测试重构后的OpticalLinear组件
"""

import torch
import torch.nn as nn
from lumina.layers.optical_linear import OpticalLinear
from lumina.layers.optical_components import HardwareConfig, Quantizer, NoiseModel

def test_new_components():
    """测试新组件的功能"""
    print("测试HardwareConfig...")
    config = HardwareConfig.from_profile("lumina_nano_v1")
    print(f"HardwareConfig: {config}")
    
    print("\n测试Quantizer...")
    quantizer = Quantizer(config)
    x = torch.tensor([0.1, 0.5, 0.9, 1.2])
    x_quantized = quantizer.quantize(x)
    print(f"原始: {x}")
    print(f"量化: {x_quantized}")
    
    print("\n测试NoiseModel...")
    noise_model = NoiseModel(config, "lumina_nano_v1")
    signal = torch.ones(2, 4)
    noisy_signal = noise_model.apply_noise(signal, training=True)
    print(f"原始信号: {signal}")
    print(f"噪声信号: {noisy_signal}")
    
def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n测试向后兼容性...")
    
    # 创建OpticalLinear实例
    layer = OpticalLinear(4, 2, hardware_profile="lumina_nano_v1")
    
    # 测试基本功能
    x = torch.randn(1, 4)
    y = layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 测试旧方法是否存在
    assert hasattr(layer, 'quantize')
    assert hasattr(layer, 'dac_convert')
    assert hasattr(layer, 'adc_convert')
    assert hasattr(layer, 'optical_matrix_multiply')
    assert hasattr(layer, 'forward')
    assert hasattr(layer, 'forward_optimized')
    assert hasattr(layer, 'forward_smart')
    
    # 测试新方法
    assert hasattr(layer, 'get_hardware_config')
    assert hasattr(layer, 'get_quantizer')
    assert hasattr(layer, 'get_noise_model')
    
    print("所有方法都存在！")
    
    # 测试获取组件
    config = layer.get_hardware_config()
    quantizer = layer.get_quantizer()
    noise_model = layer.get_noise_model()
    
    print(f"硬件配置: {config}")
    print(f"量化器类型: {type(quantizer)}")
    print(f"噪声模型类型: {type(noise_model)}")

def test_noise_randomness():
    """测试噪声随机性"""
    print("\n测试噪声随机性...")
    
    layer = OpticalLinear(4, 4, hardware_profile="edge_ultra_low_power")
    layer.train()
    
    # 使用相同输入测试多次
    x = torch.ones(1, 4)
    
    outputs = []
    for i in range(5):
        y = layer(x)
        outputs.append(y.clone())
        print(f"输出 {i+1}: {y}")
    
    # 检查输出是否不完全相同（表明有随机性）
    all_same = True
    for i in range(1, len(outputs)):
        if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
            all_same = False
            break
    
    if not all_same:
        print("✓ 噪声注入有随机性")
    else:
        print("✗ 噪声注入可能缺乏随机性")

if __name__ == "__main__":
    print("=== 测试重构后的OpticalLinear组件 ===")
    test_new_components()
    test_backward_compatibility()
    test_noise_randomness()
    print("\n=== 测试完成 ===")