#!/usr/bin/env python
"""
LuminaFlow SDK 基本功能测试脚本

用于验证包的基本功能是否正常工作
"""

import torch
import sys

print("=" * 60)
print("LuminaFlow SDK v0.1 - 基本功能测试")
print("=" * 60)

# 测试 1: 导入
print("\n[测试 1] 导入模块...")
try:
    import lumina
    import lumina.nn as lnn
    from lumina.optim import NoiseAwareTrainer
    from lumina.viz import benchmark_robustness
    print("✅ 所有模块导入成功")
    print(f"   版本: {lumina.__version__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试 2: OpticalLinear 层
print("\n[测试 2] 测试 OpticalLinear 层...")
try:
    layer = lnn.OpticalLinear(
        in_features=784,
        out_features=10,
        hardware_profile='lumina_nano_v1',
        precision=4
    )
    print("✅ OpticalLinear 创建成功")
    print(f"   配置: {layer.hardware_profile}, {layer.precision}-bit, 噪声: {layer.noise_level:.0%}")
    
    # 测试前向传播
    x = torch.randn(32, 784)
    y = layer(x)
    print(f"✅ 前向传播成功: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"❌ OpticalLinear 测试失败: {e}")
    sys.exit(1)

# 测试 3: WDMChannelMapper
print("\n[测试 3] 测试 WDMChannelMapper...")
try:
    from lumina.layers import WDMChannelMapper
    wdm = WDMChannelMapper(num_channels=3, channel_strategy='rgb')
    x = torch.randn(32, 128)
    y = wdm(x, mode='map')
    print(f"✅ WDMChannelMapper 测试成功: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"❌ WDMChannelMapper 测试失败: {e}")
    sys.exit(1)

# 测试 4: NoiseAwareTrainer
print("\n[测试 4] 测试 NoiseAwareTrainer...")
try:
    model = torch.nn.Sequential(
        lnn.OpticalLinear(784, 128, hardware_profile='lumina_nano_v1'),
        torch.nn.ReLU(),
        lnn.OpticalLinear(128, 10, hardware_profile='lumina_nano_v1'),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = NoiseAwareTrainer(
        model=model,
        optimizer=optimizer,
        robustness_target=0.98
    )
    print("✅ NoiseAwareTrainer 创建成功")
except Exception as e:
    print(f"❌ NoiseAwareTrainer 测试失败: {e}")
    sys.exit(1)

# 测试 5: 可视化函数
print("\n[测试 5] 测试可视化函数...")
try:
    from lumina.viz import plot_robustness_curve
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    accuracies = [98.5, 96.1, 91.5, 85.3, 75.2]
    plot_robustness_curve(noise_levels, accuracies, save_path="test_robustness.png")
    print("✅ 可视化函数测试成功（已生成 test_robustness.png）")
except Exception as e:
    print(f"❌ 可视化函数测试失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！LuminaFlow SDK 基本功能正常。")
print("=" * 60)

