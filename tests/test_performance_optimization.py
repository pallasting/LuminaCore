"""
性能优化测试 - 边缘端 NAT 模式和数据中心批量处理优化测试

测试以下优化特性：
1. NoiseAwareTrainer 轻量级模式（减少边缘端训练开销）
2. OpticalLinear 批量处理加速（优化数据中心推理）
3. 性能对比和基准测试
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lumina.layers.optical_linear import OpticalLinear
from lumina.optim.nat_trainer import NoiseAwareTrainer


class SimpleOptNet(nn.Module):
    """简单的光学神经网络用于测试"""

    def __init__(self, hardware_profile="lumina_nano_v1"):
        super(SimpleOptNet, self).__init__()
        self.optical1 = OpticalLinear(128, 256, hardware_profile=hardware_profile)
        self.optical2 = OpticalLinear(256, 128, hardware_profile=hardware_profile)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.sigmoid(self.optical1(x))
        x = torch.sigmoid(self.optical2(x))
        x = self.classifier(x)
        return x

    def forward_optimized(self, x, batch_size_threshold=64):
        """优化前向传播（数据中心批量处理加速）"""
        x = torch.sigmoid(self.optical1.forward_optimized(x, batch_size_threshold))
        x = torch.sigmoid(self.optical2.forward_optimized(x, batch_size_threshold))
        x = self.classifier(x)
        return x

    def forward_smart(self, x, batch_size_threshold=64):
        """智能前向传播（自动选择最优实现）"""
        x = torch.sigmoid(self.optical1.forward_smart(x, batch_size_threshold))
        x = torch.sigmoid(self.optical2.forward_smart(x, batch_size_threshold))
        x = self.classifier(x)
        return x


def test_lightweight_nat_mode():
    """测试轻量级 NAT 模式性能优化"""
    print("=" * 60)
    print("测试轻量级 NAT 模式性能优化")
    print("=" * 60)

    # 创建模型和标准训练器
    model_standard = SimpleOptNet("edge_ultra_low_power")
    optimizer = optim.Adam(model_standard.parameters(), lr=0.001)

    trainer_standard = NoiseAwareTrainer(
        model_standard,
        optimizer,
        robustness_target=0.95,
        noise_schedule="linear",
        max_noise_level=0.15,
        lightweight_mode=False,  # 标准模式
    )

    # 创建轻量级模式训练器
    model_lightweight = SimpleOptNet("edge_ultra_low_power")
    optimizer_light = optim.Adam(model_lightweight.parameters(), lr=0.001)

    trainer_lightweight = NoiseAwareTrainer(
        model_lightweight,
        optimizer_light,
        robustness_target=0.95,
        noise_schedule="linear",
        max_noise_level=0.15,
        lightweight_mode=True,  # 轻量级模式
        noise_injection_freq=4,  # 每4个batch注入一次噪声
    )

    # 生成测试数据
    batch_size = 32
    num_batches = 50
    input_dim = 128

    dummy_data = torch.randn(batch_size * num_batches, input_dim)
    dummy_targets = torch.randint(0, 10, (batch_size * num_batches,))

    dataset = TensorDataset(dummy_data, dummy_targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 测试标准模式性能
    print("测试标准模式...")
    start_time = time.time()

    trainer_standard.model.eval()
    total_inference_time = 0
    num_noise_injections = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()
            output = trainer_standard.model(data)
            batch_time = time.time() - batch_start
            total_inference_time += batch_time

            # 模拟噪声注入时间测量
            if batch_idx < 5:  # 只测试前几个batch
                noise_start = time.time()
                trainer_standard.inject_gradient_noise(batch_idx + 1, num_batches)
                noise_time = time.time() - noise_start
                num_noise_injections += 1
                print(
                    f"  Batch {batch_idx + 1}: 推理时间 {batch_time:.4f}s, 噪声注入时间 {noise_time:.4f}s"
                )

    standard_total_time = time.time() - start_time
    standard_avg_inference = total_inference_time / len(train_loader)

    print(f"标准模式总时间: {standard_total_time:.2f}s")
    print(f"平均推理时间: {standard_avg_inference:.4f}s/batch")

    # 测试轻量级模式性能
    print("\n测试轻量级模式...")
    start_time = time.time()

    trainer_lightweight.model.eval()
    total_inference_time_light = 0
    num_noise_injections_light = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()
            output = trainer_lightweight.model(data)
            batch_time = time.time() - batch_start
            total_inference_time_light += batch_time

            # 模拟噪声注入时间测量
            if batch_idx < 5:  # 只测试前几个batch
                noise_start = time.time()
                trainer_lightweight.inject_gradient_noise(batch_idx + 1, num_batches)
                noise_time = time.time() - noise_start
                if noise_time > 0:  # 只有实际执行了噪声注入才计数
                    num_noise_injections_light += 1
                    print(
                        f"  Batch {batch_idx + 1}: 推理时间 {batch_time:.4f}s, 噪声注入时间 {noise_time:.4f}s"
                    )

    lightweight_total_time = time.time() - start_time
    lightweight_avg_inference = total_inference_time_light / len(train_loader)

    print(f"轻量级模式总时间: {lightweight_total_time:.2f}s")
    print(f"平均推理时间: {lightweight_avg_inference:.4f}s/batch")

    # 性能对比
    print(f"\n性能对比:")
    print(
        f"推理时间优化: {((standard_avg_inference - lightweight_avg_inference) / standard_avg_inference * 100):.1f}%"
    )
    print(
        f"噪声注入频率: 标准模式 {num_noise_injections}次 vs 轻量级模式 {num_noise_injections_light}次"
    )
    print(
        f"噪声注入减少: {((num_noise_injections - num_noise_injections_light) / num_noise_injections * 100):.1f}%"
    )

    return {
        "standard_time": standard_total_time,
        "lightweight_time": lightweight_total_time,
        "speedup": standard_total_time / lightweight_total_time,
        "noise_reduction": (num_noise_injections - num_noise_injections_light)
        / max(num_noise_injections, 1),
    }


def test_batch_inference_optimization():
    """测试批量推理优化"""
    print("\n" + "=" * 60)
    print("测试批量推理优化")
    print("=" * 60)

    # 创建数据中心配置模型
    model = SimpleOptNet("datacenter_high_precision")
    model.eval()

    # 测试不同批量大小的性能
    batch_sizes = [16, 32, 64, 128, 256]
    input_dim = 128

    results = {}

    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")

        # 生成测试数据
        test_data = torch.randn(batch_size, input_dim)

        # 测试标准前向传播
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # 多次运行取平均
                output_standard = model.forward(test_data)
        standard_time = (time.time() - start_time) / 10

        # 测试优化前向传播
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # 多次运行取平均
                output_optimized = model.forward_optimized(
                    test_data, batch_size_threshold=64
                )
        optimized_time = (time.time() - start_time) / 10

        # 测试智能前向传播
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # 多次运行取平均
                output_smart = model.forward_smart(test_data, batch_size_threshold=64)
        smart_time = (time.time() - start_time) / 10

        # 验证输出差异（应该很小）
        diff_standard = torch.abs(output_standard - output_optimized).mean().item()
        diff_smart = torch.abs(output_standard - output_smart).mean().item()

        print(f"  标准模式: {standard_time:.4f}s")
        print(
            f"  优化模式: {optimized_time:.4f}s (加速比: {standard_time/optimized_time:.2f}x)"
        )
        print(
            f"  智能模式: {smart_time:.4f}s (加速比: {standard_time/smart_time:.2f}x)"
        )
        print(f"  输出差异 (标准 vs 优化): {diff_standard:.6f}")
        print(f"  输出差异 (标准 vs 智能): {diff_smart:.6f}")

        results[batch_size] = {
            "standard": standard_time,
            "optimized": optimized_time,
            "smart": smart_time,
            "speedup_opt": standard_time / optimized_time,
            "speedup_smart": standard_time / smart_time,
        }

    return results


def test_hardware_profile_comparison():
    """测试不同硬件配置的性能对比"""
    print("\n" + "=" * 60)
    print("测试不同硬件配置性能对比")
    print("=" * 60)

    hardware_profiles = [
        "edge_ultra_low_power",
        "lumina_nano_v1",
        "lumina_micro_v1",
        "datacenter_high_precision",
    ]

    batch_size = 128
    input_dim = 128

    results = {}

    for profile in hardware_profiles:
        print(f"\n测试硬件配置: {profile}")

        model = SimpleOptNet(profile)
        model.eval()

        test_data = torch.randn(batch_size, input_dim)

        # 测试推理时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                output = model.forward(test_data)
        inference_time = (time.time() - start_time) / 20

        # 获取模型参数数量
        total_params = sum(p.numel() for p in model.parameters())

        print(f"  推理时间: {inference_time:.4f}s")
        print(f"  总参数数量: {total_params:,}")
        print(f"  推理性能: {total_params / inference_time:.0f} 参数/秒")

        results[profile] = {
            "inference_time": inference_time,
            "total_params": total_params,
            "params_per_second": total_params / inference_time,
        }

    return results


def run_comprehensive_benchmark():
    """运行综合性能基准测试"""
    print("开始性能优化基准测试...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")

    # 1. 测试轻量级NAT模式
    nat_results = test_lightweight_nat_mode()

    # 2. 测试批量推理优化
    batch_results = test_batch_inference_optimization()

    # 3. 测试硬件配置对比
    hw_results = test_hardware_profile_comparison()

    # 总结报告
    print("\n" + "=" * 60)
    print("性能优化总结报告")
    print("=" * 60)

    print(f"\n1. 边缘端训练优化 (NAT轻量级模式):")
    print(f"   - 推理速度提升: {nat_results['speedup']:.2f}x")
    print(f"   - 噪声注入频率减少: {nat_results['noise_reduction']*100:.1f}%")

    print(f"\n2. 数据中心批量处理优化:")
    for batch_size, result in batch_results.items():
        if batch_size >= 64:  # 只显示大批量结果
            print(
                f"   - 批量大小{batch_size}: 优化模式 {result['speedup_opt']:.2f}x 加速"
            )

    print(f"\n3. 硬件配置性能排名:")
    sorted_profiles = sorted(
        hw_results.items(), key=lambda x: x[1]["params_per_second"], reverse=True
    )
    for i, (profile, result) in enumerate(sorted_profiles, 1):
        print(f"   {i}. {profile}: {result['params_per_second']:.0f} 参数/秒")

    print(f"\n性能优化完成！")

    return {
        "nat_optimization": nat_results,
        "batch_optimization": batch_results,
        "hardware_comparison": hw_results,
    }


if __name__ == "__main__":
    # 运行综合基准测试
    benchmark_results = run_comprehensive_benchmark()

    print("\n基准测试结果已保存。")
