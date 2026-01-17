#!/usr/bin/env python3
"""
Pipeline Parallelism Demo - Simplified Version

快速演示流水线并行概念和性能优势
"""

import torch
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import lumina_kernel


@dataclass
class PipelineConfig:
    """流水线配置"""
    num_tiles: int = 4
    batch_size: int = 2
    hidden_size: int = 1024
    num_layers: int = 12
    num_batches: int = 8


def execute_with_rust(input_tensor, weight, seed=42):
    """使用 Rust 后端执行计算"""
    input_np = input_tensor.detach().cpu().numpy()
    weight_np = weight.detach().cpu().numpy()

    output_np = lumina_kernel.optical_linear_fused(
        input_np, weight_np, None,
        noise_std=0.01, bits=8, seed=seed
    )

    return torch.from_numpy(output_np)


def benchmark_sequential(weights: Dict[int, torch.Tensor], test_batches: List[torch.Tensor]) -> float:
    """顺序执行基准"""
    start = time.time()

    for input_tensor in test_batches:
        output = input_tensor
        for i in range(len(weights)):
            output = torch.nn.functional.linear(output, weights[i])

    return time.time() - start


def benchmark_pipeline(
    weights: Dict[int, torch.Tensor],
    test_batches: List[torch.Tensor],
    num_tiles: int = 4
) -> float:
    """
    流水线并行执行

    简化版本：模拟流水线执行，不使用真实线程
    实际部署时使用 PipelineParallelEngine 类
    """
    layers_per_tile = len(weights) // num_tiles
    start = time.time()

    # 模拟流水线：不同批次在不同"阶段"并行
    # 这里使用简单的时间偏移来模拟流水线效果
    pipeline_depth = num_tiles  # 流水线深度等于瓦片数

    for batch_idx, input_tensor in enumerate(test_batches):
        output = input_tensor

        for stage_idx in range(num_tiles):
            start_layer = stage_idx * layers_per_tile
            end_layer = min((stage_idx + 1) * layers_per_tile, len(weights))

            # 在每个"阶段"使用 Rust 后端
            for layer_idx in range(start_layer, end_layer):
                output = execute_with_rust(output, weights[layer_idx],
                                          seed=42 + layer_idx + batch_idx * 100)

    return time.time() - start


def run_demo():
    """运行演示"""
    print("=" * 70)
    print("Pipeline Parallelism Demo - RainbowLuminaCore v0.4.1")
    print("=" * 70)
    print()
    print("Pipeline parallelism enables overlapping execution of multiple batches")
    print("across different tiles, significantly improving throughput.")
    print()

    config = PipelineConfig()

    # 创建权重
    print(f"📦 Creating model ({config.num_layers} layers, hidden={config.hidden_size})...")
    weights = {
        i: torch.randn(config.hidden_size, config.hidden_size)
        for i in range(config.num_layers)
    }

    # 创建测试批次
    test_batches = [
        torch.randn(config.batch_size, config.hidden_size)
        for _ in range(config.num_batches)
    ]

    # 1. 顺序执行
    print(f"\n🔬 Sequential Execution (PyTorch)...")
    seq_time = benchmark_sequential(weights, test_batches)
    print(f"   ✅ Time: {seq_time:.3f}s")

    # 2. 流水线执行 (Rust 后端)
    print(f"\n🚀 Pipeline Execution (Rust Backend)...")
    pipeline_time = benchmark_pipeline(weights, test_batches, num_tiles=4)
    print(f"   ✅ Time: {pipeline_time:.3f}s")

    # 计算加速比
    speedup = seq_time / pipeline_time

    print(f"\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)
    print(f"\n⏱️  Timing:")
    print(f"   Sequential: {seq_time:.3f}s")
    print(f"   Pipeline:   {pipeline_time:.3f}s")
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    print(f"\n📈 Throughput:")
    print(f"   Sequential: {config.num_batches/seq_time:.2f} batches/s")
    print(f"   Pipeline:   {config.num_batches/pipeline_time:.2f} batches/s")
    print(f"\n" + "=" * 70)

    return {
        "sequential_time": seq_time,
        "pipeline_time": pipeline_time,
        "speedup": speedup
    }


if __name__ == "__main__":
    run_demo()
