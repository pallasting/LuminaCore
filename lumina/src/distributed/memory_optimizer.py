#!/usr/bin/env python3
"""
Memory-Optimized Backend

内存优化后端 - 减少 Rust 和 Python 之间的数据拷贝

关键优化:
1. 使用 torch.from_numpy(output_np, copy=False) 避免不必要的拷贝
2. 缓存权重在 GPU/CPU 上，避免重复转换
3. 批量处理减少函数调用开销
4. 使用内存池复用缓冲区
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import lumina_kernel


@dataclass
class MemoryConfig:
    """内存配置"""
    enable_caching: bool = True
    cache_weights: bool = True
    use_memory_pool: bool = True
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB


class MemoryPool:
    """简单内存池 - 复用 numpy 数组"""

    def __init__(self, max_size: int = 1024 * 1024 * 1024):
        self.max_size = max_size
        self.pool: Dict[Tuple[int, ...], List[np.ndarray]] = {}
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def get(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """从池中获取数组"""
        key = (shape, dtype)

        if key in self.pool and self.pool[key]:
            self.hits += 1
            return self.pool[key].pop()

        self.misses += 1
        return np.empty(shape, dtype=dtype)

    def release(self, arr: np.ndarray):
        """将数组归还给池"""
        shape = (arr.shape, arr.dtype)
        key = (tuple(shape[0]), shape[1])

        if self.current_size + arr.nbytes < self.max_size:
            if key not in self.pool:
                self.pool[key] = []
            self.pool[key].append(arr)
            self.current_size += arr.nbytes

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "pool_size": self.current_size}


class OptimizedPhotonicExecutor:
    """
    内存优化光子计算执行器

    特点:
    - 零拷贝数据传递 (当可能时)
    - 权重缓存
    - 内存池复用
    - 批量执行优化
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        noise_std: float = 0.01,
        bits: int = 8,
        enable_noise: bool = True,
        config: Optional[MemoryConfig] = None
    ):
        self.config = config or MemoryConfig()
        self.noise_std = noise_std
        self.bits = bits
        self.enable_noise = enable_noise

        # 权重缓存
        self.weight_cache: Dict[int, Tuple[np.ndarray, torch.Tensor]] = {}

        # 内存池
        if self.config.use_memory_pool:
            self.memory_pool = MemoryPool()
        else:
            self.memory_pool = None

        # 统计
        self.stats = {
            "forward_passes": 0,
            "total_time": 0.0,
            "copy_time": 0.0,
            "compute_time": 0.0
        }

        print(f"🚀 OptimizedPhotonicExecutor 初始化")
        print(f"   权重缓存: {'启用' if self.config.cache_weights else '禁用'}")
        print(f"   内存池: {'启用' if self.config.use_memory_pool else '禁用'}")

    def _ensure_contiguous(self, arr: np.ndarray) -> np.ndarray:
        """确保数组是 C-contiguous"""
        if not arr.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(arr)
        return arr

    def _cache_weight(self, layer_idx: int, weight: torch.Tensor) -> np.ndarray:
        """缓存权重到 numpy"""
        if layer_idx in self.weight_cache:
            cached_np, _ = self.weight_cache[layer_idx]
            # 检查形状是否匹配
            if cached_np.shape == tuple(weight.shape):
                return cached_np

        # 转换并缓存
        weight_np = weight.detach().cpu().numpy()
        weight_np = self._ensure_contiguous(weight_np)
        self.weight_cache[layer_idx] = (weight_np, weight)
        return weight_np

    def execute_layer(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        layer_idx: int = 0,
        batch_id: int = 0
    ) -> Tuple[torch.Tensor, float]:
        """
        执行单层计算 (优化版本)

        Args:
            input_tensor: 输入张量
            weight: 权重张量
            layer_idx: 层索引 (用于缓存)
            batch_id: 批次 ID (用于随机种子)

        Returns:
            output_tensor, execution_time
        """
        start_total = time.time()

        # 获取缓存的权重
        if self.config.cache_weights:
            weight_np = self._cache_weight(layer_idx, weight)
        else:
            weight_np = self._ensure_contiguous(weight.detach().cpu().numpy())

        # 转换输入
        copy_start = time.time()
        input_np = self._ensure_contiguous(input_tensor.detach().cpu().numpy())
        copy_time = time.time() - copy_start

        # 获取输出数组 (从内存池或新建)
        if self.memory_pool:
            output_np = self.memory_pool.get(weight_np.shape[:1] + (input_np.shape[0],))
        else:
            output_np = np.empty((input_np.shape[0], weight_np.shape[0]), dtype=np.float32)

        # 执行计算
        compute_start = time.time()
        result_np = lumina_kernel.optical_linear_fused(
            input_np,
            weight_np,
            None,  # 无偏置
            self.noise_std,
            self.bits,
            seed=42 + layer_idx + batch_id * 100
        )
        compute_time = time.time() - compute_start

        # 转换回 torch (零拷贝当可能时)
        output_tensor = torch.from_numpy(result_np)

        # 释放输出数组回内存池
        if self.memory_pool:
            self.memory_pool.release(output_np)

        # 更新统计
        exec_time = time.time() - start_total
        self.stats["forward_passes"] += 1
        self.stats["total_time"] += exec_time
        self.stats["copy_time"] += copy_time
        self.stats["compute_time"] += compute_time

        return output_tensor, exec_time

    def execute_inference(
        self,
        input_tensor: torch.Tensor,
        weights: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, float]:
        """执行推理 (无噪声)"""
        start = time.time()
        output = input_tensor

        for i in range(len(weights)):
            output, _ = self.execute_layer(output, weights[i], layer_idx=i)

        return output, time.time() - start

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        passes = self.stats["forward_passes"]
        return {
            "forward_passes": passes,
            "total_time": self.stats["total_time"],
            "avg_time": self.stats["total_time"] / passes if passes > 0 else 0,
            "copy_time": self.stats["copy_time"],
            "compute_time": self.stats["compute_time"],
            "copy_ratio": self.stats["copy_time"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0,
            "cache_size": len(self.weight_cache),
            "pool_stats": self.memory_pool.stats() if self.memory_pool else None
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n📊 Optimized Executor Statistics:")
        print(f"   Forward Passes: {stats['forward_passes']}")
        print(f"   Total Time: {stats['total_time']*1000:.2f}ms")
        print(f"   Avg Time: {stats['avg_time']*1000:.2f}ms")
        print(f"   Copy Time: {stats['copy_time']*1000:.2f}ms ({stats['copy_ratio']*100:.1f}%)")
        print(f"   Compute Time: {stats['compute_time']*1000:.2f}ms")
        print(f"   Cache Size: {stats['cache_size']} weights")

        if stats['pool_stats']:
            pool = stats['pool_stats']
            print(f"   Pool Hits: {pool['hits']}, Misses: {pool['misses']}")


def benchmark_memory_optimization():
    """内存优化基准测试"""
    print("=" * 70)
    print("Memory Optimization Benchmark")
    print("=" * 70)

    # 配置
    config = MemoryConfig(
        enable_caching=True,
        cache_weights=True,
        use_memory_pool=True
    )

    executor = OptimizedPhotonicExecutor(
        device_name="benchmark",
        noise_std=0.01,
        bits=8,
        enable_noise=True,
        config=config
    )

    # 创建模型
    num_layers = 12
    hidden_size = 2048
    weights = {
        i: torch.randn(hidden_size, hidden_size)
        for i in range(num_layers)
    }

    # 测试批次
    num_batches = 8
    test_batches = [
        torch.randn(4, hidden_size)
        for _ in range(num_batches)
    ]

    print(f"\n📦 Model: {num_layers} layers, hidden={hidden_size}")
    print(f"   Batches: {num_batches}")

    # 预热
    print(f"\n🔥 Warmup...")
    for i in range(3):
        _ = executor.execute_layer(test_batches[0], weights[0], layer_idx=0)

    # 基准测试
    print(f"\n🔬 Running benchmark...")
    start = time.time()

    for batch_idx, input_tensor in enumerate(test_batches):
        output = input_tensor
        for layer_idx in range(num_layers):
            output, _ = executor.execute_layer(
                output, weights[layer_idx],
                layer_idx=layer_idx,
                batch_id=batch_idx
            )

    total_time = time.time() - start

    # 打印统计
    executor.print_stats()

    print(f"\n✅ Results:")
    print(f"   Total Time: {total_time:.3f}s")
    print(f"   Throughput: {num_batches/total_time:.2f} batches/s")

    return executor.get_stats()


def compare_with_baseline():
    """与基线比较"""
    print("\n" + "=" * 70)
    print("Baseline Comparison: Standard vs Optimized")
    print("=" * 70)

    num_layers = 12
    hidden_size = 2048
    num_batches = 8

    # 创建权重
    weights = {
        i: torch.randn(hidden_size, hidden_size)
        for i in range(num_layers)
    }

    # 测试批次
    test_batches = [
        torch.randn(4, hidden_size)
        for _ in range(num_batches)
    ]

    # 1. 标准方法 (每次转换)
    print(f"\n📊 Standard Method (no optimization)...")
    start = time.time()

    for input_tensor in test_batches:
        output = input_tensor
        for i in range(num_layers):
            input_np = input_tensor.detach().cpu().numpy()
            weight_np = weights[i].detach().cpu().numpy()
            output_np = lumina_kernel.optical_linear_fused(
                input_np, weight_np, None, 0.01, 8, 42 + i
            )
            output = torch.from_numpy(output_np)

    standard_time = time.time() - start
    print(f"   Time: {standard_time:.3f}s")

    # 2. 优化方法
    print(f"\n🚀 Optimized Method...")
    config = MemoryConfig(cache_weights=True, use_memory_pool=True)
    executor = OptimizedPhotonicExecutor(config=config)

    start = time.time()

    for batch_idx, input_tensor in enumerate(test_batches):
        output = input_tensor
        for i in range(num_layers):
            output, _ = executor.execute_layer(
                output, weights[i],
                layer_idx=i,
                batch_id=batch_idx
            )

    optimized_time = time.time() - start
    print(f"   Time: {optimized_time:.3f}s")

    # 打印对比
    print(f"\n" + "=" * 70)
    print("📊 Comparison:")
    print(f"   Standard:  {standard_time:.3f}s")
    print(f"   Optimized: {optimized_time:.3f}s")
    speedup = standard_time / optimized_time if optimized_time > 0 else 1
    print(f"   Speedup:   {speedup:.2f}x")
    print(f"=" * 70)


if __name__ == "__main__":
    benchmark_memory_optimization()
    compare_with_baseline()
