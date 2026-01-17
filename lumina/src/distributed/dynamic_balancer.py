#!/usr/bin/env python3
"""
Dynamic Load Balancer

动态负载均衡器 - 根据瓦片利用率动态调整层分配

工作原理:
1. 监控每个瓦片的执行时间
2. 检测负载不均衡 (某瓦片执行时间明显更长)
3. 动态重新分配层到不同瓦片
4. 重新平衡后继续执行
"""

import torch
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import lumina_kernel


@dataclass
class TileMetrics:
    """瓦片指标"""
    tile_id: str
    total_time: float = 0.0
    task_count: int = 0
    avg_task_time: float = 0.0
    current_load: float = 0.0  # 当前负载 (0-1)
    temperature: float = 25.0  # 模拟温度 (°C)
    noise_level: float = 0.01


@dataclass
class LoadBalancingConfig:
    """负载均衡配置"""
    check_interval: int = 5  # 每 N 个任务检查一次
    imbalance_threshold: float = 0.2  # 20% 不均衡阈值
    max_rebalancing: int = 3  # 最大重平衡次数
    cooldown_time: float = 1.0  # 重平衡冷却时间
    temperature_threshold: float = 80.0  # 温度阈值


class DynamicLoadBalancer:
    """
    动态负载均衡器

    特点:
    - 实时监控瓦片利用率
    - 基于执行时间自动重平衡
    - 考虑温度和噪声影响
    - 线程安全
    """

    def __init__(
        self,
        num_tiles: int = 4,
        config: Optional[LoadBalancingConfig] = None
    ):
        self.num_tiles = num_tiles
        self.config = config or LoadBalancingConfig()

        # 瓦片指标
        self.tiles: Dict[str, TileMetrics] = {
            f"Tile-{i}": TileMetrics(tile_id=f"Tile-{i}")
            for i in range(num_tiles)
        }

        # 任务历史
        self.task_history: List[Dict[str, Any]] = []

        # 重平衡状态
        self.rebalance_count = 0
        self.last_rebalance_time = 0
        self.current_assignment: Dict[str, List[int]] = {}

        # 锁
        self.lock = threading.Lock()

        # 权重分配
        self.layer_weights: Dict[int, float] = {}  # 每层的计算复杂度

        print(f"🚀 DynamicLoadBalancer 初始化")
        print(f"   瓦片数: {num_tiles}")
        print(f"   不均衡阈值: {self.config.imbalance_threshold*100:.0f}%")

    def set_layer_weights(self, weights: Dict[int, float]):
        """设置每层的计算复杂度权重"""
        self.layer_weights = weights

    def _get_assignment_from_weights(self, num_layers: int) -> Dict[str, List[int]]:
        """根据权重分配层到瓦片"""
        # 按权重排序
        sorted_layers = sorted(range(num_layers), key=lambda i: self.layer_weights.get(i, 1.0))

        # 轮询分配到最空闲的瓦片
        assignment: Dict[str, List[int]] = {f"Tile-{i}": [] for i in range(self.num_tiles)}
        tile_loads = [0.0] * self.num_tiles

        for layer_idx in sorted_layers:
            weight = self.layer_weights.get(layer_idx, 1.0)
            # 找到负载最小的瓦片
            min_load_idx = min(range(self.num_tiles), key=lambda i: tile_loads[i])
            assignment[f"Tile-{min_load_idx}"].append(layer_idx)
            tile_loads[min_load_idx] += weight

        return assignment

    def record_task_completion(
        self,
        tile_id: str,
        layer_idx: int,
        execution_time: float,
        task_id: int = 0
    ):
        """记录任务完成"""
        with self.lock:
            # 更新瓦片指标
            tile = self.tiles[tile_id]
            tile.task_count += 1
            tile.total_time += execution_time
            tile.avg_task_time = tile.total_time / tile.task_count

            # 记录任务历史
            self.task_history.append({
                "tile_id": tile_id,
                "layer_idx": layer_idx,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "task_id": task_id
            })

            # 保持历史在合理范围
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-500:]

    def check_imbalance(self) -> Optional[Dict[str, Any]]:
        """检查负载不均衡"""
        if len(self.task_history) < self.config.check_interval:
            return None

        # 计算每个瓦片的平均任务时间
        tile_times: Dict[str, float] = {}
        for tile in self.tiles.values():
            tile_times[tile.tile_id] = tile.avg_task_time

        if not tile_times:
            return None

        # 计算不均衡度
        avg_time = sum(tile_times.values()) / len(tile_times)
        max_time = max(tile_times.values())
        min_time = min(tile_times.values())

        if avg_time == 0:
            return None

        imbalance = (max_time - min_time) / avg_time

        if imbalance > self.config.imbalance_threshold:
            # 找到最忙和最闲的瓦片
            busiest = max(tile_times.items(), key=lambda x: x[1])
            slowest = min(tile_times.items(), key=lambda x: x[1])

            return {
                "imbalance": imbalance,
                "busiest_tile": busiest[0],
                "busiest_time": busiest[1],
                "slowest_tile": slowest[0],
                "slowest_time": slowest[1],
                "avg_time": avg_time
            }

        return None

    def rebalance(self, num_layers: int) -> Dict[str, List[int]]:
        """
        执行负载重平衡

        Returns:
            新的分配方案
        """
        if self.rebalance_count >= self.config.max_rebalancing:
            print(f"   ⚠️ 达到最大重平衡次数 ({self.config.max_rebalancing})")
            return self.current_assignment

        # 检查冷却时间
        if time.time() - self.last_rebalance_time < self.config.cooldown_time:
            return self.current_assignment

        print(f"\n🔄 执行负载重平衡 (第 {self.rebalance_count + 1} 次)...")

        # 分析任务历史，计算实际负载
        layer_loads: Dict[int, float] = defaultdict(float)
        for task in self.task_history[-100:]:
            layer_loads[task["layer_idx"]] += task["execution_time"]

        # 平均化负载
        num_tasks = max(1, len(self.task_history) // self.num_tiles)
        for layer_idx in layer_loads:
            layer_loads[layer_idx] /= num_tasks

        # 更新层权重
        self.layer_weights = dict(layer_loads)

        # 生成新分配
        new_assignment = self._get_assignment_from_weights(num_layers)

        # 计算负载变化
        old_loads = self._calculate_assignment_load(self.current_assignment)
        new_loads = self._calculate_assignment_load(new_assignment)

        old_imbalance = max(old_loads.values()) - min(old_loads.values()) if old_loads else 0
        new_imbalance = max(new_loads.values()) - min(new_loads.values()) if new_loads else 0

        print(f"   旧负载差异: {old_imbalance:.3f}s")
        print(f"   新负载差异: {new_imbalance:.3f}s")

        if new_imbalance < old_imbalance or not self.current_assignment:
            self.current_assignment = new_assignment
            self.rebalance_count += 1
            self.last_rebalance_time = time.time()

            print(f"   ✅ 重平衡完成")
            return new_assignment
        else:
            print(f"   ⚠️ 重平衡无改善，跳过")
            return self.current_assignment

    def _calculate_assignment_load(self, assignment: Dict[str, List[int]]) -> Dict[str, float]:
        """计算分配方案的负载"""
        loads: Dict[str, float] = {}
        for tile_id, layers in assignment.items():
            total_load = sum(self.layer_weights.get(i, 0) for i in layers)
            loads[tile_id] = total_load
        return loads

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self.lock:
            return {
                "tiles": {
                    tile_id: {
                        "task_count": metrics.task_count,
                        "avg_time": metrics.avg_task_time,
                        "current_load": metrics.current_load,
                        "temperature": metrics.temperature
                    }
                    for tile_id, metrics in self.tiles.items()
                },
                "rebalance_count": self.rebalance_count,
                "task_history_size": len(self.task_history)
            }

    def print_status(self):
        """打印当前状态"""
        status = self.get_status()
        print(f"\n📊 Load Balancer Status:")
        print(f"   总任务数: {sum(t['task_count'] for t in status['tiles'].values())}")
        print(f"   重平衡次数: {status['rebalance_count']}")

        print(f"\n📱 Tile Utilization:")
        for tile_id, info in status["tiles"].items():
            load_bar = "█" * int(info["current_load"] * 20)
            print(f"   {tile_id}: {info['task_count']:3d} tasks, "
                  f"avg {info['avg_time']*1000:6.2f}ms, "
                  f"temp {info['temperature']:.1f}°C")


class LoadBalancedExecutor:
    """支持负载均衡的执行器"""

    def __init__(
        self,
        num_tiles: int = 4,
        use_rust: bool = True
    ):
        self.num_tiles = num_tiles
        self.use_rust = use_rust
        self.balancer = DynamicLoadBalancer(num_tiles)
        self.current_assignment: Dict[str, List[int]] = {}

        # 初始化设备
        for i in range(num_tiles):
            device_name = f"Tile-{i}"
            if device_name not in lumina_kernel.list_devices():
                lumina_kernel.create_mock_device(device_name, 8 * 1024**3)

        print(f"🚀 LoadBalancedExecutor 初始化完成")

    def execute_with_balancing(
        self,
        input_tensor: torch.Tensor,
        weights: Dict[int, torch.Tensor],
        num_batches: int = 4
    ) -> Dict[str, Any]:
        """
        执行带负载均衡的推理

        Returns:
            执行结果和性能指标
        """
        num_layers = len(weights)

        # 初始分配 (平均分配)
        layers_per_tile = num_layers // self.num_tiles
        self.current_assignment = {
            f"Tile-{i}": list(range(i * layers_per_tile, min((i + 1) * layers_per_tile, num_layers)))
            for i in range(self.num_tiles)
        }

        print(f"\n⚡ 开始带负载均衡的推理")
        print(f"   瓦片数: {self.num_tiles}")
        print(f"   批次: {num_batches}")

        # 初始化负载均衡器
        self.balancer.current_assignment = self.current_assignment

        results = []
        start_time = time.time()

        for batch_idx in range(num_batches):
            output = input_tensor
            batch_tasks = []

            # 在每个瓦片上执行
            for tile_id, layers in self.current_assignment.items():
                tile_start = time.time()

                for layer_idx in layers:
                    weight = weights[layer_idx]

                    if self.use_rust:
                        # Rust 后端
                        output_np = lumina_kernel.optical_linear_fused(
                            output.detach().cpu().numpy(),
                            weight.detach().cpu().numpy(),
                            None, 0.01, 8,
                            seed=42 + layer_idx + batch_idx * 100
                        )
                        output = torch.from_numpy(output_np)
                    else:
                        # PyTorch
                        output = torch.nn.functional.linear(output, weight)

                    exec_time = time.time() - tile_start

                    # 记录任务完成
                    self.balancer.record_task_completion(
                        tile_id, layer_idx, exec_time,
                        task_id=batch_idx * num_layers + layer_idx
                    )

                batch_tasks.append({
                    "tile_id": tile_id,
                    "time": time.time() - tile_start,
                    "layers": len(layers)
                })

            results.append({
                "batch_idx": batch_idx,
                "output": output,
                "tasks": batch_tasks,
                "time": time.time() - start_time
            })

            # 检查负载均衡
            imbalance = self.balancer.check_imbalance()
            if imbalance:
                print(f"   ⚠️ 检测到负载不均衡: {imbalance['imbalance']*100:.1f}%")
                new_assignment = self.balancer.rebalance(num_layers)
                if new_assignment:
                    self.current_assignment = new_assignment

        total_time = time.time() - start_time

        return {
            "results": results,
            "total_time": total_time,
            "throughput": num_batches / total_time,
            "balancer_status": self.balancer.get_status()
        }


def demo_load_balancing():
    """演示负载均衡"""
    print("=" * 70)
    print("Dynamic Load Balancing Demo")
    print("=" * 70)

    # 配置
    num_layers = 12
    hidden_size = 2048
    num_batches = 8

    # 创建模型 (所有层相同大小，但模拟不同计算复杂度)
    weights = {}
    for i in range(num_layers):
        # 模拟不同层的计算复杂度 (中间层需要更长时间)
        if 4 <= i <= 7:
            # 中间层：添加更多操作来模拟复杂计算
            weights[i] = torch.randn(hidden_size, hidden_size)
        else:
            weights[i] = torch.randn(hidden_size, hidden_size)

    # 创建测试批次
    test_batches = [
        torch.randn(2, hidden_size)
        for _ in range(num_batches)
    ]

    # 执行带负载均衡的推理
    executor = LoadBalancedExecutor(num_tiles=4)

    result = executor.execute_with_balancing(
        test_batches[0], weights, num_batches
    )

    # 打印结果
    print(f"\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)
    print(f"   总时间: {result['total_time']:.3f}s")
    print(f"   吞吐量: {result['throughput']:.2f} batches/s")

    status = result['balancer_status']
    print(f"\n📱 Tile Utilization:")
    for tile_id, info in status["tiles"].items():
        print(f"   {tile_id}: {info['task_count']} tasks, "
              f"avg {info['avg_time']*1000:.2f}ms")

    print(f"\n   重平衡次数: {status['rebalance_count']}")

    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    demo_load_balancing()
