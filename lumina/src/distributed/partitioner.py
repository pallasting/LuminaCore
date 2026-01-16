"""
分布式光子计算模型分割器

将 Llama 模型智能分割到多个光子计算瓦片上
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class PartitionStrategy(Enum):
    """模型分割策略"""
    BY_LAYERS = "by_layers"           # 按 Transformer 层均匀分割
    BY_COMPUTE = "by_compute"         # 按计算复杂度分割
    BY_MEMORY = "by_memory"           # 按内存需求分割
    HYBRID = "hybrid"                  # 混合策略


@dataclass
class LayerProfile:
    """层性能配置文件"""
    layer_idx: int
    layer_type: str
    compute_units: float  # 相对计算单元数
    memory_mb: float      # 内存需求 (MB)
    photonic_efficiency: float  # 光子计算效率 (0-1)
    dependencies: List[int]  # 依赖的层索引


@dataclass 
class TileAssignment:
    """瓦片分配配置"""
    tile_id: str
    device_id: str
    layers: List[int]           # 分配的层索引
    total_compute: float        # 总计算负载
    total_memory: float         # 总内存需求
    estimated_time: float       # 预估执行时间 (ms)


class DistributedModelPartitioner:
    """分布式模型分割器"""
    
    def __init__(
        self,
        num_tiles: int = 4,
        strategy: PartitionStrategy = PartitionStrategy.HYBRID,
        memory_limit_per_tile: float = 8.0  # GB
    ):
        self.num_tiles = num_tiles
        self.strategy = strategy
        self.memory_limit_per_tile = memory_limit_per_tile
        self._devices_created = []
        
        # 创建虚拟设备
        self._create_mock_devices()
        
        # Llama 模型配置模板
        self.llama_configs = {
            "llama-7b": {"num_layers": 32, "hidden_size": 4096, "intermediate_size": 11008},
            "llama-13b": {"num_layers": 40, "hidden_size": 5120, "intermediate_size": 13824},
            "llama-33b": {"num_layers": 60, "hidden_size": 6656, "intermediate_size": 17920},
        }
    
    def _create_mock_devices(self):
        """创建模拟光子计算设备"""
        try:
            import lumina_kernel
            
            for i in range(self.num_tiles):
                device_name = f"Lumina-Tile-{i}"
                self._devices_created.append(device_name)
                print(f"📱 创建光子计算瓦片: {device_name}")
                
        except ImportError:
            print("⚠️  lumina_kernel 未找到，使用虚拟设备")
    
    def analyze_model_complexity(self, model_config: Dict[str, Any]) -> List[LayerProfile]:
        """分析模型复杂度"""
        profiles = []
        num_layers = model_config["num_layers"]
        hidden_size = model_config["hidden_size"] 
        intermediate_size = model_config["intermediate_size"]
        
        for layer_idx in range(num_layers):
            # 计算每个层的计算复杂度
            attention_compute = hidden_size ** 2  # 自注意力计算
            ffn_compute = hidden_size * intermediate_size  # 前馈网络计算
            total_compute = attention_compute + ffn_compute
            
            # 估算内存需求
            attention_memory = hidden_size ** 2 * 4 / (1024**2)  # MB
            ffn_memory = hidden_size * intermediate_size * 4 / (1024**2)  # MB
            total_memory = attention_memory + ffn_memory
            
            # 光子计算效率（基于计算模式）
            photonic_efficiency = self._estimate_photonic_efficiency(
                attention_compute, ffn_compute
            )
            
            profile = LayerProfile(
                layer_idx=layer_idx,
                layer_type="transformer",
                compute_units=total_compute / 1e6,  # 标准化
                memory_mb=total_memory,
                photonic_efficiency=photonic_efficiency,
                dependencies=[layer_idx - 1] if layer_idx > 0 else []
            )
            profiles.append(profile)
        
        return profiles
    
    def _estimate_photonic_efficiency(self, attention_compute: float, ffn_compute: float) -> float:
        """估算光子计算效率"""
        # 矩阵乘法在光子计算上效率更高
        total_matrix_ops = attention_compute + ffn_compute
        matrix_ratio = total_matrix_ops / (total_matrix_ops + 1e6)
        
        # 考虑相干性等因素的影响
        coherence_factor = 0.85  # 光子相干性效率
        noise_factor = 0.90      # 噪声容忍度
        
        return matrix_ratio * coherence_factor * noise_factor
    
    def partition_model(
        self, 
        model_name: str,
        model_config: Dict[str, Any]
    ) -> List[TileAssignment]:
        """分割模型到多个瓦片"""
        print(f"🔧 开始分割 {model_name} 到 {self.num_tiles} 个光子瓦片")
        print(f"📊 使用策略: {self.strategy.value}")
        
        # 分析模型复杂度
        profiles = self.analyze_model_complexity(model_config)
        
        # 根据策略进行分割
        if self.strategy == PartitionStrategy.BY_LAYERS:
            assignments = self._partition_by_layers(profiles)
        elif self.strategy == PartitionStrategy.BY_COMPUTE:
            assignments = self._partition_by_compute(profiles)
        elif self.strategy == PartitionStrategy.BY_MEMORY:
            assignments = self._partition_by_memory(profiles)
        else:  # HYBRID
            assignments = self._partition_hybrid(profiles)
        
        # 验证分割结果
        self._validate_assignments(assignments, profiles)
        
        return assignments
    
    def _partition_by_layers(self, profiles: List[LayerProfile]) -> List[TileAssignment]:
        """按层数均匀分割"""
        assignments = []
        layers_per_tile = len(profiles) // self.num_tiles
        remainder = len(profiles) % self.num_tiles
        
        start_idx = 0
        for tile_idx in range(self.num_tiles):
            # 分配层数，余数分配到前面的瓦片
            num_layers = layers_per_tile + (1 if tile_idx < remainder else 0)
            end_idx = start_idx + num_layers
            
            tile_layers = list(range(start_idx, end_idx))
            
            # 计算总负载
            total_compute = sum(profiles[i].compute_units for i in tile_layers)
            total_memory = sum(profiles[i].memory_mb for i in tile_layers)
            
            assignment = TileAssignment(
                tile_id=f"Tile-{tile_idx}",
                device_id=f"Lumina-Tile-{tile_idx}",
                layers=tile_layers,
                total_compute=total_compute,
                total_memory=total_memory,
                estimated_time=self._estimate_execution_time(total_compute)
            )
            assignments.append(assignment)
            start_idx = end_idx
        
        return assignments
    
    def _partition_by_compute(self, profiles: List[LayerProfile]) -> List[TileAssignment]:
        """按计算复杂度分割"""
        total_compute = sum(p.compute_units for p in profiles)
        target_compute_per_tile = total_compute / self.num_tiles
        
        assignments = []
        current_tile_layers = []
        current_compute = 0
        tile_idx = 0
        
        for i, profile in enumerate(profiles):
            current_tile_layers.append(i)
            current_compute += profile.compute_units
            
            # 如果当前瓦片计算量接近目标，开始下一个瓦片
            if (current_compute >= target_compute_per_tile or 
                i == len(profiles) - 1 or
                tile_idx == self.num_tiles - 1):
                
                total_memory = sum(profiles[j].memory_mb for j in current_tile_layers)
                
                assignment = TileAssignment(
                    tile_id=f"Tile-{tile_idx}",
                    device_id=f"Lumina-Tile-{tile_idx}",
                    layers=current_tile_layers.copy(),
                    total_compute=current_compute,
                    total_memory=total_memory,
                    estimated_time=self._estimate_execution_time(current_compute)
                )
                assignments.append(assignment)
                
                current_tile_layers.clear()
                current_compute = 0
                tile_idx += 1
        
        return assignments
    
    def _partition_by_memory(self, profiles: List[LayerProfile]) -> List[TileAssignment]:
        """按内存需求分割"""
        total_memory = sum(p.memory_mb for p in profiles)
        target_memory_per_tile = min(
            total_memory / self.num_tiles,
            self.memory_limit_per_tile * 1024  # GB to MB
        )
        
        assignments = []
        current_tile_layers = []
        current_memory = 0
        tile_idx = 0
        
        for i, profile in enumerate(profiles):
            # 检查内存限制
            if current_memory + profile.memory_mb > target_memory_per_tile:
                # 保存当前瓦片
                total_compute = sum(profiles[j].compute_units for j in current_tile_layers)
                
                assignment = TileAssignment(
                    tile_id=f"Tile-{tile_idx}",
                    device_id=f"Lumina-Tile-{tile_idx}",
                    layers=current_tile_layers.copy(),
                    total_compute=total_compute,
                    total_memory=current_memory,
                    estimated_time=self._estimate_execution_time(total_compute)
                )
                assignments.append(assignment)
                
                current_tile_layers.clear()
                current_memory = 0
                tile_idx += 1
            
            current_tile_layers.append(i)
            current_memory += profile.memory_mb
        
        # 处理最后一个瓦片
        if current_tile_layers and len(assignments) < self.num_tiles:
            total_compute = sum(profiles[j].compute_units for j in current_tile_layers)
            
            assignment = TileAssignment(
                tile_id=f"Tile-{tile_idx}",
                device_id=f"Lumina-Tile-{tile_idx}",
                layers=current_tile_layers,
                total_compute=total_compute,
                total_memory=current_memory,
                estimated_time=self._estimate_execution_time(total_compute)
            )
            assignments.append(assignment)
        
        return assignments
    
    def _partition_hybrid(self, profiles: List[LayerProfile]) -> List[TileAssignment]:
        """混合策略分割（推荐）"""
        # 1. 首先按内存进行初步分割确保内存限制
        memory_assignments = self._partition_by_memory(profiles)
        
        # 2. 然后在内存限制内优化计算负载
        assignments = self._balance_compute_load(memory_assignments, profiles)
        
        return assignments
    
    def _balance_compute_load(
        self, 
        initial_assignments: List[TileAssignment],
        profiles: List[LayerProfile]
    ) -> List[TileAssignment]:
        """在内存限制内平衡计算负载"""
        # 计算平均计算负载
        total_compute = sum(a.total_compute for a in initial_assignments)
        target_compute = total_compute / len(initial_assignments)
        
        assignments = initial_assignments.copy()
        
        # 简单的负载再平衡算法
        for _ in range(3):  # 多次迭代优化
            for i in range(len(assignments)):
                for j in range(len(assignments)):
                    if i != j and assignments[i].total_compute > target_compute * 1.2:
                        # 尝试移动层到负载较轻的瓦片
                        for layer_idx in assignments[i].layers.copy():
                            profile = profiles[layer_idx]
                            
                            # 检查目标瓦片是否有足够内存
                            if (assignments[j].total_memory + profile.memory_mb <= 
                                self.memory_limit_per_tile * 1024):
                                
                                # 移动层
                                assignments[i].layers.remove(layer_idx)
                                assignments[j].layers.append(layer_idx)
                                
                                # 更新负载计算
                                assignments[i].total_compute -= profile.compute_units
                                assignments[i].total_memory -= profile.memory_mb
                                assignments[j].total_compute += profile.compute_units
                                assignments[j].total_memory += profile.memory_mb
                                
                                break
        
        # 重新计算执行时间
        for assignment in assignments:
            assignment.estimated_time = self._estimate_execution_time(assignment.total_compute)
        
        return assignments
    
    def _estimate_execution_time(self, compute_units: float) -> float:
        """估算执行时间 (ms)"""
        # 假设光子计算加速比为 5x
        photonic_speedup = 5.0
        base_performance = 1000.0  # 基准性能 (compute_units/ms)
        
        return compute_units / (base_performance * photonic_speedup)
    
    def _validate_assignments(
        self, 
        assignments: List[TileAssignment],
        profiles: List[LayerProfile]
    ):
        """验证分割结果"""
        print("\n✅ 分割验证:")
        
        # 检查层数完整性
        all_layers = []
        for assignment in assignments:
            all_layers.extend(assignment.layers)
        all_layers.sort()
        
        expected_layers = list(range(len(profiles)))
        if all_layers != expected_layers:
            raise ValueError(f"层数不匹配: 期望 {expected_layers}, 实际 {all_layers}")
        
        # 检查内存限制
        memory_violations = 0
        for assignment in assignments:
            if assignment.total_memory > self.memory_limit_per_tile * 1024:
                memory_violations += 1
        
        if memory_violations > 0:
            print(f"⚠️  {memory_violations} 个瓦片超出内存限制")
        else:
            print("✅ 所有瓦片内存使用正常")
        
        # 计算负载均衡度
        compute_loads = [a.total_compute for a in assignments]
        avg_load = sum(compute_loads) / len(compute_loads)
        load_variance = sum((load - avg_load) ** 2 for load in compute_loads) / len(compute_loads)
        load_std = math.sqrt(load_variance)
        load_balance_ratio = 1.0 - (load_std / avg_load) if avg_load > 0 else 0
        
        print(f"📊 负载均衡度: {load_balance_ratio:.2%}")
        print(f"⏱️  平均执行时间: {avg_load / 1000:.2f} ms")
    
    def print_partition_summary(self, assignments: List[TileAssignment]):
        """打印分割摘要"""
        print(f"\n🎯 分布式分割摘要:")
        print("=" * 80)
        
        total_time = max(a.estimated_time for a in assignments)
        
        for i, assignment in enumerate(assignments):
            print(f"\n📱 {assignment.tile_id} ({assignment.device_id})")
            print(f"   层范围: {min(assignment.layers)}-{max(assignment.layers)} ({len(assignment.layers)} 层)")
            print(f"   计算负载: {assignment.total_compute:.1f} units")
            print(f"   内存需求: {assignment.total_memory:.1f} MB ({assignment.total_memory/1024:.2f} GB)")
            print(f"   预估时间: {assignment.estimated_time:.2f} ms")
            print(f"   负载占比: {assignment.total_compute / sum(a.total_compute for a in assignments) * 100:.1f}%")
        
        print(f"\n⚡ 总体性能:")
        print(f"   并行加速比: ~{len(assignments)}x")
        print(f"   瓶颈时间: {total_time:.2f} ms")
        print(f"   内存效率: {sum(a.total_memory for a in assignments) / 1024:.2f} GB")