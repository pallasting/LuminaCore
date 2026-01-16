"""
分布式光子计算执行协调器

管理多个光子计算瓦片的并行执行，处理数据依赖和同步
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
import uuid

from .partitioner import TileAssignment, LayerProfile


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComputeTask:
    """计算任务"""
    task_id: str
    tile_id: str
    device_id: str
    layer_idx: int
    input_data: Any
    dependencies: List[str]  # 依赖的任务ID
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None


@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time: float = 0.0
    tile_utilization: Dict[str, float] = field(default_factory=dict)
    communication_overhead: float = 0.0
    synchronization_time: float = 0.0


class DistributedExecutor:
    """分布式执行协调器"""
    
    def __init__(
        self,
        assignments: List[TileAssignment],
        max_workers_per_tile: int = 2,
        communication_bandwidth: float = 100.0  # GB/s
    ):
        self.assignments = assignments
        self.max_workers_per_tile = max_workers_per_tile
        self.communication_bandwidth = communication_bandwidth
        
        # 创建任务队列和执行器
        self.task_queues: Dict[str, queue.Queue] = {}
        self.executors: Dict[str, ThreadPoolExecutor] = {}
        self.running_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: Dict[str, ComputeTask] = {}
        
        # 同步机制
        self.task_dependencies: Dict[str, List[str]] = {}
        self.completion_events: Dict[str, threading.Event] = {}
        
        # 性能监控
        self.metrics = ExecutionMetrics()
        self.tile_metrics: Dict[str, Dict[str, float]] = {}
        
        # 初始化执行环境
        self._initialize_execution_environment()
    
    def _initialize_execution_environment(self):
        """初始化执行环境"""
        print("🚀 初始化分布式执行环境...")
        
        for assignment in self.assignments:
            tile_id = assignment.tile_id
            device_id = assignment.device_id
            
            # 创建任务队列
            self.task_queues[tile_id] = queue.Queue()
            
            # 创建线程池执行器
            self.executors[tile_id] = ThreadPoolExecutor(
                max_workers=self.max_workers_per_tile,
                thread_name_prefix=f"Photonic-{tile_id}"
            )
            
            # 初始化瓦片指标
            self.tile_metrics[tile_id] = {
                "tasks_executed": 0,
                "total_time": 0.0,
                "communication_time": 0.0,
                "memory_peak": 0.0
            }
            
            print(f"   ✅ {tile_id} 就绪 (设备: {device_id})")
        
        print("🔧 分布式执行环境初始化完成")
    
    def create_execution_plan(
        self, 
        input_data: Any,
        layer_profiles: List[LayerProfile]
    ) -> List[ComputeTask]:
        """创建执行计划"""
        print("📋 创建分布式执行计划...")
        
        tasks = []
        task_counter = 0
        
        # 为每个瓦片创建任务
        for assignment in self.assignments:
            tile_id = assignment.tile_id
            layers = assignment.layers
            
            for layer_idx in layers:
                task_id = f"task-{tile_id}-{layer_idx}"
                
                # 确定依赖关系
                dependencies = []
                if layer_idx > 0:
                    # 找到前一层的任务ID
                    prev_layer = layer_idx - 1
                    for prev_assignment in self.assignments:
                        if prev_layer in prev_assignment.layers:
                            dependencies.append(f"task-{prev_assignment.tile_id}-{prev_layer}")
                            break
                
                # 创建计算任务
                task = ComputeTask(
                    task_id=task_id,
                    tile_id=tile_id,
                    device_id=assignment.device_id,
                    layer_idx=layer_idx,
                    input_data=input_data,  # 简化：实际应该传递具体数据
                    dependencies=dependencies
                )
                
                tasks.append(task)
                self.task_dependencies[task_id] = dependencies
                self.completion_events[task_id] = threading.Event()
                
                task_counter += 1
        
        self.metrics.total_tasks = len(tasks)
        print(f"   📊 创建了 {len(tasks)} 个计算任务")
        
        return tasks
    
    def execute_distributed(
        self,
        tasks: List[ComputeTask],
        progress_callback: Optional[Callable[[str, TaskStatus], None]] = None
    ) -> Dict[str, Any]:
        """执行分布式计算"""
        print("⚡ 开始分布式光子计算...")
        
        start_time = time.time()
        
        # 提交任务到相应瓦片
        for task in tasks:
            self._submit_task(task, progress_callback)
        
        # 等待所有任务完成
        self._wait_for_completion(tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算执行指标
        self._calculate_metrics(total_time)
        
        # 收集结果
        results = self._collect_results(tasks)
        
        print(f"✅ 分布式执行完成，总时间: {total_time:.3f}s")
        
        return {
            "results": results,
            "metrics": self.metrics,
            "tile_metrics": self.tile_metrics,
            "execution_time": total_time
        }
    
    def _submit_task(
        self,
        task: ComputeTask,
        progress_callback: Optional[Callable[[str, TaskStatus], None]] = None
    ):
        """提交任务到瓦片"""
        def execute_task():
            tile_start_time = time.time()
            
            try:
                # 等待依赖完成
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        print(f"⏳ {task.task_id} 等待依赖 {dep_id}")
                        self.completion_events[dep_id].wait()
                
                # 更新任务状态
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                self.running_tasks[task.task_id] = task
                
                if progress_callback:
                    progress_callback(task.task_id, TaskStatus.RUNNING)
                
                # 执行光子计算
                result = self._execute_photonic_computation(task)
                
                # 更新任务结果
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.end_time = time.time()
                task.execution_time = task.end_time - task.start_time
                
                # 移动到完成队列
                self.completed_tasks[task.task_id] = task
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # 通知等待者
                self.completion_events[task.task_id].set()
                
                if progress_callback:
                    progress_callback(task.task_id, TaskStatus.COMPLETED)
                
                print(f"✅ {task.task_id} 完成 ({task.execution_time:.3f}s)")
                
            except Exception as e:
                # 处理错误
                task.error = str(e)
                task.status = TaskStatus.FAILED
                task.end_time = time.time()
                
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                self.completion_events[task.task_id].set()
                
                if progress_callback:
                    progress_callback(task.task_id, TaskStatus.FAILED)
                
                print(f"❌ {task.task_id} 失败: {e}")
            
            finally:
                # 更新瓦片指标
                tile_end_time = time.time()
                tile_time = tile_end_time - tile_start_time
                self.tile_metrics[task.tile_id]["tasks_executed"] += 1
                self.tile_metrics[task.tile_id]["total_time"] += tile_time
        
        # 提交到相应瓦片的执行器
        executor = self.executors[task.tile_id]
        future = executor.submit(execute_task)
        
        print(f"📤 任务 {task.task_id} 已提交到 {task.tile_id}")
    
    def _execute_photonic_computation(self, task: ComputeTask) -> Any:
        """执行光子计算"""
        # 模拟执行（演示模式）
        compute_time = 0.1 + (task.layer_idx % 3) * 0.05  # 模拟不同层的时间
        time.sleep(compute_time)
        
        return {
            "layer_idx": task.layer_idx,
            "tile_id": task.tile_id,
            "result": f"photonic_output_{task.layer_idx}",
            "execution_time": compute_time
        }
    
    def _generate_microcode_for_layer(self, layer_idx: int) -> str:
        """为特定层生成微码"""
        # 简化的微码生成
        microcode = f"""
# Llama Layer {layer_idx} Photonic Microcode
LAYER_SETUP idx={layer_idx}
PHOTONIC_MATRIX_MULTIPLY size=4096
QUANTIZE bits=8
NOISE_INJECTION std=0.01
OUTPUT_STORE
"""
        return microcode.strip()
    
    def _wait_for_completion(self, tasks: List[ComputeTask]):
        """等待所有任务完成"""
        print("⏳ 等待所有任务完成...")
        
        completed_count = 0
        total_tasks = len(tasks)
        
        while completed_count < total_tasks:
            time.sleep(0.1)  # 避免忙等待
            
            completed_count = len(self.completed_tasks)
            failed_count = sum(1 for t in self.completed_tasks.values() 
                             if t.status == TaskStatus.FAILED)
            
            # 打印进度
            if completed_count % 5 == 0 or completed_count == total_tasks:
                progress = (completed_count / total_tasks) * 100
                print(f"   📈 进度: {progress:.1f}% ({completed_count}/{total_tasks})")
        
        # 等待所有执行器完成
        for executor in self.executors.values():
            executor.shutdown(wait=True)
    
    def _calculate_metrics(self, total_time: float):
        """计算执行指标"""
        completed_tasks = [t for t in self.completed_tasks.values() 
                          if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.completed_tasks.values() 
                       if t.status == TaskStatus.FAILED]
        
        self.metrics.completed_tasks = len(completed_tasks)
        self.metrics.failed_tasks = len(failed_tasks)
        self.metrics.total_time = total_time
        
        # 计算瓦片利用率
        for tile_id in self.tile_metrics:
            tile_time = self.tile_metrics[tile_id]["total_time"]
            utilization = tile_time / total_time if total_time > 0 else 0
            self.metrics.tile_utilization[tile_id] = min(utilization, 1.0)
        
        # 估算通信开销 (简化模型)
        num_tiles = len(self.assignments)
        estimated_communication = (num_tiles - 1) * 0.01  # 每次通信 10ms
        self.metrics.communication_overhead = estimated_communication
        
        # 估算同步时间
        self.metrics.synchronization_time = total_time * 0.05  # 5% 同步开销
    
    def _collect_results(self, tasks: List[ComputeTask]) -> Dict[str, Any]:
        """收集执行结果"""
        results = {
            "layer_outputs": {},
            "tile_outputs": {},
            "failed_tasks": {},
            "performance_summary": {}
        }
        
        # 按层收集结果
        for task in tasks:
            if task.status == TaskStatus.COMPLETED:
                results["layer_outputs"][task.layer_idx] = {
                    "tile_id": task.tile_id,
                    "result": task.result,
                    "execution_time": task.execution_time,
                    "task_id": task.task_id
                }
                
                # 按瓦片组织结果
                if task.tile_id not in results["tile_outputs"]:
                    results["tile_outputs"][task.tile_id] = []
                results["tile_outputs"][task.tile_id].append({
                    "layer_idx": task.layer_idx,
                    "result": task.result,
                    "execution_time": task.execution_time
                })
                
            elif task.status == TaskStatus.FAILED:
                results["failed_tasks"][task.task_id] = {
                    "error": task.error,
                    "layer_idx": task.layer_idx,
                    "tile_id": task.tile_id
                }
        
        return results
    
    def print_execution_summary(self, execution_result: Dict[str, Any]):
        """打印执行摘要"""
        print(f"\n🎯 分布式执行摘要")
        print("=" * 80)
        
        metrics = execution_result["metrics"]
        tile_metrics = execution_result["tile_metrics"]
        results = execution_result["results"]
        
        print(f"\n📊 总体指标:")
        print(f"   总任务数: {metrics.total_tasks}")
        print(f"   成功任务: {metrics.completed_tasks}")
        print(f"   失败任务: {metrics.failed_tasks}")
        print(f"   总执行时间: {metrics.total_time:.3f}s")
        print(f"   通信开销: {metrics.communication_overhead:.3f}s")
        print(f"   同步时间: {metrics.synchronization_time:.3f}s")
        
        print(f"\n📱 瓦片性能:")
        for tile_id, utilization in metrics.tile_utilization.items():
            tile_metric = tile_metrics[tile_id]
            avg_time = (tile_metric["total_time"] / tile_metric["tasks_executed"] 
                       if tile_metric["tasks_executed"] > 0 else 0)
            print(f"   {tile_id}:")
            print(f"     利用率: {utilization:.1%}")
            print(f"     执行任务: {tile_metric['tasks_executed']}")
            print(f"     平均时间: {avg_time:.3f}s")
        
        print(f"\n⚡ 性能分析:")
        if metrics.total_time > 0:
            throughput = metrics.completed_tasks / metrics.total_time
            efficiency = sum(metrics.tile_utilization.values()) / len(metrics.tile_utilization)
            print(f"   吞吐量: {throughput:.1f} tasks/s")
            print(f"   并行效率: {efficiency:.1%}")
            
            if len(self.assignments) > 1:
                speedup = len(self.assignments) * efficiency
                print(f"   实际加速比: {speedup:.2f}x")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理分布式执行环境...")
        
        for executor in self.executors.values():
            executor.shutdown(wait=False)
        
        self.task_queues.clear()
        self.executors.clear()
        self.running_tasks.clear()
        self.completed_tasks.clear()
        self.task_dependencies.clear()
        self.completion_events.clear()
        
        print("✅ 清理完成")