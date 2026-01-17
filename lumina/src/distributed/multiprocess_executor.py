#!/usr/bin/env python3
"""
Multi-Process Executor Implementation

使用 Python multiprocessing 模块绕过 GIL 限制，实现真正的并行计算。
每个进程代表一个独立的光子计算瓦片 (Tile)。
"""

import torch
import numpy as np
import time
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os

# 导入 Rust 内核 (在子进程中需要重新导入)
import lumina_kernel

@dataclass
class TaskPayload:
    """任务负载"""
    task_id: str
    layer_idx: int
    input_data: np.ndarray  # 使用 numpy 传递，避免 torch 序列化问题
    weight: np.ndarray
    bias: Optional[np.ndarray]
    noise_std: float
    bits: int
    seed: int

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    output_data: np.ndarray
    execution_time: float
    process_id: int
    error: Optional[str] = None

def _worker_process(
    tile_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    device_config: Dict[str, Any]
):
    """
    工作进程函数 (独立运行在单独的进程中)
    """
    pid = os.getpid()
    print(f"   [Worker-{tile_id}] Started (PID: {pid})")
    
    # 在进程内初始化 Rust 模拟设备
    # 注意：每个进程有自己的内存空间，所以 Rust 端的静态变量是隔离的
    lumina_kernel.create_mock_device(f"Device-{tile_id}", 8 * 1024**3)
    
    while True:
        try:
            task: Optional[TaskPayload] = task_queue.get()
            if task is None:  # 终止信号
                break
            
            start_time = time.time()
            
            # 调用 Rust 内核执行计算
            # 注意：这里的数据已经是 numpy 数组
            output_np = lumina_kernel.optical_linear_fused(
                task.input_data,
                task.weight,
                task.bias,
                task.noise_std,
                task.bits,
                task.seed
            )
            
            exec_time = time.time() - start_time
            
            result = TaskResult(
                task_id=task.task_id,
                output_data=output_np,
                execution_time=exec_time,
                process_id=pid
            )
            
            result_queue.put(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_queue.put(TaskResult(
                task_id=task.task_id if 'task' in locals() else "unknown",
                output_data=np.array([]),
                execution_time=0.0,
                process_id=pid,
                error=str(e)
            ))
            
    print(f"   [Worker-{tile_id}] Stopped")

class MultiProcessExecutor:
    """
    多进程执行器
    
    特点:
    - 真正的并行执行 (绕过 GIL)
    - 进程隔离，模拟独立的计算节点
    - 基于 Queue 的通信
    """
    
    def __init__(self, num_tiles: int = 4):
        self.num_tiles = num_tiles
        self.processes: List[mp.Process] = []
        self.task_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        self.running = False
        
        print(f"🚀 Initializing MultiProcessExecutor with {num_tiles} tiles")
        self._start_processes()
        
    def _start_processes(self):
        """启动工作进程"""
        self.running = True
        for i in range(self.num_tiles):
            task_q = mp.Queue()
            result_q = mp.Queue()
            
            p = mp.Process(
                target=_worker_process,
                args=(f"Tile-{i}", task_q, result_q, {})
            )
            p.daemon = True
            p.start()
            
            self.processes.append(p)
            self.task_queues.append(task_q)
            self.result_queues.append(result_q)
            
    def execute_batch_parallel(
        self, 
        layers_data: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        并行执行一批层任务
        
        Args:
            layers_data: 任务列表，每个元素包含 input, weight, etc.
            
        Returns:
            结果列表
        """
        num_tasks = len(layers_data)
        if num_tasks == 0:
            return []
            
        # 分发任务
        for i, data in enumerate(layers_data):
            tile_idx = i % self.num_tiles
            
            # 准备数据 (转为 numpy)
            input_np = data['input']
            if isinstance(input_np, torch.Tensor):
                input_np = input_np.detach().cpu().numpy()
                
            weight_np = data['weight']
            if isinstance(weight_np, torch.Tensor):
                weight_np = weight_np.detach().cpu().numpy()
                
            bias_np = data.get('bias')
            if isinstance(bias_np, torch.Tensor):
                bias_np = bias_np.detach().cpu().numpy()
                
            payload = TaskPayload(
                task_id=f"Task-{i}",
                layer_idx=i,
                input_data=input_np,
                weight=weight_np,
                bias=bias_np,
                noise_std=data.get('noise_std', 0.01),
                bits=data.get('bits', 8),
                seed=42 + i
            )
            
            self.task_queues[tile_idx].put(payload)
            
        # 收集结果
        results = [None] * num_tasks
        collected = 0
        
        # 简单的轮询收集 (实际应该使用更复杂的异步机制)
        # 这里为了演示简单，我们假设按顺序收集，或者使用 ID 映射
        pending_tiles = list(range(min(num_tasks, self.num_tiles))) # 活跃的 tiles
        tile_task_counts = [0] * self.num_tiles
        for i in range(num_tasks):
            tile_task_counts[i % self.num_tiles] += 1
            
        for tile_idx in range(self.num_tiles):
            count = tile_task_counts[tile_idx]
            for _ in range(count):
                res = self.result_queues[tile_idx].get()
                if res.error:
                    print(f"❌ Error in task {res.task_id}: {res.error}")
                else:
                    # 解析 task_id 获取索引 "Task-{i}"
                    idx = int(res.task_id.split('-')[1])
                    results[idx] = res.output_data
                    
        return results

    def shutdown(self):
        """关闭执行器"""
        print("🛑 Shutting down MultiProcessExecutor...")
        for q in self.task_queues:
            q.put(None)
            
        for p in self.processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        
        self.running = False
        print("✅ Shutdown complete")

def benchmark_multiprocess():
    """基准测试：多线程 vs 多进程"""
    print("=" * 60)
    print("Multi-Process vs Multi-Thread Benchmark")
    print("=" * 60)
    
    num_layers = 16
    hidden_size = 2048
    batch_size = 8
    
    # 准备数据
    weights = [np.random.randn(hidden_size, hidden_size).astype(np.float32) for _ in range(num_layers)]
    inputs = [np.random.randn(batch_size, hidden_size).astype(np.float32) for _ in range(num_layers)]
    
    tasks = []
    for i in range(num_layers):
        tasks.append({
            "input": inputs[i],
            "weight": weights[i],
            "layer_idx": i
        })
        
    # 1. 多进程测试
    print("\n🚀 Multi-Process Execution...")
    mp_executor = MultiProcessExecutor(num_tiles=4)
    
    start = time.time()
    mp_results = mp_executor.execute_batch_parallel(tasks)
    mp_time = time.time() - start
    
    mp_executor.shutdown()
    print(f"   ✅ Time: {mp_time:.4f}s")
    
    # 2. 多线程测试 (使用之前的 ThreadPoolExecutor)
    print("\n🧵 Multi-Thread Execution (GIL limited)...")
    from concurrent.futures import ThreadPoolExecutor
    
    def thread_worker(task):
        return lumina_kernel.optical_linear_fused(
            task['input'], task['weight'], None, 0.01, 8, 42
        )
        
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(thread_worker, tasks))
    mt_time = time.time() - start
    print(f"   ✅ Time: {mt_time:.4f}s")
    
    # 3. 串行基准
    print("\n🐢 Sequential Execution...")
    start = time.time()
    for task in tasks:
        thread_worker(task)
    seq_time = time.time() - start
    print(f"   ✅ Time: {seq_time:.4f}s")
    
    print("\n" + "=" * 60)
    print("📊 Comparison Results")
    print("=" * 60)
    print(f"   Sequential:   {seq_time:.4f}s (1.00x)")
    print(f"   Multi-Thread: {mt_time:.4f}s ({seq_time/mt_time:.2f}x)")
    print(f"   Multi-Process:{mp_time:.4f}s ({seq_time/mp_time:.2f}x)")
    print("\n   Note: Multi-process overhead (IPC) might affect small tasks.")
    print("   For large matrix operations, MP should win significantly.")

if __name__ == "__main__":
    #必须要加这个，否则多进程在某些系统会报错
    mp.set_start_method('spawn', force=True)
    benchmark_multiprocess()
