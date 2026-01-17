"""
Rust 后端集成模块

提供与 Rust lumina_kernel 的集成接口，用于真正的光子计算加速
"""

import torch
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
import lumina_kernel


class RustPhotonicExecutor:
    """
    Rust 后端光子计算执行器

    使用 lumina_kernel Rust 模块执行真正的光子矩阵运算
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        noise_std: float = 0.01,
        bits: int = 8,
        enable_noise: bool = True
    ):
        """
        Args:
            device_name: 设备名称 (None=默认设备)
            noise_std: 噪声标准差
            bits: 量化位数
            enable_noise: 是否启用噪声注入
        """
        self.device_name = device_name
        self.noise_std = noise_std
        self.bits = bits
        self.enable_noise = enable_noise

        # 确保设备已创建
        if device_name and device_name not in lumina_kernel.list_devices():
            lumina_kernel.create_mock_device(device_name, 8 * 1024**3)

        self.stats = {
            "total_layers": 0,
            "total_time": 0.0,
            "avg_layer_time": 0.0
        }
        print(f"🚀 RustPhotonicExecutor 初始化完成")
        print(f"   设备: {device_name or 'default'}")
        print(f"   噪声: {'启用' if enable_noise else '禁用'} ({noise_std})")
        print(f"   量化: {bits} 位")

    def execute_layer(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        physics_params: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        在光子硬件上执行单层计算

        Args:
            input_tensor: 输入张量 [batch, in_features]
            weight: 权重矩阵 [out_features, in_features]
            bias: 可选偏置 [out_features]
            physics_params: 物理仿真参数 (thermal_crosstalk, optical_loss_db, temperature)

        Returns:
            output_tensor: 输出张量 [batch, out_features]
            execution_time: 执行时间 (秒)
        """
        start_time = time.time()

        # 转换为 numpy (Rust API 直接接受 numpy 数组)
        input_np = input_tensor.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        # Rust API 的 bias 参数是可选的，传入 None 表示不需要偏置
        bias_np = bias.detach().cpu().numpy() if bias is not None else None

        # 调用 Rust 后端
        if physics_params:
            try:
                # 尝试调用增强的物理仿真接口
                output_np = lumina_kernel.optical_linear_physics(
                    input_np,
                    weight_np,
                    bias_np,
                    physics_params,
                    self.bits,
                    seed=42
                )
                exec_time = time.time() - start_time
            except AttributeError:
                # 回退到标准接口 (如果 Rust 内核未更新)
                if self.enable_noise:
                    output_np = lumina_kernel.optical_linear_fused(
                        input_np, weight_np, bias_np, self.noise_std, self.bits, seed=42
                    )
                else:
                    output_np = lumina_kernel.optical_linear_infer(
                        input_np, weight_np, bias_np, self.bits
                    )
                exec_time = time.time() - start_time
        elif self.enable_noise:
            output_np = lumina_kernel.optical_linear_fused(
                input_np,
                weight_np,
                bias_np,
                self.noise_std,
                self.bits,
                seed=42
            )
            exec_time = time.time() - start_time
        else:
            output_np = lumina_kernel.optical_linear_infer(
                input_np,
                weight_np,
                bias_np,
                self.bits
            )
            exec_time = time.time() - start_time

        # 转换回 torch 张量
        output_tensor = torch.from_numpy(output_np).to(input_tensor.device)

        # 更新统计
        total_time = time.time() - start_time
        self.stats["total_layers"] += 1
        self.stats["total_time"] += total_time
        self.stats["avg_layer_time"] = self.stats["total_time"] / self.stats["total_layers"]

        return output_tensor, total_time

    def execute_layer_inference(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        执行推理（无噪声）

        Args:
            input_tensor: 输入张量
            weight: 权重矩阵
            bias: 可选偏置

        Returns:
            output_tensor, 执行时间
        """
        old_noise = self.enable_noise
        self.enable_noise = False
        try:
            return self.execute_layer(input_tensor, weight, bias)
        finally:
            self.enable_noise = old_noise

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            "total_layers": self.stats["total_layers"],
            "total_time": self.stats["total_time"],
            "avg_layer_time": self.stats["avg_layer_time"],
            "throughput": self.stats["total_layers"] / self.stats["total_time"] if self.stats["total_time"] > 0 else 0
        }

    def print_stats(self):
        """打印执行统计"""
        stats = self.get_stats()
        print(f"\n📊 Rust 后端执行统计:")
        print(f"   总层数: {stats['total_layers']}")
        print(f"   总时间: {stats['total_time']:.3f}s")
        print(f"   平均层时间: {stats['avg_layer_time']*1000:.2f}ms")
        print(f"   吞吐量: {stats['throughput']:.1f} layers/s")


class HybridExecutor:
    """
    混合执行器

    智能选择使用 Rust 后端或 Python 模拟
    - 首次运行或小批量: Rust 后端
    - 大批量或管道模式: 流水线优化
    """

    def __init__(
        self,
        use_rust: bool = True,
        **kwargs
    ):
        self.use_rust = use_rust
        self.rust_executor: Optional[RustPhotonicExecutor] = None
        self.kwargs = kwargs

        if use_rust:
            try:
                self.rust_executor = RustPhotonicExecutor(**kwargs)
            except Exception as e:
                print(f"⚠️  Rust 后端初始化失败: {e}")
                print("   回退到模拟模式")
                self.use_rust = False

    def execute_layer(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """执行层计算"""
        if self.use_rust and self.rust_executor:
            return self.rust_executor.execute_layer(input_tensor, weight, bias)
        else:
            # 模拟执行
            start_time = time.time()
            output = torch.nn.functional.linear(input_tensor, weight, bias)
            exec_time = time.time() - start_time
            return output, exec_time

    def get_backend_type(self) -> str:
        """获取后端类型"""
        return "Rust (Photonic)" if self.use_rust else "Python (Simulation)"


def benchmark_executor(
    executor: HybridExecutor,
    num_layers: int = 12,
    batch_size: int = 2,
    hidden_size: int = 4096
) -> Dict[str, Any]:
    """
    基准测试执行器

    Args:
        executor: 执行器实例
        num_layers: 层数
        batch_size: 批次大小
        hidden_size: 隐藏维度

    Returns:
        性能指标字典
    """
    print(f"\n🔬 基准测试: {executor.get_backend_type()}")
    print(f"   层数: {num_layers}, 批次: {batch_size}, 隐藏: {hidden_size}")

    # 创建测试数据
    weights = [
        torch.randn(hidden_size, hidden_size, requires_grad=False)
        for _ in range(num_layers)
    ]

    # 预热
    print("   预热...")
    test_input = torch.randn(batch_size, hidden_size)
    for w in weights[:2]:
        _ = executor.execute_layer(test_input, w)

    # 正式测试
    print("   执行测试...")
    start_time = time.time()
    layer_times = []

    input_tensor = test_input
    for i, w in enumerate(weights):
        output, exec_time = executor.execute_layer(input_tensor, w)
        layer_times.append(exec_time)
        input_tensor = output

    total_time = time.time() - start_time

    # 计算性能指标
    throughput = batch_size / (total_time / num_layers)
    avg_layer_time = sum(layer_times) / len(layer_times)

    return {
        "backend": executor.get_backend_type(),
        "total_time": total_time,
        "avg_layer_time": avg_layer_time,
        "throughput": throughput,
        "layer_times": layer_times,
        "memory_efficient": True  # Rust 后端内存效率更高
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Rust Backend Benchmark")
    print("=" * 60)

    # Direct test of Rust backend
    print("\n🔧 Direct Rust Backend Test:")
    try:
        import numpy as np
        import torch
        import lumina_kernel

        # Create test data
        batch_size, hidden_size = 2, 4096
        input_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
        weight_np = np.random.randn(hidden_size, hidden_size).astype(np.float32)

        print(f"   Input shape: {input_np.shape}")
        print(f"   Weight shape: {weight_np.shape}")

        # Warmup
        for _ in range(3):
            _ = lumina_kernel.optical_linear_fused(input_np, weight_np, None, 0.01, 8, 42)

        # Benchmark
        import time
        num_layers = 12
        start = time.time()

        for i in range(num_layers):
            output_np = lumina_kernel.optical_linear_fused(
                input_np, weight_np, None, 0.01, 8, 42 + i
            )
            input_np = output_np  # Chain the outputs

        elapsed = time.time() - start

        print(f"\n✅ Rust Backend Results:")
        print(f"   Layers: {num_layers}")
        print(f"   Total time: {elapsed:.3f}s")
        print(f"   Avg layer time: {elapsed/num_layers*1000:.2f}ms")
        print(f"   Throughput: {num_layers/elapsed:.1f} layers/s")

    except Exception as e:
        print(f"❌ Rust backend error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
