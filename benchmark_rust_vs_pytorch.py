
import torch
import lumina as lnn
import time
import numpy as np
import torch.nn.functional as F
from lumina.nn import OpticalLinear
import os

def benchmark_optical_vs_linear():
    """对比光子层与传统PyTorch层的性能"""
    
    # 增大尺寸以体现 Rust 后端优势
    in_dim = 1024
    out_dim = 1024
    
    # 创建层
    optical_layer = OpticalLinear(in_dim, out_dim, hardware_profile="datacenter_high_precision")
    
    # 定义 PyTorch 路径的模拟过程 (包含噪声和量化)
    def pytorch_full_sim(x, layer):
        with torch.no_grad():
            # 1. DAC
            x_q = layer.dac_convert(x)
            # 2. Matmul + Noise
            y = F.linear(x_q, layer.weight, None)
            y_n = layer.noise_model.apply_noise(y, True)
            # 3. ADC
            y_out = layer.adc_convert(y_n)
            return y_out

    # 测试数据
    batch_sizes = [32, 64, 128]
    num_iterations = 20
    
    print("📊 性能基准测试 (训练模式：包含噪声与量化)")
    print("=" * 70)
    print(f"{'Batch Size':<12} {'Rust Fused (ms)':<18} {'PyTorch Full (ms)':<18} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, in_dim)
        
        # 1. Rust 融合算子测试
        # 确保在训练模式下，但我们手动调用 _forward_rust 来测试它
        # 因为 OpticalLinear.forward 目前在 training=True 时会回退
        os.environ["LUMINA_USE_RUST"] = "1"
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_iterations):
            y_rust = optical_layer._forward_rust(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        rust_time = (time.time() - start) * 1000 / num_iterations
        
        # 2. PyTorch 全模拟路径测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_iterations):
            y_torch = pytorch_full_sim(x, optical_layer)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_time = (time.time() - start) * 1000 / num_iterations
        
        speedup = torch_time / rust_time
        print(f"{batch_size:<12} {rust_time:<18.3f} {torch_time:<18.3f} {speedup:<10.2f}x")
    
    print("\n🎉 基准测试完成！")

if __name__ == "__main__":
    benchmark_optical_vs_linear()
