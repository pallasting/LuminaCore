
import torch
import lumina as lnn
import time
import numpy as np

def benchmark_optical_vs_linear():
    """对比光子层与传统PyTorch层的性能"""
    
    # 创建层
    optical_layer = lnn.layers.OpticalLinear(784, 256, hardware_profile="datacenter_high_precision")
    torch_layer = torch.nn.Linear(784, 256)
    
    # 使用相同的权重
    with torch.no_grad():
        torch_layer.weight.copy_(optical_layer.weight)
        if optical_layer.bias is not None:
            torch_layer.bias.copy_(optical_layer.bias)
    
    # 测试数据
    batch_sizes = [1, 8, 32, 64, 128]
    num_iterations = 100
    
    print("📊 性能基准测试")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Optical (ms)':<15} {'Torch (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 784)
        
        # 光子层测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_iterations):
            y_optical = optical_layer(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        optical_time = (time.time() - start) * 1000 / num_iterations
        
        # PyTorch层测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_iterations):
            y_torch = torch_layer(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch_time = (time.time() - start) * 1000 / num_iterations
        
        speedup = torch_time / optical_time
        print(f"{batch_size:<12} {optical_time:<15.3f} {torch_time:<15.3f} {speedup:<10.2f}x")
    
    print("\n🎉 基准测试完成！")

if __name__ == "__main__":
    benchmark_optical_vs_linear()
