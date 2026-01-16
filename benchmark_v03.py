import torch
import torch.nn as nn
import json
import os
import time
from lumina.layers import OpticalLinear, ComplexOpticalLinear, OpticalAttention
from lumina.compiler import LuminaExporter
from lumina.compiler.instruction_set import MicroCodeCompiler
import lumina_kernel

def benchmark_v03():
    print("🚀 LuminaFlow v0.3.0 性能基准测试 & 功能验证")
    print("="*60)
    
    # 1. 验证微码编译器 (Attention 融合)
    print("\n[MCC] 验证 Transformer 算子融合...")
    model = nn.Sequential(
        OpticalAttention(embed_dim=512, num_heads=8),
        ComplexOpticalLinear(512, 128)
    )
    exporter = LuminaExporter(output_dir="bench_exports")
    peg_path = exporter.export_execution_graph(model, input_shape=(1, 512))
    
    mcc = MicroCodeCompiler()
    bin_path = mcc.compile(peg_path)
    
    with open(bin_path, 'r') as f:
        instructions = json.load(f)
    
    has_attn = any(inst["op"] == "EXEC_ATTN_MASK" for inst in instructions)
    print(f"✅ 微码指令生成成功: {len(instructions)} 条指令")
    print(f"✅ 算子融合验证: {'PASSED' if has_attn else 'FAILED'}")

    # 2. 验证 Rust Runtime
    print("\n[Runtime] 验证 Rust 运行时原型...")
    try:
        success = lumina_kernel.run_microcode(json.dumps(instructions))
        print(f"✅ Rust Runtime 执行成功: {success}")
    except Exception as e:
        print(f"❌ Rust Runtime 执行失败: {e}")

    # 3. 复数矩阵乘法性能测试 (Rust vs PyTorch)
    print("\n[Performance] 复数矩阵乘法 (Complex32) 基准测试...")
    batch_size = 64
    in_features = 1024
    out_features = 1024
    
    x = torch.randn(batch_size, in_features, dtype=torch.complex64)
    w = torch.randn(out_features, in_features, dtype=torch.complex64)
    
    # PyTorch CPU
    start = time.perf_counter()
    for _ in range(10):
        _ = torch.matmul(x, w.t())
    pt_time = (time.perf_counter() - start) / 10
    print(f"PyTorch CPU (Avg): {pt_time*1000:.2f} ms")
    
    # Rust Kernel
    x_np = x.numpy()
    w_np = w.numpy()
    # Warmup
    _ = lumina_kernel.complex_matmul(x_np, w_np)
    
    start = time.perf_counter()
    for _ in range(10):
        _ = lumina_kernel.complex_matmul(x_np, w_np)
    rust_time = (time.perf_counter() - start) / 10
    print(f"Lumina Rust Kernel (Avg): {rust_time*1000:.2f} ms")
    print(f"🚀 加速比 (Speedup): {pt_time/rust_time:.2f}x")

if __name__ == "__main__":
    benchmark_v03()
    # 清理
    import shutil
    if os.path.exists("bench_exports"):
        shutil.rmtree("bench_exports")
