#!/usr/bin/env python3
"""
测试 Python-Rust FFI 绑定

运行前需要先构建: maturin develop
"""

import numpy as np

try:
    import lumina_kernel
    
    print("=" * 60)
    print("LuminaKernel FFI 测试")
    print("=" * 60)
    
    # 基础测试
    print("\n1️⃣ 基础功能测试")
    print("✅ 成功导入 lumina_kernel 模块")
    print(f"📦 版本: {lumina_kernel.version()}")
    print(f"👋 {lumina_kernel.hello_lumina()}")
    
    # 融合算子测试
    print("\n2️⃣ 融合算子测试（训练模式）")
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bias = np.array([0.5, 0.5], dtype=np.float32)
    
    output = lumina_kernel.optical_linear_fused(
        input_data,
        weight,
        bias,
        noise_std=0.1,
        bits=8,
        seed=42
    )
    
    print(f"   输入形状: {input_data.shape}")
    print(f"   权重形状: {weight.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出样例:\n{output}")
    
    # 推理模式测试
    print("\n3️⃣ 推理模式测试（无噪声）")
    output_infer = lumina_kernel.optical_linear_infer(
        input_data,
        weight,
        bias,
        bits=8
    )
    
    print(f"   输出形状: {output_infer.shape}")
    print(f"   输出样例:\n{output_infer}")
    
    # 性能测试
    print("\n4️⃣ 性能测试（批量处理）")
    large_input = np.random.randn(32, 128).astype(np.float32)
    large_weight = np.random.randn(64, 128).astype(np.float32)
    
    import time
    start = time.time()
    for _ in range(10):
        _ = lumina_kernel.optical_linear_fused(
            large_input,
            large_weight,
            None,
            noise_std=0.1,
            bits=4,
            seed=42
        )
    elapsed = time.time() - start
    
    print(f"   批次大小: 32")
    print(f"   输入维度: 128 -> 64")
    print(f"   10次迭代耗时: {elapsed*1000:.2f} ms")
    print(f"   平均每次: {elapsed*100:.2f} ms")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n💡 提示: 请先运行 'maturin develop' 构建 Rust 扩展")
    exit(1)
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
