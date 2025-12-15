# LuminaKernel 构建与使用指南

## 🚀 快速开始

### 1. 构建 Rust 扩展

```bash
cd lumina_kernel

# 开发模式（快速编译，用于调试）
maturin develop

# 发布模式（优化编译，用于性能测试）
maturin develop --release
```

### 2. 测试 FFI

```bash
python test_ffi.py
```

预期输出：
```
============================================================
LuminaKernel FFI 测试
============================================================

1️⃣ 基础功能测试
✅ 成功导入 lumina_kernel 模块
📦 版本: 0.1.0
👋 Hello from LuminaKernel (Rust Backend)! 🚀

2️⃣ 融合算子测试（训练模式）
   输入形状: (2, 2)
   权重形状: (2, 2)
   输出形状: (2, 2)
   ...

🎉 所有测试通过！
```

### 3. 在 LuminaFlow 中使用

```bash
# 启用 Rust 后端
export LUMINA_USE_RUST=1

# 运行训练脚本
cd ..
python lumina_demo.py
```

## 📦 安装方式

### 方式 1: 开发模式（推荐）

```bash
cd lumina_kernel
maturin develop --release
```

优点：
- 快速迭代
- 直接安装到当前 Python 环境
- 修改代码后重新运行即可

### 方式 2: 构建 Wheel

```bash
cd lumina_kernel
maturin build --release

# 安装生成的 wheel
pip install target/wheels/lumina_kernel-*.whl
```

优点：
- 可分发给其他用户
- 标准 pip 包格式

### 方式 3: 发布到 PyPI（未来）

```bash
maturin publish
```

## 🔧 故障排除

### 问题 1: Cargo 依赖下载失败

**症状**：
```
error: failed to get `ndarray` as a dependency
Could not connect to server
```

**解决方案**：
已配置国内镜像（rsproxy.cn），如果仍有问题：

```bash
# 清除代理
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 重新构建
maturin develop --release
```

### 问题 2: maturin 未安装

**症状**：
```
bash: maturin: command not found
```

**解决方案**：
```bash
pip install maturin
```

### 问题 3: Rust 工具链未安装

**症状**：
```
error: no default toolchain configured
```

**解决方案**：
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 刷新环境
source $HOME/.cargo/env
```

## 🎯 使用方式

### Python 直接调用

```python
import numpy as np
import lumina_kernel

# 准备数据
input_data = np.random.randn(32, 128).astype(np.float32)
weight = np.random.randn(64, 128).astype(np.float32)
bias = np.random.randn(64).astype(np.float32)

# 训练模式（带噪声）
output = lumina_kernel.optical_linear_fused(
    input_data,
    weight,
    bias,
    noise_std=0.1,
    bits=4,
    seed=42
)

# 推理模式（无噪声）
output = lumina_kernel.optical_linear_infer(
    input_data,
    weight,
    bias,
    bits=8
)
```

### 在 LuminaFlow SDK 中使用

```python
import torch
import lumina.nn as lnn
import os

# 启用 Rust 后端
os.environ['LUMINA_USE_RUST'] = '1'

# 创建模型（自动使用 Rust 后端）
model = torch.nn.Sequential(
    lnn.OpticalLinear(784, 512, hardware_profile='lumina_nano_v1'),
    torch.nn.ReLU(),
    lnn.OpticalLinear(512, 10, hardware_profile='lumina_nano_v1'),
)

# 正常训练（Rust 加速自动生效）
optimizer = torch.optim.Adam(model.parameters())
# ... 训练循环
```

### 性能对比

```python
import time
import torch
import numpy as np

# PyTorch 基线
x_torch = torch.randn(32, 128)
w_torch = torch.randn(64, 128)

start = time.time()
for _ in range(100):
    y = torch.nn.functional.linear(x_torch, w_torch)
pytorch_time = time.time() - start

# Rust 加速
x_np = x_torch.numpy()
w_np = w_torch.numpy()

start = time.time()
for _ in range(100):
    y = lumina_kernel.optical_linear_infer(x_np, w_np, None, bits=8)
rust_time = time.time() - start

print(f"PyTorch: {pytorch_time*1000:.2f} ms")
print(f"Rust: {rust_time*1000:.2f} ms")
print(f"加速比: {pytorch_time/rust_time:.2f}x")
```

## 🔍 检查 Rust 后端状态

```python
from lumina.layers.optical_linear import USE_RUST_BACKEND, _RUST_BACKEND_AVAILABLE

print(f"Rust 后端可用: {_RUST_BACKEND_AVAILABLE}")
print(f"Rust 后端已启用: {USE_RUST_BACKEND}")
```

## 📊 性能优化建议

### 1. 使用 Release 模式

```bash
# 开发调试
maturin develop

# 性能测试
maturin develop --release
```

Release 模式启用：
- `-O3` 优化级别
- LTO（链接时优化）
- 单编译单元（更好的内联）

### 2. 批量处理

Rust 后端对批量数据处理效果最佳：

```python
# 好：批量处理
batch_input = np.random.randn(32, 128).astype(np.float32)
output = lumina_kernel.optical_linear_fused(...)

# 不好：逐个处理
for i in range(32):
    single_input = np.random.randn(1, 128).astype(np.float32)
    output = lumina_kernel.optical_linear_fused(...)
```

### 3. 避免频繁的 NumPy-Torch 转换

```python
# 好：在 NumPy 域完成计算
x_np = x_torch.numpy()
y_np = lumina_kernel.optical_linear_fused(...)
y_torch = torch.from_numpy(y_np)

# 不好：频繁转换
for _ in range(100):
    x_np = x_torch.numpy()  # 转换开销
    y_np = lumina_kernel.optical_linear_fused(...)
    y_torch = torch.from_numpy(y_np)  # 转换开销
```

## 🧪 运行测试

```bash
# Rust 单元测试
cd lumina_kernel
cargo test

# Python FFI 测试
python test_ffi.py

# 集成测试
cd ..
python test_lumina.py
```

## 📝 开发工作流

```bash
# 1. 修改 Rust 代码
vim lumina_kernel/src/fused_ops.rs

# 2. 重新构建
cd lumina_kernel
maturin develop --release

# 3. 测试
python test_ffi.py

# 4. 在主项目中测试
cd ..
export LUMINA_USE_RUST=1
python lumina_demo.py
```

## 🎉 成功标志

如果一切正常，你应该看到：

1. **构建成功**：
   ```
   📦 Built wheel for CPython 3.x to ...
   🛠 Installed lumina-kernel-0.1.0
   ```

2. **FFI 测试通过**：
   ```
   🎉 所有测试通过！
   ```

3. **性能提升**：
   ```
   加速比: 4-6x（小批量）
   加速比: 3-4x（训练）
   ```

---

**需要帮助？** 查看 [README.md](README.md) 或提交 Issue
