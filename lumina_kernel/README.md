# LuminaKernel - Rust 加速光子计算后端

> **状态**: 🚧 开发中（v0.1.0-alpha）

## 📋 项目简介

LuminaKernel 是 LuminaFlow SDK 的 Rust 加速后端，通过融合算子和 SIMD 优化实现：

- **4-6x 边缘推理加速**（小批量场景）
- **3-4x NAT 训练加速**（减少内存带宽）
- **零拷贝 Python-Rust 互操作**

## 🏗️ 架构设计

### 融合算子（Fused Kernel）

传统 PyTorch 实现需要 3 次内存访问：
```python
y = x @ w          # 1. 矩阵乘法
noise = randn(...)  # 2. 生成噪声
y = y + noise      # 3. 加噪声
y = quantize(y)    # 4. 量化
```

Rust 融合算子一次完成：
```rust
output[i] = quantize((row[i] · col[j]) + fast_rand() * noise_std)
```

## 🚀 快速开始

### 环境要求

- Rust 1.70+
- Python 3.8+
- maturin 1.0+

### 构建安装

```bash
# 安装 maturin
pip install maturin

# 开发模式构建（快速迭代）
maturin develop

# 发布模式构建（性能优化）
maturin develop --release

# 构建 wheel 包
maturin build --release
```

### 测试 FFI

```bash
python test_ffi.py
```

预期输出：
```
✅ 成功导入 lumina_kernel 模块
📦 版本: 0.1.0
👋 Hello from LuminaKernel (Rust Backend)! 🚀

🎉 FFI 测试通过！
```

## 📦 已完成功能

### ✅ 阶段 1: 基础设施（已完成）

- [x] Rust 项目结构初始化
- [x] Cargo.toml 依赖配置
  - PyO3 (Python 绑定)
  - ndarray (多维数组)
  - numpy (NumPy 互操作)
  - rayon (并行计算)
  - rand_xoshiro (快速随机数)
- [x] 基础 Python 绑定
  - `hello_lumina()` - FFI 测试函数
  - `version()` - 版本信息
- [x] Maturin 构建系统配置

## 🔨 开发中功能

### 🚧 阶段 2: 核心算法

- [ ] 并行矩阵乘法（rayon + ndarray）
- [ ] 融合算子实现
  - [ ] 矩阵乘法
  - [ ] 噪声注入（Xoshiro256++）
  - [ ] 量化模拟（位操作）
- [ ] SIMD 优化（AVX2/NEON）

### 📋 阶段 3: 集成与测试

- [ ] Python 层集成（OpticalLinear）
- [ ] 性能基准测试
- [ ] 文档更新

## 🛠️ 开发指南

### 项目结构

```
lumina_kernel/
├── Cargo.toml              # Rust 包配置
├── pyproject.toml          # Python 包配置（maturin）
├── .cargo/
│   └── config.toml         # Cargo 镜像配置
├── src/
│   ├── lib.rs              # 主模块（Python 绑定）
│   ├── compute.rs          # 并行计算核心（待实现）
│   ├── noise.rs            # 快速随机数生成器（待实现）
│   └── quantization.rs     # 量化模拟器（待实现）
├── test_ffi.py             # FFI 测试脚本
└── README.md               # 本文档
```

### 添加新功能

1. 在 `src/` 目录创建新模块
2. 在 `lib.rs` 中导入并暴露给 Python
3. 使用 `#[pyfunction]` 标记导出函数
4. 运行 `maturin develop` 重新构建
5. 在 `test_ffi.py` 中测试

### 性能优化技巧

1. **使用 `#[inline(always)]`** 强制内联关键函数
2. **避免不必要的内存分配** - 使用 `ndarray::ArrayViewMut`
3. **利用 rayon 并行** - `par_axis_iter()` 按行并行
4. **每线程独立 RNG** - 避免锁竞争

## 🐛 已知问题

### 网络问题

如果遇到 Cargo 下载依赖失败，已配置国内镜像（rsproxy.cn）。

如果仍有问题，可以手动配置：
```bash
# 清除代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 重新构建
maturin develop --release
```

## 📊 性能目标

| 场景 | PyTorch (CPU) | LuminaKernel (Rust) | 目标加速比 |
|------|---------------|---------------------|-----------|
| 小批量推理 (batch=1) | 5 ms | 0.8 ms | **~6x** |
| NAT 训练 (batch=32) | 100 ms/iter | 25 ms/iter | **~4x** |
| 内存占用 | High | Low | **显著降低** |

## 📄 许可证

Apache 2.0 - 与 LuminaFlow SDK 主项目保持一致

## 🔗 相关文档

- [LuminaKernel 设计文档](../docs/LuminaKernel_Rust-Accelerated%20Photonic%20Backend.md)
- [LuminaFlow SDK 主项目](../README.md)
- [PyO3 官方文档](https://pyo3.rs/)
- [Maturin 用户指南](https://www.maturin.rs/)

---

**Train once, survive the noise. Build for the speed of light.** ⚡
