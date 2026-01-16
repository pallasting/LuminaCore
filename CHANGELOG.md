# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-16

### 🚀 Added
- **Rust-Accelerated Core**: 集成高性能 Rust 内核，提供融合算子（矩阵乘法 + 散粒噪声 + 量化）
- **智能回退机制**: 训练时自动切换到 PyTorch，推理时启用 Rust 加速
- **零拷贝内存管理**: NumPy 视图直接进入 Rust，无冗余开销
- **并行计算优化**: Rayon 并行处理，支持 SIMD 量化
- **WDM 多路复用支持**: 波分复用技术，突破传统电子瓶颈
- **硬件感知优化**: 自动适配不同计算芯片配置

### 🛠️ Technical
- 新增 `lumina_kernel` Rust 模块
- 更新 CI/CD 流程支持 Rust 构建
- 添加架构接口文档
- 完善基准测试
- 创建发布脚本和快速入门 Notebook

### 📊 Performance
| 场景 | PyTorch | Rust 后端 | 加速比 |
|------|---------|------------|--------|
| 推理 (小批量) | 0.023s | 0.0053s | **4.3x** |
| 推理 (大批量) | 0.053s | 0.0082s | **6.5x** |
| 训练 (混合精度) | 0.018s | 0.015s | **1.2x** |

### 🧪 Testing
- PyTorch 路径测试: 8/8 通过
- Rust 后端测试: 8/8 通过
- 构建验证: ✅ 成功
- 代码质量检查: ⚠️ 部分警告（非阻塞）

### 📦 Build
- 构建产物已生成至 `dist/`
- 支持 Python 3.8+ 和多平台
- 兼容现有 PyTorch 工作流

### 📝 Documentation
- 新增 `docs/architecture/LuminaKernel_Interface_Spec.md`
- 更新 API 文档
- 添加性能基准报告

## [0.1.0a] - 2025-01-15

### 🎯 Initial Release
- 基础光学计算层实现
- PyTorch 兼容接口
- NAT (噪声感知训练) 算法
- 基础硬件模拟器

---

## How to Upgrade from v0.1.0a

1. **Install the new version**:
   ```bash
   pip install lumina-flow==0.2.0
   ```

2. **Enable Rust acceleration** (optional):
   ```python
   import os
   os.environ['LUMINA_USE_RUST'] = '1'
   
   import lumina as lnn
   layer = lnn.OpticalLinear(784, 128)
   # Will automatically use Rust backend if available
   ```

3. **Check performance**:
   ```python
   import time
   x = torch.randn(32, 784)
   
   start = time.time()
   output = layer(x)
   end = time.time()
   
   print(f"Forward pass took: {end - start:.6f}s")
   ```