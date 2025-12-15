# IFLOW 项目上下文文档

> 本文档为 iFlow CLI 提供项目上下文，帮助 AI 助手更好地理解和协助开发工作。

---

## 📋 项目概览

**项目名称**: LuminaFlow SDK (RainbowLuminaCore)  
**版本**: v0.1.0-alpha  
**类型**: Python 软件包（SDK）  
**定位**: 光子计算时代的 CUDA - 让 PyTorch 开发者能够在 10 分钟内将神经网络移植到虚拟的 LuminaCore 光子芯片上

### 核心价值主张

LuminaFlow SDK 解决了光子计算的核心挑战：**硬件噪声导致的模型准确率下降**。通过噪声感知训练（NAT），让在 GPU 上训练的模型能够在真实的光子芯片上保持高准确率。

### 技术栈

- **主要语言**: Python 3.8+
- **性能加速**: Rust (LuminaKernel 后端)
- **核心依赖**: PyTorch, NumPy, Matplotlib
- **Rust 依赖**: PyO3, ndarray, rayon, rand_xoshiro
- **前端**: React + TypeScript + Vite（可视化演示）
- **构建工具**: setuptools, pip, maturin
- **代码质量**: black, isort, mypy, flake8, pytest
- **CI/CD**: GitHub Actions

---

## 🏗️ 项目架构

### 目录结构

```
RainbowLuminaCore/
├── lumina/                    # 主 Python 包
│   ├── __init__.py           # 包入口，版本管理
│   ├── exceptions.py         # 自定义异常
│   ├── layers/               # 硬件感知层
│   │   ├── optical_linear.py      # 光子全连接层（核心，支持 Rust 后端）
│   │   ├── wdm_mapping.py         # 波分复用通道映射
│   │   └── optical_components.py  # 硬件组件（量化器、噪声模型）
│   ├── optim/                # 优化器增强
│   │   └── nat_trainer.py         # 噪声感知训练器（核心）
│   ├── viz/                  # 可视化工具
│   │   └── robustness_plot.py     # 鲁棒性曲线图生成
│   ├── compiler/             # 部署编译器（v0.2 计划）
│   ├── physics/              # 物理模型
│   └── optimization/         # 优化算法
│
├── lumina_kernel/             # Rust 加速后端（NEW）
│   ├── src/
│   │   ├── lib.rs                 # Python 绑定入口
│   │   ├── compute.rs             # 并行矩阵乘法
│   │   ├── noise.rs               # 快速随机数生成
│   │   ├── quantization.rs        # 量化模拟器
│   │   └── fused_ops.rs           # 融合算子（核心创新）
│   ├── Cargo.toml                 # Rust 包配置
│   ├── pyproject.toml             # Maturin 构建配置
│   ├── BUILD_GUIDE.md             # 构建指南
│   └── test_ffi.py                # FFI 测试
│
├── frontend/                  # React 前端（可视化演示）
│   ├── App.tsx               # 主应用组件
│   ├── index.tsx             # 入口文件
│   ├── components/           # React 组件
│   ├── services/             # API 服务
│   └── package.json          # npm 配置
│
├── tests/                     # 测试套件
│   ├── test_optical_linear.py
│   ├── test_nat_trainer.py
│   ├── test_viz.py
│   └── ...
│
├── docs/                      # 文档
│   ├── PROJECT_SUMMARY.md    # 项目总结
│   ├── BUILD_SUMMARY.md      # 构建总结
│   ├── architecture/         # 架构文档
│   ├── guides/               # 使用指南
│   └── reports/              # 性能报告
│
├── pyproject.toml            # Python 包配置
├── Makefile                  # 构建命令
├── README.md                 # 项目主文档
├── Getting_Started.ipynb     # 快速入门教程
└── .github/workflows/ci.yml  # CI 配置
```

### 核心模块说明

#### 1. `lumina.layers` - 硬件感知层

**OpticalLinear** (`optical_linear.py`):
- 模拟光子芯片的光学矩阵乘法
- 支持噪声注入（散粒噪声、热噪声、温度漂移）
- 可配置的 DAC/ADC 量化精度（2-bit 到 8-bit）
- 硬件配置预设：`lumina_nano_v1`（边缘端）、`lumina_micro_v1`（数据中心）
- 支持 WDM（波分复用）通道映射

**WDMChannelMapper** (`wdm_mapping.py`):
- 模拟 RGB/RGBW 波长通道映射
- 空间复用优化

#### 2. `lumina.optim` - 优化器增强

**NoiseAwareTrainer** (`nat_trainer.py`):
- 自动在训练阶段注入硬件噪声
- 支持多种噪声调度策略（constant, linear, cosine, exponential）
- 实时鲁棒性监控
- 训练历史记录

#### 3. `lumina.viz` - 可视化工具

**benchmark_robustness** (`robustness_plot.py`):
- 自动测试模型在不同噪声水平下的表现
- 生成抗噪曲线图
- 保存性能报告

#### 4. `lumina_kernel` - Rust 加速后端（NEW）

**核心创新：融合算子**

传统 PyTorch 需要 3 次内存访问：
```python
y = x @ w          # 矩阵乘法
noise = randn(...) # 生成噪声  
y = y + noise      # 加噪声
y = quantize(y)    # 量化
```

Rust 融合算子只需 1 次内存写入：
```rust
output[i] = quantize((row[i] · col[j]) + fast_rand() * noise_std)
```

**性能提升**：
- 小批量推理：4-6x 加速
- NAT 训练：3-4x 加速
- 内存占用：显著降低

**模块组成**：
- `lib.rs` (107 行) - Python FFI 绑定
- `compute.rs` (81 行) - Rayon 并行矩阵乘法
- `noise.rs` (134 行) - Xoshiro256++ 快速随机数
- `quantization.rs` (147 行) - k-bit 量化模拟
- `fused_ops.rs` (210 行) - 融合算子核心

**使用方式**：
```bash
# 构建 Rust 后端
cd lumina_kernel
maturin develop --release

# 启用 Rust 加速
export LUMINA_USE_RUST=1
```

详见 [LuminaKernel 构建指南](lumina_kernel/BUILD_GUIDE.md)

---

## 🚀 开发工作流

### 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd RainbowLuminaCore

# 安装开发依赖
pip install -e ".[dev]"

# 安装前端依赖（如果需要）
cd frontend && npm install
```

### 常用命令

#### Python 开发

```bash
# 安装包（开发模式）
make install
# 或
pip install -e .

# 运行测试
make test
# 或
python -m pytest tests/

# 代码格式化
make format
# 或
black lumina/ tests/
isort lumina/ tests/

# 代码检查
make lint
# 或
flake8 lumina/ tests/

# 类型检查
make type-check
# 或
mypy lumina/ tests/

# 完整质量检查
make quality

# 构建 Python 包
make build-python
# 或
python -m build
```

#### Rust 后端开发

```bash
cd lumina_kernel

# 开发模式构建（快速迭代）
maturin develop

# 发布模式构建（性能优化）
maturin develop --release

# 运行 Rust 单元测试
cargo test

# 运行 FFI 测试
python test_ffi.py

# 构建 wheel 包
maturin build --release

# 性能基准测试
cd ..
python benchmark_rust_vs_pytorch.py
```

#### 前端开发

```bash
cd frontend

# 开发服务器
npm run dev

# 构建生产版本
npm run build
# 或从根目录
make build-frontend

# 预览构建
npm run preview
```

### CI/CD 流程

GitHub Actions 自动执行：
1. 多版本 Python 测试（3.8, 3.9, 3.10, 3.11）
2. 代码覆盖率报告（pytest-cov）
3. 代码质量检查（flake8, mypy）
4. 包构建验证

---

## 🎯 核心概念

### 1. 光子计算的挑战

传统 GPU 训练的模型在光子芯片上部署时面临：
- **光路噪声**: 15-20% 的信号波动
- **量化误差**: DAC/ADC 精度限制（4-8 bit）
- **温度漂移**: 波长失配导致的信号衰减

**结果**: 准确率从 95% 下降到 60%

### 2. 噪声感知训练（NAT）

**核心思想**: 在训练阶段就注入硬件噪声，让模型学会在噪声环境下工作

**实现方式**:
```python
# 标准训练
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

# NAT 训练（一行代码开启）
trainer = NoiseAwareTrainer(model, optimizer, robustness_target=0.98)
trainer.fit(train_loader, epochs=5)
```

**效果**: 在 20% 噪声环境下，准确率从 62% 提升到 91%

### 3. 硬件配置预设

| 配置 | 噪声水平 | 精度 | 温度漂移 | 适用场景 |
|------|---------|------|---------|---------|
| `lumina_nano_v1` | 15% | 4-bit | 5% | 边缘端、低功耗 |
| `lumina_micro_v1` | 10% | 8-bit | 3% | 数据中心、高性能 |

---

## 📝 开发规范

### 代码风格

- **格式化**: Black (line-length=88)
- **导入排序**: isort (black profile)
- **类型提示**: 强制使用类型注解
- **文档字符串**: Google 风格

### 命名约定

- **类名**: PascalCase (例: `OpticalLinear`, `NoiseAwareTrainer`)
- **函数/变量**: snake_case (例: `noise_level`, `fit_epoch`)
- **常量**: UPPER_SNAKE_CASE (例: `HARDWARE_PROFILES`)
- **私有成员**: 前缀 `_` (例: `_apply_noise`)

### 测试要求

- 所有新功能必须有对应的单元测试
- 测试覆盖率目标: >80%
- 使用 pytest 框架
- Mock 外部依赖（如 torch.cuda）

### 提交规范

遵循 Conventional Commits:
```
feat: 添加新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式调整
refactor: 代码重构
test: 测试相关
chore: 构建/工具配置
```

---

## 🔧 常见任务

### 添加新的硬件配置预设

1. 编辑 `lumina/layers/optical_linear.py`
2. 在 `HARDWARE_PROFILES` 字典中添加新配置
3. 更新文档和测试

### 实现新的噪声调度策略

1. 编辑 `lumina/optim/nat_trainer.py`
2. 在 `_get_noise_schedule` 方法中添加新策略
3. 更新类型提示（`Literal`）
4. 添加单元测试

### 添加新的可视化功能

1. 在 `lumina/viz/` 目录创建新模块
2. 实现可视化函数
3. 在 `lumina/viz/__init__.py` 中导出
4. 添加使用示例到 `Getting_Started.ipynb`

---

## 📚 重要文件说明

### 配置文件

- **pyproject.toml**: Python 包的核心配置，包含依赖、构建系统、工具配置
- **Makefile**: 快捷命令定义
- **.flake8**: Flake8 代码检查配置
- **.pre-commit-config.yaml**: Git 预提交钩子

### 文档文件

- **README.md**: 面向用户的项目主文档
- **Getting_Started.ipynb**: 交互式快速入门教程
- **docs/PROJECT_SUMMARY.md**: 项目开发总结
- **RELEASE_CHECKLIST.md**: 发布前检查清单

### 测试文件

- **test_lumina.py**: 基本功能测试
- **test_optical_linear.py**: OpticalLinear 层测试
- **test_nat_trainer.py**: NAT 训练器测试
- **test_viz.py**: 可视化工具测试

---

## 🎓 技术背景

### 光学矩阵乘法原理

```
输入向量 (电压) → DAC → 光强信号 → 光栅路由 → 干涉叠加 → ADC → 输出向量
```

LuminaCore 芯片利用光的干涉叠加原理，在物理层面直接完成矩阵乘法，速度比 GPU 快数个数量级。

### 噪声模型

1. **散粒噪声（Shot Noise）**: `noise ∝ √signal`
2. **热噪声（Thermal Noise）**: 固定底噪，约 0.5%
3. **温度漂移（Thermal Drift）**: 导致信号衰减和串扰

### WDM（波分复用）

使用不同波长的光（如 RGB）在同一物理通道上传输多路信号，实现空间复用，提升计算密度。

---

## 🚦 项目状态

### v0.1 Alpha（当前版本）✅

- ✅ 核心功能实现
  - OpticalLinear 层
  - NoiseAwareTrainer
  - 鲁棒性可视化
- ✅ **Rust 加速后端（NEW）**
  - 融合算子实现（679 行 Rust 代码）
  - Python FFI 绑定
  - 性能基准测试工具
  - 4-6x 推理加速，3-4x 训练加速
- ✅ 文档完善
  - README.md
  - Getting_Started.ipynb
  - LuminaKernel BUILD_GUIDE.md
- ✅ 测试覆盖
  - 基本功能测试
  - CI/CD 配置
  - FFI 测试套件
- ✅ Logo 设计

### v0.2 计划

- 🔄 Rust 后端 SIMD 优化（AVX2/NEON）
- 🔄 部署编译器 (`compiler/exporter.py`)
- 🔄 WDM 通道映射优化
- 🔄 更多硬件配置预设
- 🔄 GPU 后端支持（CUDA/ROCm）

### v0.3 未来

- 📋 支持卷积层 (`OpticalConv2d`)
- 📋 支持注意力机制 (`OpticalAttention`)
- 📋 真实芯片校准工具

---

## 🤝 协作指南

### 问题报告

- 使用 GitHub Issues
- 提供详细的复现步骤
- 附上环境信息（Python 版本、PyTorch 版本）

### 功能请求

- 描述使用场景
- 说明预期行为
- 提供示例代码（如果可能）

### Pull Request

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📞 联系信息

- **GitHub**: https://github.com/luminaflow/lumina-flow
- **Email**: contact@luminaflow.ai
- **文档**: https://luminaflow.readthedocs.io

---

## 🎯 iFlow CLI 使用建议

### 推荐的 AI 助手任务

#### Python 层任务
1. **代码审查**: "请审查 `optical_linear.py` 的噪声注入逻辑"
2. **功能实现**: "实现一个新的噪声调度策略 'adaptive'"
3. **测试编写**: "为 `WDMChannelMapper` 编写单元测试"
4. **文档更新**: "更新 README 中的性能数据表格"
5. **Bug 修复**: "修复 NAT 训练器在 GPU 模式下的内存泄漏"
6. **重构**: "重构 `OpticalLinear` 以支持批量推理优化"

#### Rust 后端任务
1. **性能优化**: "为融合算子添加 SIMD 优化（AVX2）"
2. **功能扩展**: "实现卷积层的 Rust 加速版本"
3. **测试增强**: "为 `fused_ops.rs` 添加更多边界条件测试"
4. **基准测试**: "运行并分析 PyTorch vs Rust 性能对比"
5. **内存优化**: "优化 Rust 后端的内存分配策略"
6. **文档完善**: "为 Rust 模块添加详细的 rustdoc 注释"

### 上下文提示

在与 AI 助手交互时，可以参考：
- 本项目使用 **PyTorch 风格的 API 设计**
- 遵循 **Google Python 风格指南**
- 优先考虑 **向后兼容性**
- 注重 **性能优化**（特别是批量推理场景）
- 保持 **文档与代码同步**

---

**Train once, survive the noise. Build for the speed of light.** ⚡
