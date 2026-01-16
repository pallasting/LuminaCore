# LuminaFlow Core - AI 代理开发指南（AGENTS.md）

本文件为 LuminaCore 的代理执行任务提供统一的开发约束、构建/测试流程以及代码风格规范，便于 agentic 编程代理在仓库内协同工作。

============================

## 1) 构建、Lint 与 测试命令

### Python 核心库

````bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行完整测试集
pytest

# 运行单个测试文件（示例：Lumina Optical Linear）
pytest tests/test_optical_linear.py

# 运行特定测试函数
pytest tests/test_optical_linear.py::test_optical_linear_forward_shape_and_range -v

# 生成覆盖率报告
pytest --cov=tests --cov-report=html

# 类型检查
mypy lumina/

# 代码格式化
black lumina/ tests/
isort lumina/ tests/

# 代码检查
flake8 lumina/ tests/
````

### Rust 内核 (lumina_kernel/)

````bash
cd lumina_kernel

# 确保 Python 3.13 环境 (PyO3 0.23 要求)
# 开发构建（快速编译并安装到当前环境）
maturin develop --release

# 发布构建（优化性能，生成 Wheel）
maturin build --release

# 运行 Rust 单元测试
cargo test

# FFI 连接验证
python ../test_rust_ffi.py
````

### React 前端 (frontend/)

````bash
cd frontend

# 安装依赖
npm install

# 开发服务器
npm run dev

# 生产构建
npm run build

# 预览构建
npm run preview

# TypeScript 类型检查
npx tsc --noEmit
````

---

## 2) 代码风格指南

### 2.1 Python 代码风格

#### 1. 导入顺序与格式
```python
# 标准库导入
from typing import Any, Dict, List, Optional, Tuple, Union
import os

# 第三方库导入
import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal

# 本地导入
from .optical_components import HardwareConfig, Quantizer, NoiseModel
from ..exceptions import InvalidParameterError, ValidationError
```

#### 2. 命名约定
- 类名：PascalCase，例如 OpticalLinear、HardwareConfig
- 函数/变量：snake_case，例如 optical_forward、noise_level
- 常量：UPPER_SNAKE_CASE，例如 MAX_PRECISION_BITS、DEFAULT_NOISE_LEVEL
- 私有成员：前缀 `_`，例如 `_rust_backend`、`_validate_params`

#### 3. 文档字符串格式
```python
class OpticalLinear(nn.Module):
    """
    模拟光子芯片的光学全连接层

    特性：
    - 硬件感知的噪声注入
    - 可配置的量化精度

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        hardware_profile: 硬件配置预设

    Raises:
        InvalidParameterError: 当参数不在有效范围时
        ValidationError: 当配置不兼容时
    """
```
```

#### 4. 类型注解
```python
from typing import Any, Dict, List, Optional, Tuple, Union

def optical_forward(
    self,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """执行光子矩阵乘法"""
    pass
```

#### 5. 错误处理
```python
from ..exceptions import InvalidParameterError, ValidationError

def validate_hardware_profile(profile: str) -> None:
    valid_profiles = ['lumina_nano_v1', 'lumina_micro_v1', 'custom']
    if profile not in valid_profiles:
        raise InvalidParameterError(
            f"Invalid hardware profile '{profile}'. "
            f"Valid options: {valid_profiles}"
        )
```

### 2.2 Rust 代码风格

#### 1. 导入顺序
```rust
// 标准库
use std::collections::HashMap;

// 第三方库
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use ndarray::ArrayView2;

// 本地模块
mod compute;
mod noise;
use crate::compute::optical_matrix_multiply;
```

#### 2. 函数文档
```rust
/// 光子线性层前向传播（融合算子）
///
/// 一次性完成：矩阵乘法 + 噪声注入 + 量化
///
/// # Arguments
/// * `input` - 输入矩阵 [batch_size, in_features]
/// * `weight` - 权重矩阵 [out_features, in_features]
/// * `noise_std` - 噪声标准差
///
/// # Returns
/// 输出矩阵 [batch_size, out_features]
///
/// # Errors
/// 返回 PyError 当形状不匹配时
#[pyfunction]
fn optical_linear_fused<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f32>,
    weight: PyReadonlyArray2<f32>,
    noise_std: f32,
) -> PyResult<&'py PyArray2<f32>> {
    // 实现
    Ok(PyArray2::<f32>::zeros(py, [1, 1]))
}
```

#### 3. 错误处理
```rust
use pyo3::exceptions::PyValueError;

fn validate_matrix_shapes(input: ArrayView2<f32>, weight: ArrayView2<f32>) -> PyResult<()> {
    if input.ncols() != weight.nrows() {
        return Err(PyValueError::new_err(
            format!("Shape mismatch: input.ncols()={} != weight.nrows()={}", 
                    input.ncols(), weight.nrows())
        ));
    }
    Ok(())
}
```

### 2.3 TypeScript/React 代码风格

#### 1. 组件定义
```typescript
import React, { useState, useEffect } from 'react';
import { SystemState, LogEntry } from '../types';

type ControlPanelProps = {
  state: SystemState;
  onStateChange: (newState: SystemState) => void;
};

export const ControlPanel: React.FC<ControlPanelProps> = ({ state, onStateChange }) => {
  // 组件实现
  return null;
};
```

#### 2. 状态管理
```typescript
const [analysis, setAnalysis] = useState<string | null>(null);
const [isAnalyzing, setIsAnalyzing] = useState(false);
const [logs, setLogs] = useState<LogEntry[]>([]);
```

#### 3. 类型定义
```typescript
export interface SystemState {
  v1: number;
  v2: number;
  v3: number;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  type: 'SYS' | 'ISA' | 'WARN' | 'CMD';
  message: string;
}
```

---

## Cursor/Copilot 规则
未在 .cursor/rules/ 或 .cursorrules 发现规则；如发现，请合并到此 AGENTS.md。
未在 .github/copilot-instructions.md 发现规则文件；如后续出现，请补充。

---

## 4) 提交流程与协作
- 提交信息格式：类型(范围): 简短描述
- 示例：
  
```text
feat(lumina_kernel): add WDM multiplexing support
```
- Closes: Closes #issue_number
---

## 5) 测试与质量保障

### Python 测试最佳实践
- 命名清晰，覆盖边界
- 使用 fixtures 提高重复性

### Rust 测试与集成
- cargo test，FFI 测试优先

---

## 6) 关键文件定位
- Python：lumina/, lumina/layers/, tests/
- Rust：lumina_kernel/src/
- 前端：frontend/

---

## 7) 快速参考

```
alias build-dev="cd lumina_kernel && maturin develop"
alias test-py="pytest -v"
```

记住：我们正在构建光子计算的未来！每一行代码都在创造新的计算范式。