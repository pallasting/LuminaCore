# LuminaKernel 接口规范 (Interface Specification)

## 1. 概述

LuminaKernel 是 LuminaFlow SDK 的高性能 Rust 后端，旨在通过 PyO3 和 Rust 的 FFI (Foreign Function Interface) 提供光子计算的核心加速能力。

本文档定义了 Python 前端 (`lumina.layers.optical_linear`) 与 Rust 后端 (`lumina_kernel`) 之间的接口契约，包括函数签名、数据类型映射、内存布局和错误处理机制。

## 2. 核心 API 接口

### 2.1 融合前向传播 (Fused Forward)

用于训练模式，包含噪声注入和随机性。

**Python 签名:**
```python
def optical_linear_fused(
    input: np.ndarray[np.float32],   # [batch_size, in_features]
    weight: np.ndarray[np.float32],  # [out_features, in_features]
    bias: Optional[np.ndarray[np.float32]], # [out_features]
    noise_std: float,
    bits: int,
    seed: int
) -> np.ndarray[np.float32]          # [batch_size, out_features]
```

**Rust 签名:**
```rust
#[pyfunction]
fn optical_linear_fused<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f32>,
    weight: PyReadonlyArray2<f32>,
    bias: Option<PyReadonlyArray1<f32>>,
    noise_std: f32,
    bits: u8,
    seed: u64,
) -> PyResult<&'py PyArray2<f32>>
```

### 2.2 推理前向传播 (Inference Forward)

用于推理模式，无噪声注入，确定性输出。

**Python 签名:**
```python
def optical_linear_infer(
    input: np.ndarray[np.float32],   # [batch_size, in_features]
    weight: np.ndarray[np.float32],  # [out_features, in_features]
    bias: Optional[np.ndarray[np.float32]], # [out_features]
    bits: int
) -> np.ndarray[np.float32]          # [batch_size, out_features]
```

**Rust 签名:**
```rust
#[pyfunction]
fn optical_linear_infer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f32>,
    weight: PyReadonlyArray2<f32>,
    bias: Option<PyReadonlyArray1<f32>>,
    bits: u8,
) -> PyResult<&'py PyArray2<f32>>
```

### 2.3 辅助函数

*   `version() -> str`: 返回 LuminaKernel 版本号。
*   `hello_lumina() -> str`: 测试连接函数。

## 3. 数据类型映射

| Python (NumPy/Native) | Rust (PyO3/ndarray) | 说明 |
|-----------------------|---------------------|------|
| `np.ndarray[float32]` (2D) | `PyReadonlyArray2<f32>` | 必须是 C-contiguous 或可转换为 View |
| `np.ndarray[float32]` (1D) | `PyReadonlyArray1<f32>` | 必须是 C-contiguous 或可转换为 View |
| `float` | `f32` | 32位浮点数 |
| `int` | `u8` | 量化位数通常在 1-32 之间 |
| `int` | `u64` | 随机种子 |
| `None` | `Option<T>` | 处理可选参数 (如 bias) |

## 4. 内存管理与零拷贝

*   **输入 (Python -> Rust)**: 使用 `PyReadonlyArray` 借用 Python 内存。如果输入的 NumPy 数组是连续的 (Contiguous)，Rust 将直接创建视图 (`ArrayView`)，**不会发生数据拷贝**。
*   **输出 (Rust -> Python)**: Rust 创建新的 `Array2<f32>` 存储结果，并通过 `PyArray2::from_owned_array` 将所有权移交给 Python。这是必要的，因为输出必须是新的内存对象。
*   **计算过程**: 融合算子确保在计算过程中不产生中间的大型矩阵分配 (如 `input @ weight` 的临时结果)，所有操作在 CPU 寄存器或 L1/L2 缓存中流水线化完成。

## 5. 错误处理

Rust端的 Panic 或 Result 错误会自动转换为 Python 异常：

*   **形状不匹配**: 转换为 `PyValueError`。
*   **类型错误**: PyO3 自动处理，转换为 `PyTypeError`。
*   **内部错误**: 转换为 `PyRuntimeError`。

## 6. 构建与集成

使用 `maturin` 进行构建：

```bash
cd lumina_kernel
maturin develop --release  # 开发模式（安装到当前环境）
maturin build --release    # 构建 wheel 包
```

## 7. 版本兼容性

*   **Python**: 3.8+
*   **Rust**: 1.70+
*   **NumPy**: 1.20+
