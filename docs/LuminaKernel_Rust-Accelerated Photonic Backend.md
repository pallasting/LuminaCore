# **LuminaKernel (Rust-Accelerated Photonic Backend)**

这是一份为 AI 开发团队准备的**深度技术架构方案**。

这份方案的目标是将 **LuminaFlow** 的核心计算引擎从纯 Python/PyTorch 迁移到 **Rust**，构建一个名为 **`LuminaKernel`** 的高性能模组。

此方案旨在实现**“双重加速”**：

1. **训练加速：** 将“矩阵乘法+噪声注入+量化”融合为一个算子，大幅提升 NAT 训练效率。
2. **边缘推理加速：** 在不支持 GPU 的边缘设备（如树莓派、AR眼镜芯片）上，利用 Rust 的位操作实现比 PyTorch 快得多的推理速度。

---

# 项目代号：LuminaKernel (Rust-Accelerated Photonic Backend)

**版本：** v1.0 Design Doc
**目标：** 构建高性能、零成本抽象的虚拟光子计算核心
**技术栈：** Rust, PyO3, Rayon, SIMD (AVX2/NEON)

---

## 1. 架构设计图 (Architecture Overview)

我们将采用经典的 **F-F-I (Foreign Function Interface)** 架构。Python 负责灵活性（定义模型结构），Rust 负责计算密集型任务（模拟光子物理特性）。

```mermaid
graph TD
    subgraph "Python Layer (LuminaFlow SDK)"
        PyNet[PyTorch Neural Network]
        AutoGrad[Autograd Engine]
    end

    subgraph "Bridge Layer (PyO3)"
        PyBind[Rust Binding (.so/.pyd)]
        MemShare[Zero-Copy Memory View]
    end

    subgraph "Rust Layer (LuminaKernel)"
        Scheduler[Rayon Parallel Scheduler]
        
        subgraph "Fused Operators (融合算子)"
            OpMatMul[Matrix Multiplication]
            OpNoise[Fast PRNG Noise Injection]
            OpQuant[Bitwise Quantization]
        end
        
        SIMD[SIMD Intrinsics (AVX/Neon)]
    end

    PyNet -->|Input Tensors| PyBind
    PyBind -->|ndarray View| Scheduler
    Scheduler -->|Task Chunks| OpMatMul
    OpMatMul -->|Register Data| OpNoise
    OpNoise -->|Register Data| OpQuant
    OpQuant -->|Result| PyBind
```

---

## 2. 核心技术痛点与 Rust 解决方案

### 痛点 A：Python 中的 NAT 训练极其低效

* **现状：** 在 PyTorch 中，模拟光子计算需要三步：
    1. `y = x @ w` (计算矩阵乘法，写入显存/内存)
    2. `noise = torch.randn(...)` (生成巨大噪声矩阵，写入内存)
    3. `y = y + noise` (读取两个矩阵，相加，写入内存)
* **瓶颈：** **内存带宽 (Memory Bandwidth)**。数据在 CPU/GPU 和内存之间搬运了 3 次。

### 解决方案：Rust 融合算子 (Fused Kernel)

我们利用 Rust 构建一个**“一次成型”**的计算核。数据加载到 CPU 寄存器后，立即完成乘法、加噪和量化，中间不回写内存。

* **逻辑：** `output[i] = Quantize( (Row[i] * Col[j]) + FastRand() )`
* **收益：** 内存访问减少 3 倍，且 Rust 的 `rand_xoshiro` 生成随机数比 Python 快 10 倍以上。

---

## 3. 详细功能模块设计 (Module Specification)

### 3.1 模块结构 (`lib.rs`)

```rust
// 伪代码结构预览
use pyo3::prelude::*;
use ndarray::prelude::*;

/// 核心暴露接口：光子线性层前向传播
#[pyfunction]
fn optical_linear_forward(
    input: PyReadonlyArray2<f32>, 
    weight: PyReadonlyArray2<f32>, 
    bias: Option<PyReadonlyArray1<f32>>,
    noise_std: f32,
    bits: u8
) -> PyResult<Py<PyArray2<f32>>> {
    // 1. 获取数据视图 (Zero-Copy)
    // 2. 调用并行计算核心
    // 3. 返回结果
}
```

### 3.2 并行计算核心 (`compute.rs`)

利用 **Rayon** 库实现多线程并行，利用 **SIMD** 实现单核向量化。

* **任务拆分策略：** 按输出矩阵的行 (Row) 进行切分。
* **噪声生成策略：** 为了速度，不使用加密安全的随机数，而是使用 **Xoshiro256** 算法，每个线程维护一个独立的 RNG 状态，避免锁竞争。

### 3.3 量化模拟器 (`quantization.rs`)

这是模拟光子 DAC/ADC 精度限制的关键。Rust 的位操作性能极高。

* **算法：**

    ```rust
    // Rust 这种底层操作比 Python 快得多
    #[inline(always)]
    fn quantize(val: f32, scale: f32) -> f32 {
        (val * scale).round().clamp(MIN, MAX) / scale
    }
    ```

---

## 4. 开发任务清单 (Task List for AI Dev Team)

请将以下 Prompt 直接发给您的 AI 开发团队（或输入给 Cursor/VSCode Copilot）以生成代码。

### Task 1: 环境配置与基础绑定

* **指令：** "Create a new Rust library project named `lumina_kernel`. Configure `Cargo.toml` with dependencies: `pyo3` (with 'extension-module' feature), `ndarray`, `numpy`, `rayon`, and `rand_xoshiro`. Set up `lib.rs` to expose a basic 'hello world' function to Python."

### Task 2: 实现高性能并行矩阵乘法

* **指令：** "Implement a parallel matrix multiplication function using `ndarray` and `rayon`. The function should take two 2D f32 arrays ($A$ and $B$) and return $C = A \times B$. Use `par_axis_iter` to parallelize over rows of $A$. Ensure no unnecessary memory allocations occur inside the loop."

### Task 3: 实现“噪声注入”与“量化”融合算子

* **指令：** "Enhance the matrix multiplication loop. Inside the inner loop (after the dot product), inject Gaussian noise using `rand_xoshiro`. The noise standard deviation should be signal-dependent (`val * noise_std`). Then, apply simulated quantization to `k` bits. Make sure these operations happen in registers before writing to the output array."

### Task 4: Python 接口封装与发布

* **指令：** "Wrap the core logic with `#[pyfunction]`. Handle `Option<bias>` efficiently. Write a `setup.py` using `maturin` so this Rust project can be installed as a standard Python package via `pip install .`."

---

## 5. 预期性能收益 (Benchmarking Goals)

| 场景 | 纯 PyTorch (CPU) | LuminaKernel (Rust) | 预期加速比 |
| :--- | :--- | :--- | :--- |
| **小批量推理 (Edge)** | 5 ms | 0.8 ms | **~6x** |
| **NAT 训练 (Desktop)** | 100 ms/iter | 25 ms/iter | **~4x** |
| **内存占用** | High (中间变量) | Low (融合计算) | **显著降低** |

---

## 6. 下一步：如何集成？

一旦开发团队交付了 `lumina_kernel` 的 wheel 包，我们只需要修改 `LuminaFlow` SDK 中的一行代码：

**修改前 (`luminaflow/nn/optical_linear.py`):**

```python
# 旧的 PyTorch 实现
output = F.linear(input, weight, bias)
output = output + torch.randn_like(output) * noise
output = quantize(output)
```

**修改后:**

```python
import lumina_kernel

# 新的 Rust 加速实现
# 一行代码完成 乘法+加噪+量化，且利用了 Rust 的多线程
output = lumina_kernel.optical_linear_forward(
    input.numpy(), 
    weight.numpy(), 
    bias.numpy(), 
    self.profile.noise_std, 
    self.profile.precision_bits
)
output = torch.from_numpy(output)
```

---

### 架构师结语

这套方案不仅是**“可行的”**，而且是**“极具竞争力的”**。

通过引入 Rust，我们在纯软件层面上模拟了光子芯片**“极简、极快”**的特性。即便用户没有真实的光子芯片，他们也能在自己的笔记本电脑上体验到“光子级”的推理速度（通过 Rust 模拟的低精度计算优势）。

这对于我们打造 **LuminaFlow 生态** 是一张王牌——我们不仅卖未来的硬件，我们现在的软件就已经比别人的快了。
