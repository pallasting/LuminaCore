# LuminaFlow v0.3.0 技术规格书 (Technical Spec)

## 1. 概述 (Overview)
LuminaFlow v0.3.0 的核心目标是实现从 PyTorch 深度学习模型到光子硬件执行指令的**全栈编译路径**。本版本引入了自适应量化、复数域内核支持以及增强的模型导出能力，旨在解决光子计算中特有的硬件非理想性（噪声、串扰、热漂移）问题。

---

## 2. Lumina Compiler 核心组件

### 2.1 HWAQ (Hardware-Wide Adaptive Quantization)
**目标**：根据硬件实时功耗预算和热噪声状态，动态调整权重和激活的量化范围。

- **算法描述**：
  - 引入 `DynamicRangeEstimator`，实时感知 `HardwareConfig` 中的噪声水平。
  - 使用 `Scale-Adjusted Linear Quantization`：$W_q = \text{round}(\frac{W}{\text{scale}} + \text{offset})$。
  - **Scale** 因子与硬件的信噪比 (SNR) 挂钩：$SNR \uparrow \implies bits \uparrow$。
- **接口定义** (`lumina/compiler/quantizer.py`)：
  ```python
  class WeightQuantizer:
      def calibrate(self, weights: torch.Tensor, hardware_state: Dict[str, float]):
          """根据硬件状态计算最佳 scale 和 zero_point"""
          pass
  ```

### 2.2 WDM 智能规划器 (Smart WDM Planner)
**目标**：最大化光纤带宽利用率，同时通过波长间隔优化最小化通道间串扰。

- **核心算法**：
  - **串扰敏感分配 (Crosstalk-Aware Allocation)**：基于物理邻近度和波长重叠模型，避免将高功率信号分配给相邻频率。
  - **自动通道压缩**：在噪声容许范围内，自动压缩 WDM 间隔。
- **接口定义** (`lumina/compiler/planner.py`)：
  ```python
  class WDMPlanner:
      def optimize_allocation(self, model_graph: Any):
          """为执行图中的每个张量流分配波长"""
          pass
  ```

---

## 3. 高性能内核增强 (Rust Kernel++)

### 3.1 复数域原生支持 (Native Complex Support)
**目标**：模拟相干光学计算（Coherent Computing），支持振幅和相位的联合运算。

- **底层实现**：
  - 在 Rust 中使用 `num-complex` 库。
  - 实现 `ComplexMatrixMultiply` 算子，支持 `(A + bi) * (C + di)`。
- **Python 衔接**：
  - 对接 `torch.complex64` 和 `torch.complex128`。
  - 提供 `ComplexOpticalLinear` 层。

### 3.2 Causal Mask 算子融合
**目标**：针对 Transformer 模型，在 Rust 层融合 Causal Mask 与 Softmax，减少访存开销。

---

## 4. 模型导出与部署 (LuminaExporter)

### 4.1 静态执行图 (Photonic Execution Graph - PEG)
**目标**：将 PyTorch 动态图转化为适合光子阵列执行的静态拓扑结构。

- **导出格式**：
  - `.lmn` 格式（基于 Protobuf 或 JSON）。
  - 包含：层拓扑、量化 LUT、WDM 映射表、串扰补偿参数。
- **部署目标**：
  - 支持纯 Rust 运行时，脱离 Python 依赖，实现在嵌入式光子加速器上的秒级启动。

---

## 5. 里程碑与验收标准 (Milestones)

| 阶段 | 交付物 | 验收标准 |
| :--- | :--- | :--- |
| **M1: 编译基石** | `WeightQuantizer` 增强版 | 支持自适应 Scale，量化误差降低 15% |
| **M2: 相干计算** | Rust 复数内核 | 复数矩阵乘法性能优于 PyTorch CPU 3x 以上 |
| **M3: 链路打通** | `LuminaExporter` 1.0 | 成功导出并在纯 Rust 环境下加载执行图 |

---

## 6. 约束与限制
- 初始版本仅支持固定拓扑模型（如 GPT、ResNet）。
- 串扰补偿目前仅支持线性模型。
