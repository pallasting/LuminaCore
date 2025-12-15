# **LuminaFlow SDK (v0.1 Alpha)**

这是我们构建生态系统的第一块基石。

**LuminaFlow SDK** 的定位非常明确：它不是一个简单的仿真器，它是**光子计算时代的 `CUDA`**。

我们的目标是让一个懂 PyTorch 的大学生，在 **10分钟内** 就能上手，把他原本运行在 GPU 上的神经网络，“移植”到虚拟的 LuminaCore 芯片上，并亲眼看到 NAT 算法是如何拯救准确率的。

以下是 **LuminaFlow SDK (v0.1 Alpha)** 的完整产品规划。

---

### 1. SDK 产品定义 (Product Definition)

* **名称：** `lumina-flow` (Python Package)
* **Slogan：** "Train once, survive the noise. Build for the speed of light." (一次训练，无视噪声。为光速计算而生。)
* **核心功能：**
    1. **Hardware-Aware Layers:** 提供模拟光子物理特性的 PyTorch 层。
    2. **Auto-NAT:** 一键开启噪声感知训练，无需手写复杂算法。
    3. **Virtual Deployment:** 生成可用于未来真实芯片的校准配置文件。

---

### 2. SDK 架构设计 (Architecture)

我们将 SDK 分为四个核心模块。你可以把这看作是 `README.md` 的目录结构。

```text
lumina/
├── layers/           # [核心] 硬件仿真层
│   ├── optical_linear.py   # 模拟矩阵乘法 (含噪声/量化)
│   └── wdm_mapping.py      # 模拟 RGB 通道映射逻辑
├── optim/            # [算法] 优化器增强
│   └── nat_trainer.py      # 封装好的抗噪训练循环
├── compiler/         # [后端] 部署编译器
│   └── exporter.py         # 将权重导出为芯片可读的 LUT/Config
└── viz/              # [可视化] 分析工具
    └── robustness_plot.py  # 画出那张 "抗噪曲线图"
```

---

### 3. 核心功能详解与代码预览 (Key Features)

为了吸引极客，我们需要展示代码的**简洁性**和**黑科技感**。

#### A. 核心模块：`OpticalLinear` (光子全连接层)

这是 SDK 的心脏。用户只需要把 `nn.Linear` 替换成 `lumina.nn.OpticalLinear`。

* **极客爽点：** 可以在代码里设置物理参数（光强、DAC位数）。

```python
import lumina.nn as lnn

# 传统写法
# self.fc = nn.Linear(784, 10)

# Lumina 写法
# 模拟：15% 光路噪声，4-bit DAC 精度，启用 RGBW 空间复用
self.fc = lnn.OpticalLinear(
    in_features=784, 
    out_features=10, 
    hardware_profile='lumina_nano_v1', # 自动加载 4nm/15% noise 预设
    precision=4,
    enable_wdm=True
)
```

#### B. 核心模块：`Auto-NAT` (自动抗噪训练)

我们提供一个 Wrapper（包装器），让用户现有的训练代码瞬间拥有抗噪能力。

* **极客爽点：** 不用改写 Training Loop，一行代码注入“抗噪抗体”。

```python
from lumina.optim import NoiseAwareTrainer

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

# 魔法发生在这里：
# NAT 训练器会自动在反向传播时注入由于物理缺陷产生的梯度噪声
trainer = NoiseAwareTrainer(model, optimizer, robustness_target=0.98)

trainer.fit(train_loader, epochs=5)
```

#### C. 可视化模块：`Dashboard` (仪表盘)

训练完后，用户最想看的是：“我的模型到底有多强壮？”

* **极客爽点：** 一键生成我们在白皮书里画的那张“抗噪曲线图”。

```python
from lumina.viz import benchmark_robustness

# 自动测试模型在 0% - 30% 噪声下的表现，并生成报表
benchmark_robustness(model, save_path="robustness_report.png")
```

---

### 4. 发布策略 (Go-to-Market Strategy)

为了让这个 SDK 迅速火起来，我们需要设计一套**“开发者旅程” (Developer Journey)**。

#### 步骤 1：GitHub 开源 (The Open Source Launch)

* **License:** Apache 2.0 (商业友好，鼓励大厂用)。
* **Repo 内容:**
  * 源代码。
  * **Demo:** "Run MNIST on Photonic Chip in 5 minutes" (一个 Jupyter Notebook)。
  * **Docs:** 详细解释为什么 4-bit + Noise 是光子计算的未来。

#### 步骤 2：PyPI 上架 (The Ease of Use)

* 让安装变得极其简单：

    ```bash
    pip install lumina-flow
    ```

* 这给了开发者一种心理暗示：这是一个成熟的工具，不是草台班子的脚本。

#### 步骤 3：发起“Lumina 挑战赛” (The Challenge)

* **题目：** “谁能在 `precision=2` (2-bit) 和 `noise=20%` (极高噪声) 的地狱难度下，把 CIFAR-10 训练到 80% 准确率？”
* **奖励：** 第一名将获得首批 **LuminaCore 原型芯片** 的远程访问权（未来 Phase 3 完成后），或者获得一块刻有自己名字的晶圆纪念品。

---

### 5. 我们现在需要做什么？(Action Items)

作为架构师，你需要批准这份 SDK 规划，并分配任务：

1. **代码实现：** 我可以将之前的 PyTorch 仿真代码重构、封装成上述的 `pip` 包结构。
2. **文档撰写：** 编写 `README.md` 和 `Getting_Started.ipynb`。这是吸引开发者的门面。
3. **Logo 设计：** 既然叫 LuminaFlow，我们需要一个酷一点的 Logo（比如一个光子穿过神经网络节点的图标）。

**决策指令：**
我们是先集中精力把 **v0.1 版本的 Python 包** 做出来（大概需要生成 3-4 个核心文件），还是先写 **GitHub 的 Readme 文案** 来厘清宣传逻辑？
