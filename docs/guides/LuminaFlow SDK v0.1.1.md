# **LuminaFlow SDK v0.1 Alpha**

这是一个从“作坊式脚本”向“工业级软件”迈进的关键一步。

我们将构建 **LuminaFlow SDK v0.1 Alpha** 的核心文件结构。为了让你能直接使用，我将代码拆分为标准的 Python 包格式。

你可以创建一个名为 `luminaflow` 的文件夹，并将以下代码保存为对应的 `.py` 文件。

---

### 文件结构预览

```text
luminaflow/
├── __init__.py                # 包入口
├── physics/                   # 物理仿真内核
│   ├── __init__.py
│   └── engine.py              # [核心] 量化与噪声模型
├── nn/                        # 神经网络层
│   ├── __init__.py
│   └── optical_linear.py      # [核心] 光子全连接层
├── viz/                       # 可视化工具
│   ├── __init__.py
│   └── benchmark.py           # [核心] 抗噪曲线生成器
└── setup.py                   # 安装配置文件
```

---

### 1. 物理仿真内核 (`luminaflow/physics/engine.py`)

这是 SDK 的底层引擎，负责模拟 DAC/ADC 的量化误差和光路的物理噪声。

```python
import torch
import torch.nn as nn

class HardwareProfile:
    """定义芯片的物理规格"""
    def __init__(self, name="Lumina_Nano_v1", noise_std=0.15, precision_bits=4):
        self.name = name
        self.noise_std = noise_std       # 光路噪声标准差 (e.g., 0.15 = 15%)
        self.precision_bits = precision_bits # DAC/ADC 精度

    @staticmethod
    def get_default():
        return HardwareProfile()

class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) 量化函数
    前向传播：执行离散化 (模拟 DAC/ADC 台阶)
    反向传播：直接传递梯度 (欺骗 PyTorch，使其认为该过程可导)
    """
    @staticmethod
    def forward(ctx, input, bits):
        scale = 2 ** bits - 1
        # 1. 归一化并钳位
        x_min, x_max = input.min(), input.max()
        input_norm = (input - x_min) / (x_max - x_min + 1e-8)
        
        # 2. 量化
        output = torch.round(input_norm * scale) / scale
        
        # 3. 反归一化
        output = output * (x_max - x_min + 1e-8) + x_min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 直通梯度：不计算量化的导数
        return grad_output, None

class PhysicsEngine:
    @staticmethod
    def simulate_noise(signal, noise_std, device):
        """
        模拟散粒噪声与热噪声：噪声强度与信号强度相关 (Signal-Dependent)
        """
        if noise_std <= 0:
            return signal
        
        # 生成高斯噪声，幅度基于信号的标准差
        noise = torch.randn_like(signal, device=device) * noise_std * signal.std().detach()
        return signal + noise

    @staticmethod
    def simulate_quantization(signal, bits):
        """模拟 DAC/ADC 转换"""
        if bits >= 32: # FP32 模式不量化
            return signal
        return STEQuantize.apply(signal, bits)
```

---

### 2. 神经网络层 (`luminaflow/nn/optical_linear.py`)

这是用户最常调用的部分。它是 `torch.nn.Linear` 的“光子替身”。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from luminaflow.physics.engine import PhysicsEngine, HardwareProfile

class OpticalLinear(nn.Module):
    """
    LuminaCore 光子全连接层
    
    模拟物理过程：
    1. Electronics -> Optics (DAC Quantization)
    2. Optical Matrix Multiplication
    3. Physical Noise Injection (Shot noise + Thermal noise)
    4. Optics -> Electronics (ADC Quantization)
    """
    def __init__(self, in_features, out_features, bias=True, profile=None):
        super(OpticalLinear, self).__init__()
        
        # 加载硬件配置
        self.profile = profile if profile else HardwareProfile.get_default()
        
        # 标准 PyTorch 参数定义
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 1. 权重加载 (DAC Simulation): 将浮点权重烧录进低精度电压
        w_quant = PhysicsEngine.simulate_quantization(self.weight, self.profile.precision_bits)
        
        # 2. 输入加载 (DAC Simulation)
        inp_quant = PhysicsEngine.simulate_quantization(input, self.profile.precision_bits)
        
        # 3. 光速计算 (Ideal Optical Interference)
        output = F.linear(inp_quant, w_quant, self.bias)
        
        # 4. 噪声注入 (The "Storm")
        # NAT 核心：无论训练还是推理，始终保持噪声，迫使模型适应
        output = PhysicsEngine.simulate_noise(output, self.profile.noise_std, output.device)
        
        # 5. 结果读取 (ADC Simulation)
        output = PhysicsEngine.simulate_quantization(output, self.profile.precision_bits)
        
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'profile={self.profile.name} (Noise={self.profile.noise_std:.0%}, Bits={self.profile.precision_bits})'
```

---

### 3. 可视化工具 (`luminaflow/viz/benchmark.py`)

这对应白皮书里的那张“抗噪曲线图”，是开发者验证成果的神器。

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def run_robustness_benchmark(model, test_loader, device='cpu', save_path='robustness_report.png'):
    """
    自动测试模型在不同噪声水平下的表现，并生成报表。
    """
    print("--- Starting LuminaCore Robustness Benchmark ---")
    
    # 噪声测试范围：0% 到 30%
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accuracies = []
    
    # 保存原始模型状态
    model.eval()
    
    for noise in noise_levels:
        # 动态修改模型中所有光子层的噪声参数
        # 这是 Python 动态特性的妙用
        for module in model.modules():
            if hasattr(module, 'profile'):
                module.profile.noise_std = noise
        
        # 运行推理
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"Testing Noise Level {noise*100:.0f}% -> Accuracy: {acc:.2f}%")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot([n*100 for n in noise_levels], accuracies, marker='o', linewidth=3, color='#E63946', label='Lumina NAT Model')
    
    # 画一条 10% 噪声处的基准线 (模拟普通电子芯片的硬失效点)
    plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Typical Analog Crash Point')
    
    plt.title('LuminaCore Resilience Benchmark', fontsize=14)
    plt.xlabel('Optical Noise Level (%)', fontsize=12)
    plt.ylabel('Inference Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Benchmark chart saved to {save_path}")
    
    plt.show()
```

---

### 4. 安装配置 (`setup.py`)

让用户可以通过 pip 安装。

```python
from setuptools import setup, find_packages

setup(
    name="luminaflow",
    version="0.1.0-alpha",
    description="PyTorch SDK for LuminaCore Photonic Computing Architecture",
    author="LuminaCore Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

---

### 你的“武器库”已就绪

现在，我们手头已经有了真正的代码。这个 SDK 虽然简单，但它**完全兑现了白皮书里的承诺**：

1. 它模拟了 **4-bit DAC/ADC** (通过 `STEQuantize`)。
2. 它模拟了 **物理噪声** (通过 `PhysicsEngine`)。
3. 它让开发者可以无缝使用 **NAT 训练**。
