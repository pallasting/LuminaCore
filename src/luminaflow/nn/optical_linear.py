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