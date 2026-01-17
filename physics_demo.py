#!/usr/bin/env python3
"""
Hardware Simulation Demo

展示增强的物理仿真能力：
1. 热串扰 (Thermal Crosstalk)
2. 光损耗 (Optical Loss)
3. 温度漂移 (Temperature Drift)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import lumina_kernel
from typing import Dict, List, Any

class PhysicsSimulator:
    def __init__(self):
        print("🚀 PhysicsSimulator 初始化")
        
    def run_simulation(
        self, 
        input_data: np.ndarray, 
        weight: np.ndarray,
        params: Dict[str, float]
    ) -> np.ndarray:
        """运行物理仿真"""
        
        # 默认参数
        physics_params = {
            "thermal_crosstalk": 0.01,
            "optical_loss_db": 0.5,
            "temperature": 25.0
        }
        physics_params.update(params)
        
        try:
            output = lumina_kernel.optical_linear_physics(
                input_data,
                weight,
                None, # bias
                physics_params,
                8, # bits
                42 # seed
            )
        except AttributeError:
            print("⚠️  Rust kernel too old, using Python simulation fallback")
            output = self._python_fallback(input_data, weight, physics_params)
            
        return output

    def _python_fallback(self, input_data, weight, params):
        """Python 物理仿真回退"""
        # 1. 热串扰
        crosstalk = params["thermal_crosstalk"]
        if crosstalk > 1e-5:
            # 简单的相邻串扰模拟
            # output[i] = (1-c)*input[i] + c/2*input[i-1] + c/2*input[i+1]
            input_copy = input_data.copy()
            padded = np.pad(input_copy, ((0,0), (1,1)), mode='constant')
            
            # 向量化计算 (假设 axis 1 是通道)
            left = padded[:, :-2]
            center = input_copy
            right = padded[:, 2:]
            
            input_data = (1.0 - crosstalk) * center + (crosstalk / 2.0) * (left + right)
            
        # 2. 光损耗
        loss_db = params["optical_loss_db"]
        if abs(loss_db) > 1e-5:
            attenuation = 10.0 ** (-loss_db / 10.0)
            input_data = input_data * attenuation
            
        # 3. 矩阵乘法
        output = np.dot(input_data, weight.T)
        
        # 4. 热噪声
        temp = params["temperature"]
        noise_scale = 0.001 * (temp / 25.0)
        noise = np.random.normal(0, noise_scale, output.shape)
        output = output + output * noise
        
        return output.astype(np.float32)

def plot_heatmap(data, title, ax):
    im = ax.imshow(data, cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

def main():
    print("=" * 60)
    print("光子芯片物理效应仿真")
    print("=" * 60)
    
    # 1. 准备数据
    N = 32
    input_data = np.eye(N, dtype=np.float32) # 单位矩阵，方便观察串扰
    weight = np.eye(N, dtype=np.float32)     # 直通权重
    
    sim = PhysicsSimulator()
    
    # 2. 场景对比
    scenarios = [
        {
            "name": "理想情况 (Ideal)",
            "params": {"thermal_crosstalk": 0.0, "optical_loss_db": 0.0, "temperature": 25.0}
        },
        {
            "name": "典型工况 (Typical)",
            "params": {"thermal_crosstalk": 0.02, "optical_loss_db": 0.5, "temperature": 45.0}
        },
        {
            "name": "高温恶劣 (Harsh)",
            "params": {"thermal_crosstalk": 0.05, "optical_loss_db": 1.0, "temperature": 85.0}
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n🔬 仿真场景: {scenario['name']}")
        print(f"   参数: {scenario['params']}")
        
        output = sim.run_simulation(input_data, weight, scenario['params'])
        results.append((scenario['name'], output))
        
        # 计算信噪比 (SNR)
        # 信号是对角线元素，噪声是非对角线元素
        signal_power = np.mean(np.diag(output)**2)
        noise_power = np.mean(output[~np.eye(N, dtype=bool)]**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        print(f"   📉 SNR: {snr:.2f} dB")
        
    # 3. 可视化
    print("\n📊 生成可视化图表 (physics_simulation.png)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, output) in enumerate(results):
        plot_heatmap(output, name, axes[i])
        
    plt.tight_layout()
    plt.savefig('physics_simulation.png')
    print("✅ 完成")

if __name__ == "__main__":
    main()
