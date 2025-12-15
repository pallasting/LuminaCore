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