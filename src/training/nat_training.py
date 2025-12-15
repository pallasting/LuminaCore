import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OpticalNoiseLayer(nn.Module):
    """
    模拟光子芯片物理噪声的 PyTorch 层
    用于在训练过程中注入噪声 (NAT - Noise Aware Training)
    """
    def __init__(self, noise_level=0.1, drift_level=0.05):
        super(OpticalNoiseLayer, self).__init__()
        self.noise_level = noise_level
        self.drift_level = drift_level

    def forward(self, x):
        if self.training:
            # 1. 模拟光强波动 (Multiplicative Shot Noise)
            # 信号越强，噪声越大 (近似 Poisson 分布特性)
            noise = torch.randn_like(x) * self.noise_level * torch.sqrt(torch.abs(x) + 1e-6)
            
            # 2. 模拟热漂移/串扰 (Additive Drift Noise)
            # 这是一个低频偏置，但在 batch 内部可能表现为随机偏移
            drift = torch.randn_like(x) * self.drift_level
            
            # 3. 模拟链路损耗 (Attenuation) - 假设 20% 损耗
            attenuation = 0.8
            
            return (x * attenuation) + noise + drift
        else:
            # 推理时，我们通常不加噪声（除非是为了测试鲁棒性）
            # 但为了模拟真实芯片推理，我们可以选择开启
            return x * 0.8 

class LuminaNet(nn.Module):
    def __init__(self, use_nat=False):
        super(LuminaNet, self).__init__()
        self.use_nat = use_nat
        
        # 简单的 MLP 结构用于 MNIST
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.noise_layer1 = OpticalNoiseLayer(noise_level=0.15) # 15% 噪声注入
        self.fc2 = nn.Linear(512, 10)
        self.noise_layer2 = OpticalNoiseLayer(noise_level=0.10) # 10% 噪声注入

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        
        if self.use_nat:
            x = self.noise_layer1(x)
            
        x = self.fc2(x)
        
        if self.use_nat:
            x = self.noise_layer2(x)
            
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, noise_injection=False):
    model.eval()
    test_loss = 0
    correct = 0
    
    # 如果需要在测试时注入噪声来验证鲁棒性
    noise_layer = OpticalNoiseLayer(noise_level=0.15) if noise_injection else None
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if noise_injection:
                # 手动在输出层注入噪声模拟芯片读取
                output = noise_layer(output)
                
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set (Noise Injection={noise_injection}): Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')
    return acc

def main():
    # 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 使用 Fake Data 避免下载 (为了演示速度)
    # 实际使用时请取消注释下面的真实数据集
    # dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    
    # 这里我们创建一个简单的随机数据集来验证代码跑通
    print("注意：正在使用随机生成的 Dummy 数据集进行演示...")
    dataset1 = datasets.FakeData(size=1000, image_size=(1, 28, 28), num_classes=10, transform=transform)
    dataset2 = datasets.FakeData(size=200, image_size=(1, 28, 28), num_classes=10, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000, shuffle=False)

    print("--- 实验 1: 普通训练 (Standard Training) ---")
    model_std = LuminaNet(use_nat=False).to(device)
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.001)
    
    for epoch in range(1, 3):
        train(model_std, train_loader, optimizer_std, epoch)
    
    print(">>> 测试普通模型在 15% 光路噪声下的表现:")
    acc_std = test(model_std, test_loader, noise_injection=True)

    print("\n--- 实验 2: 噪声感知训练 (NAT) ---")
    model_nat = LuminaNet(use_nat=True).to(device)
    optimizer_nat = optim.Adam(model_nat.parameters(), lr=0.001)
    
    for epoch in range(1, 3):
        train(model_nat, train_loader, optimizer_nat, epoch)
        
    print(">>> 测试 NAT 模型在 15% 光路噪声下的表现:")
    acc_nat = test(model_nat, test_loader, noise_injection=True)
    
    print(f"结论: NAT 带来了 {acc_nat - acc_std:.2f}% 的准确率提升 (在模拟数据上).")

if __name__ == '__main__':
    main()
