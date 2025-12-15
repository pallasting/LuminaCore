"""
NoiseAwareTrainer - 噪声感知训练器

封装好的抗噪训练循环，自动在反向传播时注入物理缺陷产生的梯度噪声
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class NoiseAwareTrainer:
    """
    噪声感知训练器（Noise-Aware Training, NAT）

    核心思想：在训练阶段就注入硬件噪声，让模型学会在噪声环境下工作。
    这样训练出的模型在真实的光子芯片上部署时，具有更强的鲁棒性。

    特性：
    - 自动注入梯度噪声
    - 可配置的鲁棒性目标
    - 支持自定义噪声注入策略
    - 训练过程监控和日志
    - 轻量级模式优化（边缘端训练开销减少）

    Args:
        model: PyTorch 模型（应包含 OpticalLinear 层）
        optimizer: 优化器
        criterion: 损失函数（默认 CrossEntropyLoss）
        robustness_target: 目标鲁棒性（0.0-1.0，用于早停和监控）
        noise_schedule: 噪声调度策略 ('constant', 'linear', 'cosine', 'custom')
        max_noise_level: 最大噪声水平（用于噪声调度）
        device: 计算设备
        lightweight_mode: 是否启用轻量级模式（减少计算开销，适合边缘端）
        noise_injection_freq: 噪声注入频率（轻量级模式下每N次batch注入一次噪声）
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        robustness_target: float = 0.98,
        noise_schedule: str = "linear",
        max_noise_level: float = 0.20,
        device: Optional[torch.device] = None,
        lightweight_mode: bool = False,
        noise_injection_freq: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.robustness_target = robustness_target
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.lightweight_mode = lightweight_mode
        self.noise_injection_freq = noise_injection_freq

        self.model.to(self.device)

        # 轻量级模式计数器
        self.batch_counter = 0

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "robustness": [],
        }

        # 当前 epoch
        self.current_epoch = 0

    def inject_gradient_noise(self, epoch: int, total_epochs: int):
        """
        在梯度中注入噪声（模拟硬件缺陷对梯度的影响）

        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        """
        # 轻量级模式：频率降采样噪声注入，减少计算开销
        if self.lightweight_mode:
            self.batch_counter += 1
            if self.batch_counter % self.noise_injection_freq != 0:
                return  # 跳过本次噪声注入

        # 计算当前噪声水平（根据调度策略）
        if self.noise_schedule == "constant":
            noise_level = self.max_noise_level
        elif self.noise_schedule == "linear":
            noise_level = self.max_noise_level * (epoch / total_epochs)
        elif self.noise_schedule == "cosine":
            noise_level = (
                self.max_noise_level * (1 - np.cos(np.pi * epoch / total_epochs)) / 2
            )
        else:
            noise_level = self.max_noise_level

        # 轻量级模式：简化噪声注入策略
        if self.lightweight_mode:
            # 使用更简单的噪声模型，减少计算复杂度
            for param in self.model.parameters():
                if param.grad is not None:
                    # 使用固定大小的噪声张量，减少随机数生成开销
                    noise_scale = noise_level * 0.1  # 减少噪声强度
                    param.grad = param.grad + torch.randn_like(param.grad) * noise_scale
        else:
            # 标准模式：为每个参数注入噪声
            for param in self.model.parameters():
                if param.grad is not None:
                    # 注入与梯度相关的噪声（模拟散粒噪声特性）
                    noise = (
                        torch.randn_like(param.grad)
                        * noise_level
                        * torch.abs(param.grad)
                    )
                    param.grad = param.grad + noise

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int):
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()

            # 注入梯度噪声（NAT 核心）
            self.inject_gradient_noise(epoch, total_epochs)

            # 更新参数
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        self.history["train_loss"].append(epoch_loss)
        self.history["train_acc"].append(epoch_acc)

        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader, inject_noise: bool = False):
        """
        验证模型

        Args:
            val_loader: 验证数据加载器
            inject_noise: 是否在验证时注入噪声（测试鲁棒性）

        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # 可选：在输出层注入噪声（模拟真实芯片推理）
                if inject_noise:
                    noise = torch.randn_like(output) * 0.05  # 5% 输出噪声
                    output = output + noise

                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            epochs: 训练轮数
            val_loader: 验证数据加载器（可选）
            verbose: 是否打印训练信息
        """
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs)

            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader, inject_noise=False)
                robustness = (
                    self.validate(val_loader, inject_noise=True)[1] / val_acc
                    if val_acc > 0
                    else 0.0
                )
                self.history["robustness"].append(robustness)

                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                        f"Robustness: {robustness:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                    )

    def get_history(self) -> Dict[str, list]:
        """获取训练历史"""
        return self.history.copy()
