"""
Robustness Plotting - 鲁棒性可视化

生成抗噪曲线图，展示模型在不同噪声水平下的表现
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def benchmark_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    noise_levels: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    title: str = "Model Robustness Analysis",
) -> Tuple[List[float], List[float]]:
    """
    测试模型在不同噪声水平下的鲁棒性

    自动测试模型在 0% - 30% 噪声下的表现，并生成报表。
    这是 LuminaFlow SDK 的核心可视化功能。

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        noise_levels: 噪声水平列表（默认 [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]）
        device: 计算设备
        save_path: 保存图片的路径（默认 "robustness_report.png"）
        title: 图表标题

    Returns:
        (noise_levels, accuracies) 元组
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    accuracies = []

    print(
        f"开始鲁棒性测试，噪声水平范围: {min(noise_levels):.0%} - {max(noise_levels):.0%}"
    )

    for noise_level in noise_levels:
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # 前向传播
                output = model(data)

                # 注入噪声（模拟真实芯片推理）
                if noise_level > 0:
                    # 在输出层注入噪声
                    noise = torch.randn_like(output) * noise_level * torch.abs(output)
                    output = output + noise

                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        accuracies.append(accuracy)

        print(f"  噪声水平 {noise_level:.0%}: 准确率 {accuracy:.2f}%")

    # 绘制图表
    plot_robustness_curve(noise_levels, accuracies, save_path, title)

    return noise_levels, accuracies


def plot_robustness_curve(
    noise_levels: List[float],
    accuracies: List[float],
    save_path: Optional[str] = None,
    title: str = "Model Robustness Analysis",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    绘制鲁棒性曲线图

    这是 LuminaFlow SDK 的标志性可视化：展示模型在噪声环境下的表现。

    Args:
        noise_levels: 噪声水平列表
        accuracies: 对应的准确率列表
        save_path: 保存路径（默认 "robustness_report.png"）
        title: 图表标题
        figsize: 图表尺寸
    """
    if save_path is None:
        save_path = "robustness_report.png"

    plt.figure(figsize=figsize)

    # 绘制曲线
    plt.plot(
        [n * 100 for n in noise_levels],
        accuracies,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Model Accuracy",
        color="#2E86AB",
    )

    # 添加基准线（理想情况：无噪声时的准确率）
    if len(accuracies) > 0:
        baseline = accuracies[0]
        plt.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Baseline ({baseline:.2f}%)",
        )

    # 标记关键点
    for i, (noise, acc) in enumerate(zip(noise_levels, accuracies)):
        if i == 0 or i == len(noise_levels) - 1 or acc < baseline * 0.9:
            plt.annotate(
                f"{acc:.1f}%",
                xy=(noise * 100, acc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    plt.xlabel("Noise Level (%)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(noise_levels) * 100 * 1.1)
    plt.ylim(0, 105)

    # 添加说明文字
    plt.text(
        0.02,
        0.98,
        "LuminaFlow SDK v0.1\nTrain once, survive the noise.",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n鲁棒性报告已保存至: {save_path}")

    # 可选：显示图表
    # plt.show()
