#!/usr/bin/env python
"""
LuminaFlow Logo 生成器

生成一个"光子穿过神经网络节点"的 Logo
设计理念：展示光速计算和 AI 的融合
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def generate_logo(output_path='logo.png', size=(800, 800), dpi=300):
    """
    生成 LuminaFlow Logo
    
    Args:
        output_path: 输出文件路径
        size: 图片尺寸 (width, height)
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 背景渐变（深蓝到紫色，代表光速和科技感）
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 10, 0, 10], aspect='auto', 
              cmap='plasma', alpha=0.3, zorder=0)
    
    # 绘制神经网络节点（3层，每层3-4个节点）
    node_layers = [
        [(2, 2), (2, 4), (2, 6), (2, 8)],  # 输入层
        [(5, 3), (5, 5), (5, 7)],           # 隐藏层
        [(8, 4), (8, 6)]                     # 输出层
    ]
    
    nodes = []
    for layer_idx, layer in enumerate(node_layers):
        for x, y in layer:
            # 节点颜色：从蓝色（输入）到紫色（输出）
            color = plt.cm.plasma(0.3 + layer_idx * 0.3)
            node = Circle((x, y), 0.4, color=color, 
                         ec='white', linewidth=2, zorder=3)
            ax.add_patch(node)
            nodes.append((x, y, color))
    
    # 绘制连接线（神经网络边）
    connections = [
        # 输入层 -> 隐藏层
        (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 1, 2),
        (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 1, 2),
        (0, 2, 1, 0), (0, 2, 1, 1), (0, 2, 1, 2),
        (0, 3, 1, 0), (0, 3, 1, 1), (0, 3, 1, 2),
        # 隐藏层 -> 输出层
        (1, 0, 2, 0), (1, 0, 2, 1),
        (1, 1, 2, 0), (1, 1, 2, 1),
        (1, 2, 2, 0), (1, 2, 2, 1),
    ]
    
    for conn in connections:
        layer1, node1, layer2, node2 = conn
        x1, y1 = node_layers[layer1][node1]
        x2, y2 = node_layers[layer2][node2]
        
        # 连接线（半透明，表示数据流）
        line = plt.Line2D([x1, x2], [y1, y2], 
                         color='white', alpha=0.2, linewidth=0.5, zorder=1)
        ax.add_line(line)
    
    # 绘制光子轨迹（从输入到输出，穿过网络）
    # 使用渐变色表示光子的能量
    photon_paths = [
        # 主要路径：从第一个输入节点到第一个输出节点
        [(1.6, 2), (3, 3.5), (5, 4.5), (7.4, 4)],
        # 次要路径：从最后一个输入节点到最后一个输出节点
        [(1.6, 8), (3, 6.5), (5, 5.5), (7.4, 6)],
    ]
    
    for path in photon_paths:
        # 绘制光子轨迹（带光晕效果）
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 主轨迹（亮色，代表光子）
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', 
                color=plt.cm.plasma(0.8),
                linewidth=3,
                alpha=0.9,
                zorder=4,
                mutation_scale=20
            )
            ax.add_patch(arrow)
            
            # 光晕效果（多层渐变）
            for alpha, width in [(0.3, 5), (0.15, 8)]:
                line = plt.Line2D([x1, x2], [y1, y2],
                                 color=plt.cm.plasma(0.8),
                                 linewidth=width,
                                 alpha=alpha,
                                 zorder=2)
                ax.add_line(line)
    
    # 添加光子粒子（在关键位置）
    photon_positions = [(3, 3.5), (5, 4.5), (3, 6.5), (5, 5.5)]
    for x, y in photon_positions:
        # 光子粒子（发光点）
        photon = Circle((x, y), 0.15, 
                       color='white', 
                       ec=plt.cm.plasma(0.9),
                       linewidth=2,
                       zorder=5)
        ax.add_patch(photon)
        
        # 光晕
        for radius, alpha_val in [(0.25, 0.2), (0.35, 0.1)]:
            halo = Circle((x, y), radius,
                        color=plt.cm.plasma(0.9),
                        alpha=alpha_val,
                        zorder=4)
            ax.add_patch(halo)
    
    # 添加文字 "LuminaFlow"
    ax.text(5, 9.5, 'LuminaFlow', 
           ha='center', va='top',
           fontsize=32, fontweight='bold',
           color='white',
           family='sans-serif',
           zorder=6)
    
    # 添加副标题
    ax.text(5, 8.8, 'Train once, survive the noise',
           ha='center', va='top',
           fontsize=12,
           color='white',
           alpha=0.8,
           family='sans-serif',
           style='italic',
           zorder=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    print(f"✅ Logo 已生成: {output_path}")
    
    # 同时生成一个简化版本（用于小图标）
    generate_simple_logo('logo_simple.png', dpi=dpi)
    
    plt.close()

def generate_simple_logo(output_path='logo_simple.png', dpi=300):
    """
    生成简化版 Logo（用于 favicon 等小尺寸场景）
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('black')
    
    # 简化的神经网络（2层，每层2个节点）
    # 输入层
    node1 = Circle((3, 5), 0.8, color=plt.cm.plasma(0.4), 
                   ec='white', linewidth=2, zorder=3)
    node2 = Circle((3, 7), 0.8, color=plt.cm.plasma(0.4), 
                   ec='white', linewidth=2, zorder=3)
    ax.add_patch(node1)
    ax.add_patch(node2)
    
    # 输出层
    node3 = Circle((7, 5), 0.8, color=plt.cm.plasma(0.8), 
                   ec='white', linewidth=2, zorder=3)
    node4 = Circle((7, 7), 0.8, color=plt.cm.plasma(0.8), 
                   ec='white', linewidth=2, zorder=3)
    ax.add_patch(node3)
    ax.add_patch(node4)
    
    # 连接线
    for y1 in [5, 7]:
        for y2 in [5, 7]:
            line = plt.Line2D([3.8, 6.2], [y1, y2],
                             color='white', alpha=0.3, linewidth=1, zorder=1)
            ax.add_line(line)
    
    # 光子轨迹（从中心穿过）
    arrow = FancyArrowPatch(
        (3.8, 6), (6.2, 6),
        arrowstyle='->',
        color=plt.cm.plasma(0.9),
        linewidth=4,
        alpha=0.9,
        zorder=4,
        mutation_scale=25
    )
    ax.add_patch(arrow)
    
    # 光子粒子
    photon = Circle((5, 6), 0.2,
                   color='white',
                   ec=plt.cm.plasma(0.9),
                   linewidth=2,
                   zorder=5)
    ax.add_patch(photon)
    
    # 光晕
    for radius, alpha_val in [(0.4, 0.15), (0.6, 0.08)]:
        halo = Circle((5, 6), radius,
                     color=plt.cm.plasma(0.9),
                     alpha=alpha_val,
                     zorder=4)
        ax.add_patch(halo)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"✅ 简化版 Logo 已生成: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("🎨 正在生成 LuminaFlow Logo...")
    generate_logo('logo.png', size=(1200, 1200), dpi=300)
    print("\n✨ Logo 生成完成！")
    print("   - logo.png: 完整版 Logo（用于 README、网站等）")
    print("   - logo_simple.png: 简化版 Logo（用于 favicon、小图标等）")

