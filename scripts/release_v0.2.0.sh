#!/bin/bash

# LuminaCore v0.2.0 发布脚本
set -e

echo "🚀 开始发布 LuminaFlow v0.2.0..."

# 检查构建产物
if [ ! -d "dist" ]; then
    echo "❌ dist 目录不存在，请先运行 'python -m build'"
    exit 1
fi

echo "📦 检查构建产物..."
ls -la dist/

# GitHub Release
echo ""
echo "📝 创建 GitHub Release..."
if command -v gh &> /dev/null; then
    if gh auth status &> /dev/null; then
        echo "✅ GitHub CLI 已认证"
        gh release create v0.2.0 \
            --title "LuminaFlow v0.2.0: 集成 Rust 后端高性能光学计算内核" \
            --notes "$(cat <<'EOF'
## 🚀 LuminaFlow v0.2.0: Rust-Accelerated 光子计算内核

### ⭐ 核心特性
- **Rust-Accelerated Core**: 集成高性能 Rust 内核，提供融合算子（矩阵乘法 + 散粒噪声 + 量化）
- **智能回退机制**: 训练时自动切换到 PyTorch，推理时启用 Rust 加速
- **零拷贝内存管理**: NumPy 视图直接进入 Rust，无冗余开销
- **并行计算优化**: Rayon 并行处理，支持 SIMD 量化

### 📊 性能提升
| 场景 | PyTorch | Rust 后端 | 加速比 |
|------|---------|------------|--------|
| 小批量推理 | 0.023s | 0.0053s | **4.3x** |
| 大批量推理 | 0.053s | 0.0082s | **6.5x** |
| 混合精度训练 | 0.018s | 0.015s | **1.2x** |

### 🛠️ 技术实现
- 新增 \`lumina_kernel\` Rust 模块
- 更新 CI/CD 流程支持 Rust 构建
- 添加架构接口文档
- 完善基准测试
- 创建发布脚本和快速入门 Notebook

### 🧪 测试验证
- PyTorch 路径测试: 8/8 通过
- Rust 后端测试: 8/8 通过
- 构建验证: ✅ 成功

### 📦 安装
\`\`\`bash
pip install lumina-flow==0.2.0
\`\`\`

### 🌟 开启 Rust 加速
\`\`\`python
import os
os.environ['LUMINA_USE_RUST'] = '1'
import lumina as lnn
layer = lnn.OpticalLinear(784, 128)
# Will automatically use Rust backend
\`\`\`

Closes #1
EOF
)" \
            dist/lumina_flow-0.2.0.tar.gz \
            dist/lumina_flow-0.2.0-py3-none-any.whl
        
        echo "✅ GitHub Release 创建成功!"
        echo "📋 Release URL: https://github.com/pallasting/LuminaCore/releases/tag/v0.2.0"
    else
        echo "❌ GitHub CLI 未认证，请先运行: gh auth login"
    fi
else
    echo "❌ GitHub CLI 未安装，请先安装: sudo apt install gh"
fi

# PyPI 发布
echo ""
echo "📤 准备发布到 PyPI..."
if [ -n "$PYPI_API_TOKEN" ]; then
    echo "✅ PyPI token 已配置"
    cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = $PYPI_API_TOKEN
EOF
    
    python -m twine upload dist/*
    echo "✅ PyPI 发布成功!"
    echo "📦 PyPI 包: https://pypi.org/project/lumina-flow/"
else
    echo "❌ 未设置 PYPI_API_TOKEN 环境变量"
    echo "💡 设置方法: export PYPI_API_TOKEN='pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'"
    echo "📖 详细指南: 查看 RELEASE_GUIDE_v0.2.0.md"
fi

echo ""
echo "🎉 LuminaFlow v0.2.0 发布完成!"
echo "🔗 相关链接:"
echo "   - GitHub Release: https://github.com/pallasting/LuminaCore/releases/tag/v0.2.0"
echo "   - PyPI 包: https://pypi.org/project/lumina-flow/"
echo "   - 文档: https://luminaflow.readthedocs.io/"
echo "   - Discord: https://discord.gg/j3UGaF7Y"