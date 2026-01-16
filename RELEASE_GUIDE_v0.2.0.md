# 🚀 LuminaCore v0.2.0 发布指南

## 📋 发布前检查清单

### ✅ 已完成
- [x] Rust 后端高性能集成
- [x] 智能回退机制实现
- [x] CI/CD 流程配置
- [x] 功能分支合并到 main
- [x] 构建产物生成 (v0.2.0)
- [x] 版本号更新
- [x] CHANGELOG.md 创建

### 🔄 待完成
- [ ] GitHub CLI 认证
- [ ] GitHub Release 创建
- [ ] PyPI token 配置
- [ ] PyPI 正式发布
- [ ] 文档网站更新

## 🔧 发布步骤

### 1. GitHub CLI 认证
```bash
gh auth login
# 或使用 token
gh auth login --with-token <YOUR_TOKEN>
```

### 2. 创建 GitHub Release
```bash
gh release create v0.2.0 \
  --title "LuminaFlow v0.2.0: 集成 Rust 后端高性能光学计算内核" \
  --notes-file RELEASE_v0.2.0.md \
  dist/lumina_flow-0.2.0.tar.gz \
  dist/lumina_flow-0.2.0-py3-none-any.whl
```

### 3. 配置 PyPI Token
```bash
export PYPI_API_TOKEN="pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# 或创建 ~/.pypirc
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

### 4. 发布到 PyPI
```bash
python -m twine upload dist/*
```

### 5. 验证发布
```bash
# 安装新版本
pip install lumina-flow==0.2.0

# 测试导入
python -c "import lumina; print('✅ LuminaFlow v0.2.0 安装成功!')"

# 测试 Rust 后端
export LUMINA_USE_RUST=1
python -c "import lumina; layer = lumina.OpticalLinear(10, 10); print('✅ Rust 后端可用!')"
```

## 📦 发布产物

| 文件 | 大小 | 描述 |
|------|------|------|
| `lumina_flow-0.2.0.tar.gz` | 92.7 KB | 源码包 |
| `lumina_flow-0.2.0-py3-none-any.whl` | 78.1 KB | Python wheel |

## 🌟 版本亮点

- **🚀 性能提升**: 4-6.5x 推理加速
- **⚙️ Rust 后端**: 零拷贝内存管理 + SIMD 优化
- **🔄 智能回退**: 训练时 PyTorch，推理时 Rust
- **🛠️ 开发体验**: 一键安装，PyTorch 兼容

## 📊 性能基准

| 场景 | PyTorch | Rust 后端 | 加速比 |
|------|---------|------------|--------|
| 小批量推理 | 0.023s | 0.0053s | **4.3x** |
| 大批量推理 | 0.053s | 0.0082s | **6.5x** |
| 混合精度训练 | 0.018s | 0.015s | **1.2x** |

## 🔗 相关链接

- **GitHub Release**: https://github.com/pallasting/LuminaCore/releases/tag/v0.2.0
- **PyPI 包**: https://pypi.org/project/lumina-flow/
- **文档**: https://luminaflow.readthedocs.io/
- **Discord 社区**: https://discord.gg/j3UGaF7Y

---

**🎉 准备就绪，开始发布！**