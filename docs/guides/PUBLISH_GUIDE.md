# LuminaFlow SDK 发布指南

本指南说明如何将 LuminaFlow SDK 发布到 PyPI。

## 📋 发布前检查清单

- [x] 代码已完成并通过测试
- [x] 文档完整（README.md, Getting_Started.ipynb）
- [x] LICENSE 文件已添加
- [x] pyproject.toml 配置正确
- [x] 本地构建和安装测试通过
- [x] twine check 通过

## 🚀 发布步骤

### 1. 准备构建

确保你在项目根目录，并激活虚拟环境：

```bash
cd /path/to/RainbowLuminaCore
source venv/bin/activate  # 或 Windows: venv\Scripts\activate
```

### 2. 清理旧的构建文件（可选）

```bash
rm -rf dist/ build/ *.egg-info
```

### 3. 构建包

```bash
python -m build
```

这将生成：
- `dist/lumina_flow-0.1.0a0.tar.gz` (源码包)
- `dist/lumina_flow-0.1.0a0-py3-none-any.whl` (wheel 包)

### 4. 检查构建产物

```bash
twine check dist/*
```

应该看到：
```
Checking dist/lumina_flow-0.1.0a0-py3-none-any.whl: PASSED
Checking dist/lumina_flow-0.1.0a0.tar.gz: PASSED
```

### 5. 测试本地安装（可选但推荐）

```bash
pip install --force-reinstall dist/lumina_flow-0.1.0a0-py3-none-any.whl
python -c "import lumina; print(lumina.__version__)"
```

### 6. 上传到 TestPyPI（测试）

**首次发布前，强烈建议先上传到 TestPyPI 进行测试！**

#### 6.1 注册 TestPyPI 账号

访问 https://test.pypi.org/account/register/ 注册账号

#### 6.2 创建 API Token

1. 登录 TestPyPI
2. 进入 Account settings → API tokens
3. 创建新的 API token，scope 选择 "Entire account"
4. 复制 token（格式：`pypi-xxxxx`）

#### 6.3 配置认证（方法一：使用 token）

创建或编辑 `~/.pypirc`：

```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-你的token
```

#### 6.4 上传到 TestPyPI

```bash
twine upload --repository testpypi dist/*
```

#### 6.5 从 TestPyPI 测试安装

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lumina-flow
```

**注意**：由于 TestPyPI 和正式 PyPI 是分离的，如果依赖包（如 torch）在 TestPyPI 上不存在，需要使用 `--extra-index-url` 从正式 PyPI 获取。

### 7. 上传到正式 PyPI

**确认 TestPyPI 测试无误后，再上传到正式 PyPI！**

#### 7.1 注册 PyPI 账号

访问 https://pypi.org/account/register/ 注册账号

#### 7.2 创建 API Token

1. 登录 PyPI
2. 进入 Account settings → API tokens
3. 创建新的 API token
4. 复制 token

#### 7.3 配置认证

编辑 `~/.pypirc`，添加正式 PyPI 配置：

```ini
[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-你的正式token
```

#### 7.4 上传到正式 PyPI

```bash
twine upload dist/*
```

**重要提示**：
- 一旦上传到正式 PyPI，版本号就不能再使用
- 确保版本号正确（当前：0.1.0a0）
- 上传后，包将立即可用：`pip install lumina-flow`

### 8. 验证发布

上传成功后，等待几分钟让 PyPI 索引更新，然后：

```bash
pip install lumina-flow
python -c "import lumina; print(lumina.__version__)"
```

访问 https://pypi.org/project/lumina-flow/ 查看你的包页面。

## 🔄 版本号管理

遵循 [PEP 440](https://peps.python.org/pep-0440/) 版本号规范：

- **Alpha 版本**：`0.1.0a0`, `0.1.0a1`, ...
- **Beta 版本**：`0.1.0b0`, `0.1.0b1`, ...
- **正式版本**：`0.1.0`, `0.1.1`, `0.2.0`, ...

更新版本号：
1. 修改 `pyproject.toml` 中的 `version` 字段
2. 修改 `lumina/__init__.py` 中的 `__version__`
3. 重新构建和上传

## 📝 发布说明（Release Notes）

每次发布时，建议创建 GitHub Release，包含：

- 版本号
- 主要功能更新
- Bug 修复
- 已知问题
- 升级指南

示例：

```markdown
# LuminaFlow SDK v0.1.0 Alpha

## 🎉 首次发布

LuminaFlow SDK 是光子计算时代的 CUDA，让开发者轻松将神经网络"移植"到虚拟的光子芯片上。

## ✨ 主要功能

- **Hardware-Aware Layers**: `OpticalLinear` 层，模拟光子芯片的物理特性
- **Auto-NAT**: 噪声感知训练器，一键开启抗噪训练
- **鲁棒性可视化**: 自动生成抗噪曲线图

## 🚀 快速开始

```bash
pip install lumina-flow
```

## 📚 文档

- [README](README.md)
- [快速入门教程](Getting_Started.ipynb)
```

## ⚠️ 常见问题

### Q: 上传时提示 "File already exists"

A: 该版本已经存在，需要更新版本号。

### Q: TestPyPI 安装失败，提示找不到依赖

A: 使用 `--extra-index-url` 从正式 PyPI 获取依赖：
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lumina-flow
```

### Q: 如何删除已发布的版本？

A: PyPI 不允许删除已发布的版本，只能标记为隐藏。联系 PyPI 管理员或发布新版本。

## 🔗 相关链接

- [PyPI 官方文档](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
- [Twine 文档](https://twine.readthedocs.io/)
- [TestPyPI](https://test.pypi.org/)
- [正式 PyPI](https://pypi.org/)

---

**准备好发布了吗？** 按照上述步骤，你的包就可以被全世界的开发者使用了！🚀

