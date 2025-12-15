# 贡献指南

感谢你对 LuminaFlow SDK 的兴趣！我们欢迎所有形式的贡献。

## 如何贡献

### 报告问题

如果你发现了 bug 或有功能建议，请：

1. 检查 [Issues](https://github.com/luminaflow/lumina-flow/issues) 中是否已有相关问题
2. 如果没有，请创建一个新的 Issue，包含：
   - 清晰的问题描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息（Python 版本、PyTorch 版本等）

### 提交代码

1. **Fork 仓库**
   ```bash
   git clone https://github.com/your-username/lumina-flow.git
   cd lumina-flow
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **安装开发环境**
   ```bash
   pip install -e ".[dev]"
   ```

4. **编写代码**
   - 遵循 PEP 8 代码风格
   - 添加必要的文档字符串
   - 为新功能编写测试

5. **运行测试**
   ```bash
   python test_lumina.py
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "Add: 描述你的更改"
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 描述你的更改和原因
   - 等待代码审查

## 代码规范

### Python 风格

- 使用 4 个空格缩进
- 遵循 PEP 8
- 使用类型提示（Type Hints）
- 添加文档字符串（Docstrings）

### 提交信息

使用清晰的提交信息：

```
Add: 新功能描述
Fix: 修复的问题描述
Update: 更新的内容描述
Docs: 文档更新
Test: 测试相关
```

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/luminaflow/lumina-flow.git
cd lumina-flow

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 运行测试
python test_lumina.py
```

## 项目结构

```
lumina/
├── layers/          # 硬件感知层
├── optim/           # 优化器增强
├── viz/             # 可视化工具
└── compiler/        # 部署编译器（v0.2）

docs/                # 文档
tests/               # 测试文件
```

## 问题？

如果你有任何问题，请：

- 查看 [文档](README.md)
- 在 [Issues](https://github.com/luminaflow/lumina-flow/issues) 中提问
- 发送邮件到 contact@luminaflow.ai

感谢你的贡献！🎉

