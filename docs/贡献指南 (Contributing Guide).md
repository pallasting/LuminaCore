# 贡献指南 (Contributing Guide)

## 🌟 欢迎加入 LuminaFlow 社区

**我们正在构建光子计算的未来，每一份贡献都至关重要。**

无论你是经验丰富的开发者、研究者，还是对新技术充满热情的学习者，这里都有适合你的位置。

---

## 🚀 核心贡献者招募令 (Bounty Program)

### 🎯 为什么加入我们？

**我们不是在"改进"现有技术，而是在"创造"新 paradigm。**

- **颠覆性创新**：光子计算将重塑整个AI生态
- **历史机遇**：成为计算革命的见证者和创造者
- **实际影响**：你的代码将影响数亿设备和应用
- **学习机会**：接触最前沿的光子物理和神经形态计算

### 💰 核心贡献者奖励计划

我们为以下关键领域提供**优先权益**和**实质奖励**：

#### 👨‍🔬 光学物理专家 (最高优先级)

**当前需求**：3-5名具备FDTD仿真或稀土材料背景的专家

**任务目标**：

- 改进噪声模型的物理准确性
- 优化WDM通道串扰补偿算法
- 开发温度漂移预测模型

**奖励权益**：

- 🥇 **LuminaCore首发硬件优先测试权** (价值$10,000+)
- 📝 **联合论文发表机会** (Nature/ Science子刊)
- 💼 **技术合伙人机会** (股权激励)
- 🎓 **访问顶级实验室资源** (MIT/NUS等)

#### 👨‍💻 编译器工程师 (核心技术)

**当前需求**：2-3名具备LLVM/MLIR经验的工程师

**任务目标**：

- 实现PyTorch到光子指令的自动转换
- 开发光子计算的中间表示(IR)
- 构建跨平台编译优化器

**奖励权益**：

- 🥈 **核心技术合伙人** (股权 + 技术决策权)
- 💰 **专项研发经费** ($50,000/年)
- 🏆 **技术发明人署名** (专利申请)
- 🌐 **国际会议演讲机会**

#### 🤖 机器学习研究者 (算法创新)

**当前需求**：5-8名具备Transformer/扩散模型经验的研究者

**任务目标**：

- 开发光子加速的Transformer模型
- 实现光子友好的注意力机制优化
- 创建光子计算的模型压缩算法

**奖励权益**：

- 📊 **联合论文发表** (ICML/NeurIPS等顶会)
- 💡 **算法发明人署名**
- 🔬 **访问学术计算资源**
- 🎖️ **最佳贡献者奖金** ($5,000-10,000)

#### 🎨 前端开发者 (用户体验)

**当前需求**：2-3名具备React/Three.js经验的开发者

**任务目标**：

- 构建光子计算可视化平台
- 开发交互式硬件模拟器
- 创建教育性演示应用

**奖励权益**：

- 🎨 **产品设计决策权**
- 💼 **UI/UX合伙人机会**
- 🌟 **开源项目明星地位**

---

## 📋 贡献工作流

### 快速开始

1. **Fork** 本仓库
2. **创建功能分支**：`git checkout -b feature/amazing-feature`
3. **提交更改**：`git commit -m 'Add amazing feature'`
4. **推送分支**：`git push origin feature/amazing-feature`
5. **创建 Pull Request**

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-repo/lumina-flow.git
cd lumina-flow

# 安装依赖
pip install -e .[dev]

# 运行测试
make test

# 代码质量检查
make quality
```

### 代码规范

- **Python**: 遵循 Black + isort 格式化
- **类型注解**: 必须为所有函数添加类型提示
- **文档**: 所有公共API需要 docstring
- **测试**: 新功能需要对应的单元测试

---

## 🎯 贡献类型

### 💻 代码贡献

- **核心算法**：改进光子计算内核
- **API设计**：扩展Python接口
- **性能优化**：Rust加速和内存管理
- **测试完善**：增加测试覆盖率

### 📚 文档贡献

- **教程编写**：创建使用指南
- **API文档**：完善函数文档
- **示例代码**：提供最佳实践
- **翻译工作**：多语言支持

### 🐛 问题报告

- **Bug报告**：使用 [Issue模板](.github/ISSUE_TEMPLATE/bug_report.md)
- **功能请求**：描述具体需求和使用场景
- **性能问题**：提供基准测试数据

### 🤝 社区贡献

- **问题解答**：帮助其他开发者
- **代码审查**：Review Pull Requests
- **推广传播**：分享项目到相关社区

---

## 🔧 开发指南

### 项目结构

```
lumina-flow/
├── lumina/           # Python包
│   ├── layers/       # 光子层实现
│   ├── optim/        # 优化器
│   ├── physics/      # 物理建模
│   └── core/         # 核心组件
├── src/              # Rust源码 (未来)
├── tests/            # 测试套件
├── docs/             # 文档
└── examples/         # 示例代码
```

### 关键文件

- `lumina/layers/optical_linear.py` - 核心光子层
- `lumina/optim/nat_trainer.py` - 噪声感知训练
- `tests/test_optical_linear.py` - 核心测试
- `pyproject.toml` - 项目配置

### 测试要求

```bash
# 运行所有测试
make test

# 运行特定测试
python -m pytest tests/test_optical_linear.py::TestOpticalLinear::test_forward_pass

# 性能基准测试
python -m pytest tests/ -k "performance" --benchmark-only
```

---

## 🎖️ 贡献者认可

### 贡献者墙 (Hall of Fame)

我们将定期更新贡献者名单，认可每一位贡献者的价值：

- **核心贡献者**: 获得特殊徽章和优先支持
- **活跃贡献者**: 月度认可和社区投票权
- **新手之友**: 帮助新贡献者的导师角色

### 透明度承诺

- **公开路线图**: 季度更新项目进展
- **财务透明**: 定期公布资金使用情况
- **决策民主**: 重要决策通过社区投票

---

## 📞 联系我们

### 即时沟通

- **GitHub Discussions**: 技术讨论和问题解答
- **Discord**: [lumina-flow/community](https://discord.gg/lumina-flow)
- **邮件**: <contributors@luminacore.ai>

### 定期活动

- **周会**: 每周三 20:00 (北京时间)
- **黑客松**: 每月第二个周末
- **线上分享**: 技术分享和最佳实践

---

## 📜 行为准则

### 我们的承诺

LuminaFlow 致力于提供一个开放、包容和尊重的社区环境。

### 不可接受的行为

- 歧视性语言或行为
- 骚扰或威胁
- 破坏性批评
- 知识产权侵权

### 举报机制

如遇到违规行为，请通过以下方式举报：

- 发送邮件至: <conduct@luminacore.ai>
- 在GitHub Issue中标记管理员

---

## 🙏 致谢

**每一个贡献，无论大小，都在推动光子计算的边界。**

特别感谢所有早期支持者和贡献者，你们的热情和专业精神让我们相信：**光子计算的未来，从这里开始。**

---

<div align="center">

**🚀 准备好改变计算的历史了吗？**

[开始贡献](https://github.com/your-repo/lumina-flow/fork) | [查看任务](https://github.com/your-repo/lumina-flow/issues) | [加入讨论](https://github.com/your-repo/lumina-flow/discussions)

</div>
