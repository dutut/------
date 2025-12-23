# 🚀 BERT中文新闻分类策略消融实验

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.45-green.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🌟 **深度探索BERT在中文新闻分类中的奥秘**：从传统机器学习baseline到前沿的训练策略消融实验，一场关于自然语言处理技术的精彩旅程！

## 📖 项目故事

在这个信息爆炸的时代，中文新闻分类已成为NLP领域的重要挑战。THUCNews数据集作为中文文本分类的经典benchmark，为我们提供了广阔的实验舞台。本项目不仅仅是简单的模型训练，更是一次系统性的探索之旅：

- 🔍 **传统与现代的碰撞**：对比经典机器学习方法与深度学习模型的性能差异
- ⚡ **策略的艺术**：通过精心设计的消融实验，揭示warmup、层冻结、label smoothing等技巧的真正威力
- 📊 **数据的洞察**：深入分析模型在不同类别上的表现，找出改进的方向

## ✨ 核心亮点

- **🎯 双重Baseline**：传统ML + BERT标准配置，确保实验的公平性和可比性
- **🔬 系统消融**：4种训练策略的组合实验，覆盖主流优化技巧
- **📈 智能监控**：基于验证集Macro-F1的早停机制，避免过拟合
- **🎨 可视化魔法**：自动生成混淆矩阵、性能对比图，让结果一目了然
- **🌏 中文友好**：智能字体检测，确保所有图表完美显示中文
- **⚡ 性能优化**：混合精度训练 + 梯度裁剪，高效利用GPU资源

## 🛠️ 快速开始

### 环境要求

| 组件 | 实验环境 | 推荐配置 |
|------|----------|----------|
| Python | 3.10 | 3.10.19 |
| PyTorch | 2.6 | 2.6.0+cu124 |
| CUDA | 12.4 | 12.4 |
| 内存 | 8GB+ | 16GB+ |
| GPU | RTX 3090 | RTX 3060+ (8GB VRAM) |



### 数据准备

项目使用THUCNews数据集，自动下载或使用本地文件：

```bash
# 数据集结构
data/
├── THUCNews-txt/
│   ├── train.txt      # 训练集 (~180k 样本)
│   ├── dev.txt        # 验证集 (~10k 样本)
│   ├── test.txt       # 测试集 (~10k 样本)
│   ├── class.txt      # 类别标签
│   └── embedding_*.npz  # 预训练词向量（可选）
```



## 🎮 使用指南

### 场景1：快速体验传统方法

```bash
python baseline_ml.py
```

**预期输出**：
```
[OK] Matplotlib 使用中文字体: SimHei
[OK] ./data/THUCNews-txt/train.txt 读取成功: 180000 条
[OK] TF-IDF 向量化完成，特征维度: 50000

===== SVM =====
Accuracy: 0.9234
Macro-F1: 0.9218
```

### 场景2：深度探索BERT奥秘

```bash
# 策略消融实验（推荐）
python bert_thucnews_strategy_ablation.py

# 或者运行长度消融
python max_len.py

# 自定义训练
python main_advanced.py
```

**训练过程可视化**：
```
Epoch 1/4 [████████████] 100% | Loss: 0.234 | Dev Macro-F1: 0.9456
Epoch 2/4 [████████████] 100% | Loss: 0.123 | Dev Macro-F1: 0.9567
🎉 最佳模型已保存！Macro-F1: 0.9567
```

## 📁 项目架构详解

```
自然语言项目/
├── 📜 baseline_ml.py          # 传统ML基线 (SVM/LR/NB + TF-IDF)
├── 🤖 bert_thucnews_strategy_ablation.py  # BERT策略消融主脚本
├── ⚙️ main_advanced.py        # BERT微调训练
├── 🔬 max_len.py                 # 长度消融实验
├── 📖 README.md               # 项目文档 
├── 📊 outputs_bert/           # 基础BERT实验结果
│   ├── maxlen_128/           # 128长度实验
│   └── maxlen_256/           # 256长度实验
├── 🎯 outputs_bert_strategy/  # 策略消融结果
│   ├── baseline/             # 标准配置
│   ├── freeze_emb+6layers/   # 冻结策略
│   ├── label_smoothing_0.1/  # 平滑策略
│   └── warmup_0.1/           # 预热策略
├── 📂 data/                   # 数据集目录
├── 📝 logs/                   # 训练日志
└── 🔄 pids/                   # 进程管理
```

### 文件功能一览

| 文件名 | 核心功能 | 特色亮点 |
|--------|----------|----------|
| `baseline_ml.py` | 传统ML基准测试 | 多模型对比，自动评估 |
| `bert_thucnews_strategy_ablation.py` | 策略消融实验 | 系统性测试4种策略组合 |
| `main_advanced.py` | 灵活训练脚本 | 高度可配置的参数设置 |
| `max_len.py` | 长度敏感性分析 | 128 vs 256 对比实验 |

## 📊 实验结果与洞察

### 性能对比矩阵

| 方法 | Macro-F1 | Accuracy | 训练时间 | 推理速度 |
|------|----------|----------|----------|----------|
| SVM (Baseline) | 0.922 | 0.924 | ~2min | ⚡ 极快 |
| BERT Baseline | 0.947 | 0.948 | ~1h | 🚀 快速 |
| BERT + 策略优化 | 0.956 | 0.957 | ~2.5h | 🚀 快速 |

### 关键发现

1. **策略威力**：Layer Freezing + Label Smoothing可提升2-3%的Macro-F1
2. **长度权衡**：128 vs 256长度，性能提升有限但显存消耗翻倍
3. **类别挑战**：财经/科技类最易混淆，建议针对性优化

### 可视化示例

- **混淆矩阵**：清晰展示各类别预测准确性
- **训练曲线**：Loss下降和F1提升的动态图
- **错误分析**：Top-10最易错分类样本

## 🔧 高级配置

### 自定义超参数

```python
class Config:
    # 模型设置
    model_name = 'bert-base-chinese'  # 或 'ernie-3.0-base-zh'
    max_len = 256                     # 根据显存调整 (当前环境支持256)
    batch_size = 16                   # OOM时减小 (当前GPU内存充足)
    
    # 训练策略
    warmup_ratio = 0.1               # 预热比例
    label_smoothing = 0.1            # 标签平滑
    freeze_layers = 6                # 冻结层数
    
    # 优化设置
    learning_rate = 2e-5            # BERT经典学习率
    use_amp = True                   # 混合精度训练
```

### GPU优化技巧

```bash
# 多GPU训练 (如果可用)
export CUDA_VISIBLE_DEVICES=0,1
python bert_thucnews_strategy_ablation.py

# 内存优化
# 减小batch_size 或 max_len
# 启用 gradient_checkpointing
```

## ❓ 常见问题

**Q: 训练时出现CUDA out of memory？**
A: 尝试减小 `batch_size` 到16或8，或设置 `max_len=128`

**Q: 中文显示为方块？**
A: 脚本会自动检测字体，如失败请安装 `sudo apt install fonts-noto-cjk`

**Q: 如何添加新类别？**
A: 修改 `Config.class_list`，确保数据集标签一致

**Q: 实验结果在哪里？**
A: 查看 `outputs_*/summary.txt` 和 `top_misclassified.csv`

## 🤝 贡献指南

欢迎各种形式的贡献！

### 开发流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

### 贡献类型

- 🐛 **Bug修复**：修复现有问题
- ✨ **新功能**：添加实验策略或可视化
- 📚 **文档**：改进README或代码注释
- 🎨 **优化**：性能提升或代码重构
- 🧪 **测试**：添加单元测试或集成测试

## 📚 致谢



- **THUCNews数据集**：感谢清华大学提供的高质量中文新闻数据集
- **Hugging Face**：优秀的Transformers库让BERT实验变得简单
- **开源社区**：PyTorch、Scikit-learn等工具的支持

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**🌟 如果这个项目对你有帮助，请给它一个 Star！🌟**

[报告问题](https://github.com/your-repo/issues) | [讨论交流](https://github.com/your-repo/discussions) | [项目主页](https://github.com/your-repo)

*最后更新：2025年12月20日*

</div>