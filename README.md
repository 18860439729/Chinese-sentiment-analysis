# 基于超图神经网络的情感分析系统 (Sentiment_Model)

## 1. 项目目录说明
- **.vscode/**: 存放本地开发环境配置（含 settings.json）。
- **dataset/**: 存放核心训练数据（train/test/dev.json）。
- **data_preprocess.py**: 数据预处理模块，负责 BERT 分词及超图 H 矩阵构建。
- **model.py**: 模型定义模块，包含 BERT + HGNN + Attention 结构。
- **README_ToSarcasm.md**: 数据集原始说明文档（含标签定义）。
【1. .vscode 文件夹：你的“私人助理”
结论：千万不要删，它非常有用。
它的角色：它存放的是你这个项目专属的编辑器配置。
具体作用：
Python 解释器路径：如果你在 .vscode/settings.json 里配置了某个 Conda 环境，删了它，你的编辑器可能就找不到 PyTorch 了。

调试配置 (Debug)：如果你以后设置了断点调试，配置都会存在这里。

远程 SSH 映射：当你之后配置连接 10.12.1.9 服务器时，相关的本地映射记录也可能存在这里。

日志记录建议：“.vscode 文件夹：存放 VS Code 本地配置（如 Python 路径、编辑器插件设置），保证开发环境一致性。不可删除。”

2. README_ToSarcasm.md：你的“论文素材库”
结论：代码运行不需要它，但写论文时它是“救命稻草”。
它的角色：它是你从 GitHub 下载的数据集的说明书。
具体作用：
标签含义：它会告诉你 label 0 代表什么，1 代表什么（比如：0 是正常文本，1 是反讽）。

数据来源：你论文的“实验设置”章节需要引用这个数据集的来源和统计信息。

版权与引用：如果你的论文要发表，必须根据这个文件里的要求引用原作者的论文。

日志记录建议：“README_ToSarcasm.md：ToSarcasm 原始数据集说明文档。包含数据标签定义、引用信息。代码运行无关，论文写作必看。”】
## 2. 环境依赖
- Python 3.8+
- PyTorch, Transformers, DHG

## 3. 开发日志
- [2026/02/05] 清理了原 GitHub 冗余代码，仅保留核心数据至 dataset 文件夹。
- [2026/02/05] 确立了使用 BERT Tokenizer 代替手动词汇表的方案。







# BERT + HGNN + Attention 文本分类模型
这是一个结合了BERT、超图神经网络(HGNN)和注意力机制的文本分类模型的模块化实现。

## 项目结构

```
├── data_preprocess.py    # 数据预处理模块，负责HanLP处理和超图结构生成
├── model.py             # 模型定义模块，包含BERT+HGNN+Attention网络结构
├── utils.py             # 工具函数模块，包含各种辅助功能
├── main.py              # 主程序入口，实验的入口文件
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明文档
```

## 模块说明

### data_preprocess.py
- `DataPreprocessor`: 数据预处理器类
  - 使用HanLP进行文本处理（分词、词性标注、命名实体识别等）
  - 构建超图结构（节点和边）
  - 生成词汇表和特征矩阵
- `load_dataset`: 数据集加载函数
- `create_data_loaders`: 创建PyTorch数据加载器

### model.py
- `HypergraphConvolution`: 超图卷积层
- `MultiHeadAttention`: 多头注意力机制
- `HGNN`: 超图神经网络
- `BertHGNNModel`: 主模型类，融合BERT、HGNN和注意力机制

### utils.py
- `EarlyStopping`: 早停机制
- `MetricsCalculator`: 指标计算器
- `Visualizer`: 可视化工具
- `AverageMeter`: 平均值计算器
- 其他辅助函数（日志设置、模型保存/加载等）

### main.py
- 命令行参数解析
- 训练和验证循环
- 模型评估和结果可视化
- 实验管理和日志记录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练命令

```bash
python main.py \
    --train_data path/to/train.txt \
    --val_data path/to/val.txt \
    --bert_model bert-base-chinese \
    --num_classes 2 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 2e-5
```

### 主要参数说明

- `--train_data`: 训练数据路径
- `--val_data`: 验证数据路径  
- `--test_data`: 测试数据路径（可选）
- `--bert_model`: BERT模型名称（默认：bert-base-chinese）
- `--hgnn_hidden_dims`: HGNN隐藏层维度列表（默认：[512, 256]）
- `--num_attention_heads`: 注意力头数（默认：8）
- `--num_classes`: 分类类别数（默认：2）
- `--batch_size`: 批次大小（默认：32）
- `--epochs`: 训练轮数（默认：50）
- `--learning_rate`: 学习率（默认：2e-5）
- `--save_dir`: 模型保存目录（默认：./checkpoints）
- `--resume`: 恢复训练的检查点路径
- `--evaluate_only`: 仅进行评估

### 数据格式

训练数据应为制表符分隔的文本文件，每行格式为：
```
文本内容\t标签
```

例如：
```
这是一个正面评论	1
这是一个负面评论	0
```

### 模型特点

1. **BERT编码**: 使用预训练的BERT模型进行文本编码
2. **超图神经网络**: 通过HanLP构建的语言学特征建立超图结构
3. **多头注意力**: 增强模型对重要信息的关注能力
4. **特征融合**: 将BERT特征和HGNN特征进行融合

### 输出文件

训练完成后会生成以下文件：
- `best_model.pth`: 最佳模型检查点
- `config.json`: 训练配置
- `training_history.png`: 训练历史曲线
- `confusion_matrix.png`: 混淆矩阵
- 训练日志文件

## 扩展说明

这个模块化设计便于：
1. 独立测试各个组件
2. 替换不同的预处理方法
3. 尝试不同的模型架构
4. 添加新的评估指标
5. 集成到更大的系统中

每个模块都有清晰的接口和文档，可以根据具体需求进行修改和扩展。