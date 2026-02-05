# BERT + HGNN + Attention 文本分类模型【如果你觉得原作者的 README.md 以后可能有用（比如看数据集的标签定义），你可以把它也移出来改名为 README_ToSarcasm.md 留着备查】

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