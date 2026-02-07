# 三分类模型快速开始指南

## 🚀 5 分钟快速开始

### 步骤 1: 数据已准备好 ✅

数据已经清洗完成，位于：
- `dataset/processed/train.json` (109,486 条)
- `dataset/processed/dev.json` (13,685 条)
- `dataset/processed/test.json` (13,688 条)

### 步骤 2: 修改代码（必须！）

#### 2.1 修改 main.py

找到第 30 行左右：
```python
parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
```

改为：
```python
parser.add_argument('--num_classes', type=int, default=3, help='分类类别数')
```

#### 2.2 添加类别权重（推荐）

在 `main.py` 的 `main()` 函数中，找到定义损失函数的地方（约第 230 行）：

```python
# 定义损失函数和优化器 - 分层学习率策略
criterion = nn.CrossEntropyLoss()
```

改为：
```python
# 定义损失函数和优化器 - 分层学习率策略
# 添加类别权重处理反讽数据不平衡（反讽只占3.56%）
class_weights = torch.tensor([1.0, 1.0, 13.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
logger.info(f"使用类别权重: {class_weights.tolist()}")
```

### 步骤 3: 训练模型

```bash
python main.py --dataset_dir dataset/processed --num_classes 3 --batch_size 16 --epochs 50
```

### 步骤 4: 查看结果

训练完成后，查看：
- 训练日志：`logs/training_*.log`
- 模型文件：`checkpoints/best_model.pth`
- 训练曲线：`checkpoints/training_history.png`
- 混淆矩阵：`checkpoints/confusion_matrix.png`

## 📊 三分类标签说明

| Label | 含义 | 示例 |
|-------|------|------|
| 0 | 正常-正面 | "这个产品真的很好用！" |
| 1 | 正常-负面 | "太差了，完全不能用" |
| 2 | 阴阳怪气 | "呵呵，真是太好了呢" |

## ⚠️ 常见问题

### Q1: 为什么要用类别权重？
A: 反讽数据只占 3.56%，不加权重模型会倾向于预测 Label 0/1，导致反讽识别率低。

### Q2: 如何评估模型？
A: 不要只看 Accuracy，重点关注：
- F1-Score (Macro)
- 每个类别的 Precision/Recall
- 混淆矩阵（特别是反讽的识别率）

### Q3: 训练多久？
A: 建议至少 30 个 epoch，观察验证集 F1-Score 是否收敛。

### Q4: 显存不够怎么办？
A: 降低 batch_size 到 8 或 4。

## 🎯 预期效果

### 好的模型应该达到：
- **Accuracy**: > 85%
- **F1-Score (Macro)**: > 0.75
- **反讽 F1-Score**: > 0.60

### 如果效果不好：
1. 增加反讽的类别权重（如 20.0）
2. 使用 Focal Loss
3. 增加训练轮数
4. 调整学习率

## 📝 完整命令示例

```bash
# 训练
python main.py \
    --dataset_dir dataset/processed \
    --num_classes 3 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 2e-5 \
    --patience 10

# 评估
python main.py \
    --dataset_dir dataset/processed \
    --num_classes 3 \
    --evaluate_only \
    --resume checkpoints/best_model.pth
```

## 🔍 调试技巧

### 查看数据分布
```bash
cd dataset
python verify_data.py
```

### 测试单个样本
```python
from inference import predict

text = "呵呵，真是太好了呢"
result = predict(text, model_path='checkpoints/best_model.pth')
print(f"预测: {result['label']}, 概率: {result['probs']}")
```

## 📚 更多文档

- 详细说明：`dataset/README.md`
- 迁移指南：`dataset/MIGRATION_GUIDE.md`
- 开发日志：`process.txt`

## 🎉 开始训练吧！

修改完代码后，直接运行：
```bash
python main.py --dataset_dir dataset/processed --num_classes 3
```

祝训练顺利！🚀
