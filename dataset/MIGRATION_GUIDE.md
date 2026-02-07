# 三分类升级迁移指南

## 📋 变更总结

### 从二分类到三分类

**旧方案（二分类）**:
- Label 0: 非反讽
- Label 1: 反讽

**新方案（三分类）**:
- Label 0: 正常-正面
- Label 1: 正常-负面
- Label 2: 阴阳怪气

## 🔧 必须修改的代码

### 1. main.py

**修改命令行参数默认值**:
```python
# 找到这一行（约第 30 行）
parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')

# 改为
parser.add_argument('--num_classes', type=int, default=3, help='分类类别数')
```

### 2. model.py

**检查输出层定义**:
```python
# 确保 num_classes 参数正确传递
def __init__(self, ..., num_classes=3, ...):
    ...
    self.classifier = nn.Linear(hidden_size, num_classes)
```

### 3. 训练命令

**使用新数据训练**:
```bash
python main.py --dataset_dir dataset/processed --num_classes 3
```

## 📊 数据变更

### 数据规模
- **总样本**: 136,859 条
- **训练集**: 109,486 条
- **验证集**: 13,685 条
- **测试集**: 13,688 条

### 标签分布
- **Label 0 (正面)**: 48.22%
- **Label 1 (负面)**: 48.22%
- **Label 2 (反讽)**: 3.56%

## ⚠️ 类别不平衡处理

### 方法 1: 类别权重

在 `main.py` 中添加：

```python
# 定义损失函数时添加权重
class_weights = torch.tensor([1.0, 1.0, 13.5]).to(device)  # 反讽权重提高
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 方法 2: Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 使用
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### 方法 3: 评估指标调整

不仅看 Accuracy，重点关注：
- **F1-Score (Macro)**: 平等对待每个类别
- **F1-Score (Weighted)**: 按样本数加权
- **每个类别的 Precision/Recall**
- **混淆矩阵**

## 📈 评估指标更新

### utils.py 修改建议

在 `MetricsCalculator.calculate_metrics()` 中：

```python
# 添加每个类别的指标
for i, label_name in enumerate(['正面', '负面', '反讽']):
    metrics[f'{label_name}_precision'] = precision_per_class[i]
    metrics[f'{label_name}_recall'] = recall_per_class[i]
    metrics[f'{label_name}_f1'] = f1_per_class[i]
```

### 打印分类报告时

```python
labels = ['正面', '负面', '反讽']
print(classification_report(y_true, y_pred, target_names=labels))
```

## 🎯 "强撑"识别实现

### 推理时的后处理

```python
def detect_complex_emotion(probs):
    """
    检测复杂情感（如"强撑"）
    
    Args:
        probs: [P(正面), P(负面), P(反讽)]
    
    Returns:
        emotion_type: 'positive', 'negative', 'sarcastic', 'struggling'
    """
    p_pos, p_neg, p_sar = probs
    
    # 强撑：正面概率高 + 反讽概率也不低
    if p_pos > 0.4 and p_sar > 0.2:
        return 'struggling'  # 强撑/苦笑
    
    # 正常判断
    max_idx = probs.argmax()
    if max_idx == 0:
        return 'positive'
    elif max_idx == 1:
        return 'negative'
    else:
        return 'sarcastic'
```

## 🧪 测试建议

### 1. 单元测试

测试每个类别的识别：

```python
test_cases = [
    ("这个产品真的很好用！", 0),  # 正面
    ("太差了，完全不能用", 1),    # 负面
    ("呵呵，真是太好了呢", 2),    # 反讽
]

for text, expected_label in test_cases:
    pred = model.predict(text)
    assert pred == expected_label
```

### 2. 边界案例测试

```python
edge_cases = [
    "还行吧",           # 中性，可能是正面或负面
    "我太开心了呢[微笑]", # 可能是强撑
    "真是厉害啊",       # 可能是反讽
]
```

### 3. 混淆矩阵分析

重点关注：
- 正面 vs 反讽 的混淆
- 负面 vs 反讽 的混淆

## 📝 完整训练流程

```bash
# 1. 数据清洗（如果还没做）
cd dataset
python build_dataset_3class.py

# 2. 验证数据
python verify_data.py

# 3. 修改代码
# - main.py: num_classes=3
# - model.py: 确认输出层
# - 添加类别权重（可选）

# 4. 训练模型
cd ..
python main.py \
    --dataset_dir dataset/processed \
    --num_classes 3 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 2e-5

# 5. 评估
python main.py \
    --dataset_dir dataset/processed \
    --num_classes 3 \
    --evaluate_only \
    --resume checkpoints/best_model.pth
```

## 🎓 论文写作建议

### 创新点
1. 带情感极性的反讽检测（三分类）
2. 复杂情感识别（强撑/苦笑）
3. 多数据源融合策略

### 消融实验
1. 二分类 vs 三分类
2. 不同数据源的贡献
3. 类别权重的影响

### 评估指标
- Accuracy
- F1-Score (Macro/Weighted)
- 每个类别的 P/R/F1
- 混淆矩阵
- AUC (One-vs-Rest)

## ✅ 检查清单

- [ ] 修改 main.py 的 num_classes 默认值
- [ ] 修改 model.py 的输出层
- [ ] 添加类别权重或 Focal Loss
- [ ] 更新评估指标（F1-Score）
- [ ] 测试三个类别的识别
- [ ] 实现"强撑"检测逻辑
- [ ] 准备消融实验
- [ ] 撰写论文相关章节

## 🎉 预期效果

使用三分类后：
- ✅ 更准确的情感分析
- ✅ 可以识别反讽
- ✅ 可以检测复杂情感
- ✅ 更高的学术价值
- ✅ 更强的实用性

祝训练顺利！🚀
