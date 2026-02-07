# ✈️ 训练前检查清单（Pre-flight Checklist）

## 状态：✅ 全部通过，可以起飞！

检查时间：2026-02-07

---

## 📋 关键检查项

### 1. ✅ 数据泄露修复
- [x] 所有样本都有 topic（100%）
- [x] ChnSentiCorp 填充随机通用 topic
- [x] Weibo 填充随机通用 topic
- [x] ToSarcasm 保留真实新闻标题
- [x] 模型无法通过"有无 topic"作弊

**验证命令**:
```bash
python verify_fix.py
```

**预期输出**:
```
✅ 包含 topic 的样本: 22389/22389 (100.00%)
🎉 修复成功！所有样本都有 topic，数据泄露问题已解决！
```

---

### 2. ✅ 类别平衡
- [x] Label 0 (正面): 42.04%
- [x] Label 1 (负面): 40.55%
- [x] Label 2 (反讽): 17.40%
- [x] 不平衡比例: 2.42:1（可接受）

**对比**:
| 版本 | Label 0 | Label 1 | Label 2 | 不平衡比例 |
|------|---------|---------|---------|-----------|
| 旧版本 | 44% | 44% | 3% | 25:1 ❌ |
| 新版本 | 42% | 41% | 17% | 2.42:1 ✅ |

---

### 3. ✅ 权重配置修正
- [x] 权重从 15.0 降至 2.5
- [x] 符合当前数据分布（17% 反讽）
- [x] 避免过度预测反讽

**配置**:
```python
class_weights = torch.tensor([1.0, 1.0, 2.5]).to(device)
```

**有效权重验证**:
```
Label 0: 42% × 1.0 = 42%
Label 1: 41% × 1.0 = 41%
Label 2: 17% × 2.5 = 42.5%
结果: 三个类别基本平衡 ✅
```

---

### 4. ✅ 代码配置
- [x] `main.py`: num_classes = 3
- [x] `main.py`: class_weights = [1.0, 1.0, 2.5]
- [x] `utils.py`: 支持三分类指标
- [x] `utils.py`: f1_macro 计算
- [x] `utils.py`: 多分类 AUC

**验证命令**:
```bash
python check_3class_ready.py
```

**预期输出**:
```
✅ 所有检查通过！
🎉 模型已准备就绪，可以开始训练！
```

---

### 5. ✅ 数据文件
- [x] `dataset/processed/train.json`: 22,389 样本
- [x] `dataset/processed/dev.json`: 2,798 样本
- [x] `dataset/processed/test.json`: 2,800 样本
- [x] 所有文件包含三个类别 {0, 1, 2}

**验证命令**:
```bash
cd dataset
python verify_data.py
```

---

### 6. ✅ 显存优化
- [x] Batch Size: 16（适配 12GB 显存）
- [x] Max Length: 256（适配 12GB 显存）
- [x] 预期显存使用: 6-8 GB

**配置**:
```python
--batch_size 16
max_length=256  # 训练/验证/测试集统一
```

---

## 🎯 训练参数总览

### 数据参数
```bash
--dataset_dir dataset/processed
--num_classes 3
--batch_size 16
```

### 模型参数
```python
bert_model: bert-base-chinese
hgnn_hidden_dims: [512, 256]
num_attention_heads: 8
dropout: 0.1
```

### 训练参数
```python
epochs: 50
learning_rate: 2e-5 (BERT), 1e-3 (HGNN+Attention)
weight_decay: 0.01
patience: 7 (早停)
```

### 损失函数
```python
criterion: CrossEntropyLoss(weight=[1.0, 1.0, 2.5])
```

---

## 🚀 启动训练

### 完整命令
```bash
python main.py --dataset_dir dataset/processed --num_classes 3
```

### 简化命令（使用默认参数）
```bash
python main.py
```

---

## 📊 重点观察指标

### 主要指标
1. **F1 Macro** - 平等对待每个类别的 F1 分数
2. **反讽_f1** - Label 2 的 F1 分数（最重要）
3. **反讽_precision** - 避免误报
4. **反讽_recall** - 找到真正的反讽

### 次要指标
- Accuracy（整体准确率）
- AUC（模型区分能力）
- 正面_f1、负面_f1

### 预期表现
```
修复后（真实场景）:
- Accuracy: 70-85%
- F1 Macro: 65-80%
- 反讽_f1: 60-75%
- 反讽_precision: 60-80%
- 反讽_recall: 60-80%
```

**注意**: 修复后的准确率可能比修复前低，这是**正常且正确的**！

---

## ⚠️ 常见问题

### Q1: 为什么准确率比修复前低？
**A**: 修复前模型通过"有无 topic"作弊，准确率虚高。修复后模型必须真正学习反讽特征，准确率更真实。

### Q2: 如果反讽 Recall 太低怎么办？
**A**: 可以适当提高权重：
```python
class_weights = torch.tensor([1.0, 1.0, 3.0]).to(device)
```

### Q3: 如果反讽 Precision 太低怎么办？
**A**: 可以适当降低权重：
```python
class_weights = torch.tensor([1.0, 1.0, 2.0]).to(device)
```

### Q4: 显存不足怎么办？
**A**: 降低 batch size：
```bash
python main.py --batch_size 8
```

---

## 📝 修复历史

### 第一次修复：数据泄露（2026-02-07）
- **问题**: ToSarcasm 有 topic，ChnSentiCorp/Weibo 无 topic
- **修复**: 为所有数据填充 topic
- **脚本**: `dataset/build_dataset_3class_fixed.py`

### 第二次修复：权重配置（2026-02-07）
- **问题**: 权重 15.0 针对 3% 反讽设计，现在反讽占 17%
- **修复**: 权重从 15.0 降至 2.5
- **文件**: `main.py` (第 250-256 行)

---

## ✅ 最终确认

### 所有检查项通过
- ✅ 数据泄露已修复
- ✅ 类别分布已平衡
- ✅ 权重配置已修正
- ✅ 代码配置已验证
- ✅ 数据文件已准备
- ✅ 显存优化已完成

### 文档完整
- ✅ `process.txt` - 完整开发日志
- ✅ `DATA_LEAKAGE_FIX_COMPLETE.md` - 数据泄露修复文档
- ✅ `WEIGHT_FIX_COMPLETE.md` - 权重修正文档
- ✅ `PRE_FLIGHT_CHECKLIST.md` - 本检查清单

---

## 🎉 准备就绪！

**状态**: ✅ 所有检查通过  
**时间**: 2026-02-07  
**下一步**: 开始训练

```bash
python main.py --dataset_dir dataset/processed --num_classes 3
```

**预计训练时间**: 
- 12GB GPU: 约 2-3 小时（50 epochs）
- 可以使用早停机制提前结束

**祝训练顺利！** 🚀
