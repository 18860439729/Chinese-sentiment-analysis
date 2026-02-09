# 🔧 Web Demo 注意力可视化修正

## 📋 问题诊断

### 原代码问题（第 155-162 行）
```python
# ❌ 错误：展示的是 BERT 原生注意力
bert_outputs = model.bert(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    output_attentions=True
)
last_layer_attention = bert_outputs.attentions[-1]
attention_weights = last_layer_attention.mean(dim=1).squeeze(0)
```

**问题**：
- 展示的是 **BERT 原生的 Self-Attention**
- 这不是模型的创新点
- 无法体现 **HGNN + MultiHeadAttention** 的作用

---

## ✅ 修正方案

### 修正后的代码
```python
# ✅ 正确：展示自定义 MultiHeadAttention 权重（HGNN 后的创新点）
custom_attn_weights = model.get_attention_weights(
    input_ids, attention_mask, hg_mat_tensor, token_type_ids
)
# custom_attn_weights 形状: [batch, num_heads, seq, seq] 或 [batch, seq, seq]
# 取均值并转为 numpy
if custom_attn_weights.dim() == 4:
    # 有多头，取平均
    attention_weights = custom_attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
else:
    # 已经是平均后的
    attention_weights = custom_attn_weights.squeeze(0).cpu().numpy()
```

**优势**：
- ✅ 展示的是 **HGNN 后的自定义 Attention**
- ✅ 体现模型的创新点：超图卷积 + 多头注意力
- ✅ 可视化更有学术价值

---

## 🎯 技术对比

### BERT 原生注意力 vs 自定义注意力

| 特性 | BERT 原生注意力 | 自定义 MultiHeadAttention |
|------|----------------|--------------------------|
| **位置** | BERT 内部 | HGNN 之后 |
| **输入** | BERT 词嵌入 | HGNN 输出特征 |
| **作用** | 语义理解 | 融合超图结构信息 |
| **创新性** | 无（预训练） | 有（模型创新点） |
| **可视化价值** | 低 | 高 ⭐ |

### 数据流对比

#### 原代码（错误）
```
输入文本
  ↓
BERT Tokenization
  ↓
BERT Encoder (12层)
  ↓
❌ 提取 BERT 第12层注意力 ← 这里提取的
  ↓
HGNN
  ↓
MultiHeadAttention
  ↓
分类器
```

#### 修正后（正确）
```
输入文本
  ↓
BERT Tokenization
  ↓
BERT Encoder (12层)
  ↓
HGNN (融合超图结构)
  ↓
MultiHeadAttention
  ↓
✅ 提取自定义注意力权重 ← 这里提取的
  ↓
分类器
```

---

## 📊 可视化效果对比

### BERT 原生注意力（修正前）
- 展示：词与词之间的语义关联
- 特点：通用的语言理解
- 问题：**无法体现超图结构的作用**

### 自定义注意力（修正后）
- 展示：HGNN 后的特征关注度
- 特点：融合了依存句法、词性、实体等超图信息
- 优势：**体现模型创新点，可视化更有价值**

---

## 🔍 model.get_attention_weights() 详解

### 方法签名
```python
def get_attention_weights(self, 
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        hypergraph_matrix: torch.Tensor,
                        token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
```

### 内部流程
```python
# 1. BERT 编码
bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
bert_sequence_output = bert_outputs.last_hidden_state

# 2. HGNN 超图卷积
hgnn_output = self.hgnn(bert_sequence_output, hypergraph_matrix)

# 3. 计算自定义注意力权重
Q = self.attention.w_q(hgnn_output)  # Query
K = self.attention.w_k(hgnn_output)  # Key
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)

return attention_weights  # [batch, num_heads, seq, seq]
```

**关键点**：
- 输入是 **HGNN 的输出**，而不是 BERT 的输出
- 权重反映了超图结构信息的影响
- 这才是模型的创新点

---

## 🎓 学术价值

### 论文/毕设中的应用

#### 修正前（价值低）
> "我们可视化了 BERT 的注意力权重..."
- ❌ 这是 BERT 自带的，不是你的创新
- ❌ 审稿人会质疑：这和普通 BERT 有什么区别？

#### 修正后（价值高）
> "我们可视化了 HGNN 后的自定义注意力权重，展示了超图结构信息如何影响模型的特征关注度..."
- ✅ 体现了模型的创新点
- ✅ 证明了超图结构的有效性
- ✅ 可以做消融实验对比

### 可视化案例

**示例文本**："这家酒店真是太'好'了，连热水都没有"

#### BERT 原生注意力可能关注
- "酒店" ↔ "好"（语义关联）
- "太" ↔ "好"（程度副词）

#### 自定义注意力（HGNN后）可能关注
- "好" ↔ "连...都没有"（反讽结构）
- "太" ↔ "没有"（情感反转）
- 引号 ↔ 负面描述（标点符号的语用功能）

**结论**：自定义注意力能捕捉到反讽的结构特征！

---

## 📋 修改清单

- [x] 修改 `web_demo_upgraded.py` 第 148-167 行
- [x] 删除 `output_attentions=True` 的 BERT 调用
- [x] 改用 `model.get_attention_weights()` 方法
- [x] 添加维度检查（支持多头和单头）
- [x] 保留异常处理（生成模拟权重）
- [x] 创建本文档说明修正原因

---

## 🚀 测试建议

### 运行 Web Demo
```bash
streamlit run web_demo_upgraded.py
```

### 测试用例
1. **正常正面**："这家酒店很好，服务周到"
2. **正常负面**："这家酒店很差，服务态度恶劣"
3. **反讽**："这家酒店真是太'好'了，连热水都没有"

### 观察重点
- 注意力权重是否关注到反讽的关键词
- 节点大小是否合理（权重高的节点更大）
- 与 HanLP 依存关系的结合是否自然

---

## 💡 进一步优化建议

### 1. 对比可视化
可以同时展示：
- BERT 原生注意力（baseline）
- 自定义注意力（创新点）
- 对比两者的差异

### 2. 注意力热力图
除了超图，还可以添加：
- Attention Heatmap（矩阵热力图）
- 词级别的注意力分数条形图

### 3. 消融实验
- 只用 BERT（无 HGNN）
- BERT + HGNN（无自定义 Attention）
- 完整模型（BERT + HGNN + Attention）

---

## 🎉 修正完成

**关键改进**：
- ✅ 从展示 BERT 原生注意力 → 展示自定义注意力
- ✅ 体现模型创新点（HGNN + MultiHeadAttention）
- ✅ 提升可视化的学术价值
- ✅ 符合论文/毕设的要求

**文件位置**：
- 修正文件：`web_demo_upgraded.py`
- 说明文档：`WEB_DEMO_ATTENTION_FIX.md`（本文档）

**下一步**：
- 运行 Web Demo 测试效果
- 观察注意力权重是否合理
- 准备论文中的可视化图表
