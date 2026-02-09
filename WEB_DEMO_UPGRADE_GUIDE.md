# 🎉 Web Demo 升级完成指南

## ✅ 升级内容总览

### 新文件
- **web_demo_upgraded.py** - 全新升级版 Web 界面

### 核心升级

#### 1. 🔥 注意力热力图（Attention Heatmap）
- **技术**: Plotly 交互式热力图
- **功能**: 可视化 BERT 最后一层的注意力权重
- **展示**: 横轴和纵轴都是 Token，颜色深浅表示权重大小
- **位置**: "Mochi 视角" 区域的折叠卡片中

#### 2. 🕸️ PyVis 交互式超图
- **技术**: PyVis Network（替代 Graphviz）
- **创新点**:
  - 节点大小由注意力权重决定（越重要越大）
  - 节点颜色由注意力权重决定（越重要越红）
  - 边的颜色和粗细表示依存关系类型
  - 可拖拽、可缩放的交互式体验
- **超图特性**: 通过颜色和粗细体现"超边"概念

#### 3. 📊 批量数据处理（真实实现）
- **功能**: 上传 CSV/Excel 文件
- **处理**: 自动循环调用预测函数
- **输出**: 带情感标签的结果表格
- **下载**: 提供 CSV 下载按钮

#### 4. 🔌 API 接口文档（真实展示）
- **内容**: 标准 RESTful API 调用示例
- **格式**: JSON Request/Response 结构
- **示例**: Python 和 cURL 调用代码
- **目的**: 展示工程化落地能力

#### 5. ⚙️ 系统设置（真实逻辑）
- **参数**: 最小置信度阈值（Slider）
- **开关**: 显示详细日志（Toggle）
- **存储**: 使用 st.session_state 保存设置
- **效果**: 设置实时影响主界面显示

#### 6. 👤 用户中心（真实登录）
- **登录**: 预设账号 admin/123456, demo/demo
- **状态**: 登录后显示用户信息
- **历史**: 显示历史查询记录
- **导出**: 提供历史记录下载

---

## 🚀 使用方法

### 1. 安装依赖
```bash
pip install streamlit pyvis plotly pandas openpyxl
```

### 2. 运行升级版
```bash
streamlit run web_demo_upgraded.py
```

### 3. 访问地址
```
http://localhost:8501
```

---

## 📋 功能模块详解

### 🚀 情感分析实验室（主功能）

#### 输入
- **话题 (Topic)**: 可选，提供语境信息
- **文本 (Text)**: 必填，待分析的中文短文本

#### 输出
1. **情感结果卡片**
   - 情感类别：正面/负面/反讽
   - 置信度百分比
   - 动画效果

2. **情感倾向分布**
   - 三个类别的概率分布
   - 进度条可视化

3. **Mochi 视角（右侧）**
   - **超图结构**: PyVis 交互式网络图
     - 节点大小 = 注意力权重
     - 节点颜色 = 注意力权重（红色越深越重要）
     - 边的颜色 = 依存关系类型
     - 可拖拽、可缩放
   
   - **注意力热力图**: Plotly 热力图
     - X轴: Key (被关注的词)
     - Y轴: Query (关注者)
     - 颜色: 注意力权重强度

4. **调试信息**（开启详细日志后）
   - Token 数量
   - 注意力矩阵形状
   - HanLP 数据键名
   - 设备信息

---

### 📊 批量数据处理

#### 使用流程
1. 准备 CSV/Excel 文件，必须包含 `text` 列
2. 可选包含 `topic` 列（否则默认为"网友评论"）
3. 上传文件
4. 点击"开始批量分析"
5. 等待进度条完成
6. 查看结果表格
7. 下载分析结果

#### 输出字段
- `text`: 原始文本
- `topic`: 话题
- `sentiment`: 情感类别（正面/负面/反讽）
- `confidence`: 置信度
- `prob_positive`: 正面概率
- `prob_negative`: 负面概率
- `prob_sarcasm`: 反讽概率

---

### 🔌 API 接口文档

#### 展示内容
1. **接口地址**: POST https://api.mochi-sentiment.com/v1/analyze
2. **请求示例**: JSON 格式
3. **响应示例**: JSON 格式（包含详细字段）
4. **Python 调用**: requests 库示例代码
5. **cURL 调用**: 命令行示例

#### 目的
- 展示系统的工程化能力
- 说明如何程序化调用模型
- 适合答辩时展示"落地方案"

---

### ⚙️ 系统设置

#### 可调参数
1. **最小置信度阈值** (0.0 - 1.0)
   - 用途: 低于此阈值的预测标记为不确定
   - 存储: st.session_state.min_confidence

2. **显示详细日志** (开/关)
   - 用途: 显示模型内部处理细节
   - 存储: st.session_state.show_debug

3. **界面主题** (下拉选择)
   - 选项: 默认主题/深色模式/简约模式
   - 状态: 仅展示，暂未实现切换

4. **界面语言** (下拉选择)
   - 选项: 简体中文/English
   - 状态: 仅展示，暂未实现切换

#### 保存机制
- 点击"保存设置"按钮
- 设置保存到 st.session_state
- 主界面实时读取这些设置

---

### 👤 用户中心

#### 登录功能
- **预设账号**:
  - admin / 123456 (管理员)
  - demo / demo (普通用户)
- **登录状态**: 保存在 st.session_state.logged_in
- **用户名**: 保存在 st.session_state.username

#### 用户信息
- 用户名
- 账户类型（管理员/普通用户）
- 注册时间（模拟）
- 上次登录时间（实时）
- 累计分析次数

#### 历史记录
- **字段**: 时间、文本、话题、情感、置信度
- **存储**: st.session_state.history
- **功能**: 
  - 表格展示
  - CSV 导出下载

---

## 🎯 技术亮点

### 1. 注意力权重提取
```python
# 从 BERT 提取最后一层注意力
bert_outputs = model.bert(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    output_attentions=True  # 关键参数
)
last_layer_attention = bert_outputs.attentions[-1]
attention_weights = last_layer_attention.mean(dim=1).squeeze(0)
```

### 2. PyVis 超图节点大小映射
```python
# 节点大小由注意力权重决定
score = attention_weights[i].mean()
size = 10 + 40 * (score - min_score) / (max_score - min_score + 1e-6)
```

### 3. 颜色映射（注意力 → 红色强度）
```python
color_intensity = int(255 * (score - min_score) / (max_score - min_score + 1e-6))
color = f"#{255-color_intensity:02x}{color_intensity:02x}{color_intensity:02x}"
```

### 4. Session State 管理
```python
# 初始化
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# 读取
if st.session_state.logged_in:
    # 显示已登录状态

# 修改
st.session_state.logged_in = True
```

---

## 📊 答辩演示建议

### 1. 情感分析实验室
- **展示**: 输入一句阴阳怪气的话
- **重点**: 
  - 右侧超图中，关键词（如"真"、"太"、"了"）节点更大更红
  - 注意力热力图显示模型关注的词对
  - 置信度和概率分布

### 2. 批量处理
- **展示**: 上传一个包含 10 条评论的 CSV
- **重点**: 
  - 进度条实时更新
  - 结果表格清晰
  - 可下载 CSV

### 3. API 文档
- **展示**: 滚动展示 JSON 和代码示例
- **重点**: 
  - 说明"这是工程化落地方案"
  - 展示 Python 和 cURL 调用方式

### 4. 系统设置
- **展示**: 调整置信度阈值，切换详细日志
- **重点**: 
  - 说明"设置会影响主界面"
  - 展示配置表格

### 5. 用户中心
- **展示**: 登录 admin 账号，查看历史记录
- **重点**: 
  - 说明"支持多用户管理"
  - 展示历史记录导出

---

## ⚠️ 注意事项

### 1. 模型文件路径
确保模型文件在正确位置：
```
checkpoints/3class_final/best_model.pth
```

### 2. 注意力权重提取
如果 BERT 不支持 `output_attentions=True`，代码会自动生成模拟的注意力矩阵（用于演示）。

### 3. 依赖安装
```bash
pip install streamlit pyvis plotly pandas openpyxl
```

### 4. 浏览器兼容性
- PyVis 交互式图表需要现代浏览器（Chrome/Firefox/Edge）
- 避免使用 IE

---

## 🎉 升级效果对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 超图可视化 | Graphviz 静态树形图 | PyVis 交互式超图（节点大小=注意力） |
| 注意力展示 | 无 | Plotly 热力图 |
| 批量处理 | "VIP 专用"提示 | 真实上传+分析+下载 |
| API 文档 | "VIP 专用"提示 | 完整 JSON/Python/cURL 示例 |
| 系统设置 | 侧边栏静态控件 | 真实 Session State 管理 |
| 用户中心 | "VIP 专用"提示 | 登录+历史记录+导出 |

---

## 🚀 下一步优化建议

### 短期（答辩前）
1. 准备几个典型的测试样例
2. 录制演示视频
3. 准备答辩讲解词

### 长期（毕设后）
1. 实现真实的后端 API
2. 添加数据库存储用户和历史
3. 实现主题切换功能
4. 添加更多可视化图表

---

**升级完成时间**: 2026-02-07  
**文件**: web_demo_upgraded.py  
**状态**: ✅ 可直接运行演示
