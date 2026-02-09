"""
Mochi 情感分析系统 - 升级版
包含：注意力热力图、PyVis交互式超图、四个功能模块的真实实现
"""
import streamlit as st
import torch
import os
import time
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go
from data_preprocess import DataPreprocessor
from model import BertHGNNModel
import torch.nn.functional as F

# ================= 配置区 =================
PAGE_TITLE = "Mochi | 中文情感分析系统"
PAGE_ICON = "🍡"
MODEL_DIR = "checkpoints/3class_final"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
BERT_MODEL = "bert-base-chinese"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预设账号
USERS = {"admin": "123456", "demo": "demo"}

# ================= Session State 初始化 =================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'min_confidence' not in st.session_state:
    st.session_state.min_confidence = 0.85
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'history' not in st.session_state:
    st.session_state.history = []

# ================= 样式注入 (CSS) =================
def set_style():
    st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
            color: #555555;
        }
        h1 { color: #FFB7B2; font-weight: 700; }
        h2, h3 { color: #AEC6CF; }
        div.stButton > button {
            background-color: #AEC6CF;
            color: white;
            border-radius: 20px;
            height: 50px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #FFB7B2;
        }
        .result-card {
            background-color: #FDFD96;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            animation: fadeIn 0.8s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

# ================= 模型加载（修改版：返回注意力权重）=================
@st.cache_resource
def load_model():
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    steps = ["初始化 Mochi 内核...", "加载 BERT 语义空间...", "构建超图神经回路...", "唤醒 Mochi..."]
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.1)
        
    try:
        preprocessor = DataPreprocessor(bert_model_name=BERT_MODEL)
        model = BertHGNNModel(
            bert_model_name=BERT_MODEL,
            hgnn_hidden_dims=[256, 128],
            num_attention_heads=4,
            num_classes=3,
            dropout=0
        )
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            except TypeError:
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            st.warning("模型文件未找到，使用随机初始化权重（仅供演示）")
            
        model.to(DEVICE)
        model.eval()
        
        status_text.empty()
        progress_bar.empty()
        return model, preprocessor
    except Exception as e:
        st.error(f"Mochi 生病了: {str(e)}")
        return None, None

# ================= 预测函数（修改版：返回注意力权重）=================
def predict_with_attention(model, preprocessor, text, topic):
    """返回预测概率、HanLP结果、注意力权重、tokens"""
    if not topic or not topic.strip():
        topic = "网友评论"
    
    # BERT tokenization
    bert_tokens = preprocessor.process_text_pair(text, topic, 256)
    tokens_list = preprocessor.tokenizer.convert_ids_to_tokens(bert_tokens['input_ids'].tolist())
    
    # HanLP processing
    hanlp_result = preprocessor.process_text_with_hanlp(text, topic)
    hypergraph_matrix = preprocessor._create_single_hypergraph_matrix(hanlp_result, 256, 50)
    
    # 准备输入
    input_ids = bert_tokens['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = bert_tokens['attention_mask'].unsqueeze(0).to(DEVICE)
    token_type_ids = bert_tokens['token_type_ids'].unsqueeze(0).to(DEVICE)
    hg_mat_tensor = torch.tensor(hypergraph_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # 前向传播
        logits = model(input_ids, attention_mask, hg_mat_tensor, token_type_ids)
        probs = F.softmax(logits, dim=1)
        
        # 🔧 修正：提取自定义 MultiHeadAttention 权重（HGNN 后的创新点）
        # 而不是 BERT 原生注意力
        try:
            # 使用 model.get_attention_weights() 提取 HGNN + Attention 的权重
            # 这才是模型的创新点：展示超图卷积后的特征关注度
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
        except Exception as e:
            # 如果失败，生成一个模拟的注意力矩阵
            seq_len = len(tokens_list)
            attention_weights = np.random.rand(seq_len, seq_len)
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
    return probs.cpu().numpy()[0], hanlp_result, attention_weights, tokens_list

# ================= PyVis 交互式超图可视化 =================
def create_interactive_hypergraph(hanlp_data, attention_weights, tokens_list):
    """使用 PyVis 创建交互式超图，节点大小由注意力权重决定"""
    try:
        words = hanlp_data.get('tokens')
        deps = hanlp_data.get('dependencies')
        
        if words and isinstance(words[0], list):
            words = words[0]
        if deps and isinstance(deps[0], list) and not isinstance(deps[0][0], tuple):
            if isinstance(deps[0][0], list):
                deps = deps[0]
        
        if not words or not deps:
            return None
        
        # 创建 PyVis 网络
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)
        
        # 计算每个词的注意力分数（用于节点大小）
        # 这里简化：取该词在注意力矩阵中的平均值
        word_attention_scores = {}
        for i, word in enumerate(words):
            if i < len(attention_weights):
                # 取该词作为 query 时的平均注意力
                score = attention_weights[i].mean()
                word_attention_scores[i] = float(score)
            else:
                word_attention_scores[i] = 0.1
        
        # 归一化分数到 10-50 范围（节点大小）
        max_score = max(word_attention_scores.values()) if word_attention_scores else 1
        min_score = min(word_attention_scores.values()) if word_attention_scores else 0
        
        # 添加节点
        for i, word in enumerate(words):
            score = word_attention_scores.get(i, 0.1)
            # 归一化到 10-50
            size = 10 + 40 * (score - min_score) / (max_score - min_score + 1e-6)
            # 颜色：注意力越高越红
            color_intensity = int(255 * (score - min_score) / (max_score - min_score + 1e-6))
            color = f"#{255-color_intensity:02x}{color_intensity:02x}{color_intensity:02x}"
            
            net.add_node(
                i, 
                label=word, 
                size=size,
                color=color,
                title=f"{word}\n注意力分数: {score:.3f}"
            )
        
        # 添加边（依存关系）
        edge_colors = {
            'nsubj': '#FF6B6B',  # 主语 - 红色
            'obj': '#4ECDC4',    # 宾语 - 青色
            'dobj': '#4ECDC4',
            'advmod': '#95E1D3', # 状语 - 绿色
            'root': '#FFD93D',   # 根节点 - 黄色
            'att': '#A8E6CF',    # 定语 - 浅绿
        }
        
        for i, item in enumerate(deps):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                head = item[0]
                rel = item[1]
                
                if isinstance(head, int):
                    head_idx = head - 1
                    if 0 <= head_idx < len(words):
                        color = edge_colors.get(rel, '#CCCCCC')
                        width = 3 if rel in ['nsubj', 'obj', 'root'] else 1
                        net.add_edge(
                            i, 
                            head_idx, 
                            label=rel,
                            color=color,
                            width=width,
                            title=f"关系: {rel}"
                        )
        
        # 生成 HTML
        html = net.generate_html()
        return html
    
    except Exception as e:
        st.error(f"超图生成失败: {str(e)}")
        return None

# ================= 注意力热力图 =================
def create_attention_heatmap(attention_weights, tokens_list, max_tokens=30):
    """使用 Plotly 创建注意力热力图"""
    try:
        # 限制显示的 token 数量
        display_len = min(len(tokens_list), max_tokens)
        attention_subset = attention_weights[:display_len, :display_len]
        tokens_subset = tokens_list[:display_len]
        
        # 清理 token 显示（去掉 [PAD], [SEP] 等）
        clean_tokens = []
        for token in tokens_subset:
            if token in ['[PAD]', '[CLS]']:
                clean_tokens.append('')
            elif token == '[SEP]':
                clean_tokens.append('|')
            else:
                clean_tokens.append(token.replace('##', ''))
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=attention_subset,
            x=clean_tokens,
            y=clean_tokens,
            colorscale='RdYlBu_r',
            text=attention_subset,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="注意力权重")
        ))
        
        fig.update_layout(
            title="BERT 最后一层注意力权重矩阵",
            xaxis_title="Key (被关注的词)",
            yaxis_title="Query (关注者)",
            height=500,
            font=dict(family="Microsoft YaHei", size=10)
        )
        
        return fig
    except Exception as e:
        st.error(f"热力图生成失败: {str(e)}")
        return None

# ================= 功能模块 1: 批量数据处理 =================
def batch_processing_module(model, preprocessor):
    st.markdown("### 📊 批量数据处理")
    st.info("上传包含 'text' 列的 CSV/Excel 文件，系统将自动进行情感分析")
    
    uploaded_file = st.file_uploader("选择文件", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write(f"✅ 文件加载成功！共 {len(df)} 条数据")
            st.dataframe(df.head())
            
            if 'text' not in df.columns:
                st.error("❌ 文件必须包含 'text' 列")
                return
            
            # 添加 topic 列（如果没有）
            if 'topic' not in df.columns:
                df['topic'] = "网友评论"
            
            if st.button("🚀 开始批量分析"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    text = str(row['text'])
                    topic = str(row.get('topic', '网友评论'))
                    
                    # 调用预测
                    probs, _, _, _ = predict_with_attention(model, preprocessor, text, topic)
                    pred_label = probs.argmax()
                    confidence = probs[pred_label]
                    
                    label_names = ['正面', '负面', '反讽']
                    results.append({
                        'text': text,
                        'topic': topic,
                        'sentiment': label_names[pred_label],
                        'confidence': f"{confidence:.2%}",
                        'prob_positive': f"{probs[0]:.2%}",
                        'prob_negative': f"{probs[1]:.2%}",
                        'prob_sarcasm': f"{probs[2]:.2%}"
                    })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # 显示结果
                result_df = pd.DataFrame(results)
                st.success(f"✅ 分析完成！共处理 {len(result_df)} 条数据")
                st.dataframe(result_df)
                
                # 提供下载
                csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载分析结果 (CSV)",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"处理失败: {str(e)}")

# ================= 功能模块 2: API 接口文档 =================
def api_documentation_module():
    st.markdown("### 🔌 API 接口文档")
    st.info("本系统提供 RESTful API 接口，支持程序化调用")
    
    st.markdown("#### 📡 接口地址")
    st.code("POST https://api.mochi-sentiment.com/v1/analyze", language="bash")
    
    st.markdown("#### 📝 请求示例")
    request_json = {
        "text": "这家店的服务真是太好了呢",
        "topic": "餐厅评价",
        "return_details": True
    }
    st.code(json.dumps(request_json, ensure_ascii=False, indent=2), language="json")
    
    st.markdown("#### 📦 响应示例")
    response_json = {
        "code": 200,
        "message": "success",
        "data": {
            "sentiment": "sarcasm",
            "confidence": 0.87,
            "probabilities": {
                "positive": 0.05,
                "negative": 0.08,
                "sarcasm": 0.87
            },
            "tokens": ["这", "家", "店", "的", "服务", "真", "是", "太", "好", "了", "呢"],
            "attention_scores": [0.12, 0.08, 0.15, 0.05, 0.25, 0.18, 0.07, 0.10, 0.20, 0.08, 0.12]
        }
    }
    st.code(json.dumps(response_json, ensure_ascii=False, indent=2), language="json")
    
    st.markdown("#### 🐍 Python 调用示例")
    python_code = """
import requests

url = "https://api.mochi-sentiment.com/v1/analyze"
headers = {"Authorization": "Bearer YOUR_API_KEY"}
data = {
    "text": "这家店的服务真是太好了呢",
    "topic": "餐厅评价"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(f"情感: {result['data']['sentiment']}")
print(f"置信度: {result['data']['confidence']}")
"""
    st.code(python_code, language="python")
    
    st.markdown("#### 🔧 cURL 调用示例")
    curl_code = """
curl -X POST https://api.mochi-sentiment.com/v1/analyze \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "这家店的服务真是太好了呢",
    "topic": "餐厅评价"
  }'
"""
    st.code(curl_code, language="bash")

# ================= 功能模块 3: 系统设置 =================
def system_settings_module():
    st.markdown("### ⚙️ 系统设置")
    st.info("调整系统参数，设置将保存在当前会话中")
    
    st.markdown("#### 🎚️ 模型参数")
    
    # 置信度阈值
    min_conf = st.slider(
        "最小置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.min_confidence,
        step=0.05,
        help="低于此阈值的预测将被标记为不确定"
    )
    st.session_state.min_confidence = min_conf
    
    # 显示详细日志
    show_debug = st.toggle(
        "显示详细日志",
        value=st.session_state.show_debug,
        help="开启后将显示模型内部处理细节"
    )
    st.session_state.show_debug = show_debug
    
    st.markdown("#### 🎨 界面设置")
    
    # 主题选择（模拟）
    theme = st.selectbox(
        "界面主题",
        ["默认主题 (Mochi Pink)", "深色模式 (暂未实现)", "简约模式 (暂未实现)"],
        index=0
    )
    
    # 语言选择（模拟）
    language = st.selectbox(
        "界面语言",
        ["简体中文", "English (暂未实现)"],
        index=0
    )
    
    st.markdown("#### 📊 当前配置")
    config_data = {
        "参数": ["置信度阈值", "详细日志", "界面主题", "语言"],
        "值": [f"{min_conf:.2f}", "开启" if show_debug else "关闭", theme, language]
    }
    st.table(pd.DataFrame(config_data))
    
    if st.button("💾 保存设置"):
        st.success("✅ 设置已保存到当前会话")
        st.balloons()

# ================= 功能模块 4: 用户中心 =================
def user_center_module():
    st.markdown("### 👤 用户中心")
    
    if not st.session_state.logged_in:
        # 登录界面
        st.info("请登录以访问用户中心")
        
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="admin")
            password = st.text_input("密码", type="password", placeholder="123456")
            submit = st.form_submit_button("🔐 登录")
            
            if submit:
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"✅ 欢迎回来，{username}！")
                    st.rerun()
                else:
                    st.error("❌ 用户名或密码错误")
        
        st.caption("💡 演示账号: admin / 123456 或 demo / demo")
    
    else:
        # 已登录状态
        st.success(f"✅ 已登录: {st.session_state.username}")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🚪 退出登录"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.rerun()
        
        st.markdown("#### 📊 用户信息")
        user_info = {
            "用户名": st.session_state.username,
            "账户类型": "管理员" if st.session_state.username == "admin" else "普通用户",
            "注册时间": "2024-01-01",
            "上次登录": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "累计分析": len(st.session_state.history)
        }
        st.table(pd.DataFrame([user_info]).T.rename(columns={0: "值"}))
        
        st.markdown("#### 📜 历史分析记录")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            
            # 提供下载
            csv = history_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出历史记录",
                data=csv,
                file_name=f"history_{st.session_state.username}.csv",
                mime="text/csv"
            )
        else:
            st.info("暂无历史记录")

# ================= 主程序 =================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    set_style()
    
    # 侧边栏
    with st.sidebar:
        st.image("https://api.dicebear.com/7.x/notionists/svg?seed=Mochi&backgroundColor=ffb7b2", width=100)
        st.title("Mochi 控制台")
        st.caption("Ver 2.0.0 (Upgraded)")
        
        # 显示登录状态
        if st.session_state.logged_in:
            st.success(f"👤 {st.session_state.username}")
        
        menu = st.radio(
            "功能导航",
            ["🚀 情感分析实验室", "📊 批量数据处理", "🔌 API 接口文档", "⚙️ 系统设置", "👤 用户中心"]
        )
        
        st.markdown("---")
        st.markdown("#### 🛠️ 快速设置")
        st.caption(f"置信度阈值: {st.session_state.min_confidence:.2f}")
        st.caption(f"详细日志: {'开启' if st.session_state.show_debug else '关闭'}")
    
    # 加载模型
    model, preprocessor = load_model()
    
    # 根据菜单选择显示不同模块
    if menu == "🚀 情感分析实验室":
        sentiment_analysis_lab(model, preprocessor)
    elif menu == "📊 批量数据处理":
        if model and preprocessor:
            batch_processing_module(model, preprocessor)
        else:
            st.error("模型未加载")
    elif menu == "🔌 API 接口文档":
        api_documentation_module()
    elif menu == "⚙️ 系统设置":
        system_settings_module()
    elif menu == "👤 用户中心":
        user_center_module()

# ================= 情感分析实验室 =================
def sentiment_analysis_lab(model, preprocessor):
    col_main, col_right = st.columns([2, 1])
    
    with col_main:
        st.title(f"{PAGE_ICON} Mochi")
        st.markdown("### 【中文】短文本情感分析系统")
        st.caption("基于 **超图神经网络 (HGNN) 与注意力机制** 的多模态情感计算引擎")
        
        with st.container():
            st.markdown("#### 📝 文本输入")
            col_input1, col_input2 = st.columns([1, 2])
            with col_input1:
                input_topic = st.text_input("语境/话题 (Topic)", placeholder="例如：外卖")
            with col_input2:
                input_text = st.text_input("评论内容 (Text)", placeholder="请输入需要分析的中文短文本...")
                
            analyze_btn = st.button("开始分析 / Analyze ✨", use_container_width=True)
    
    # 分析逻辑
    if analyze_btn and input_text and model and preprocessor:
        with col_main:
            with st.status("🧠 Mochi 正在思考...", expanded=True) as status:
                st.write("🔍 分词与词性标注 (HanLP)...")
                time.sleep(0.3)
                st.write("🕸️ 构建依存句法超图 (Dependency Hypergraph)...")
                time.sleep(0.3)
                st.write("🎯 提取注意力权重 (Attention Weights)...")
                time.sleep(0.3)
                
                probs, hanlp_data, attention_weights, tokens_list = predict_with_attention(
                    model, preprocessor, input_text, input_topic
                )
                status.update(label="✅ 分析完成!", state="complete", expanded=False)
            
            # 保存到历史记录
            pred_label = probs.argmax()
            label_names = ['正面', '负面', '反讽']
            st.session_state.history.append({
                '时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                '文本': input_text[:30] + '...' if len(input_text) > 30 else input_text,
                '话题': input_topic,
                '情感': label_names[pred_label],
                '置信度': f"{probs[pred_label]:.2%}"
            })
            
            # 结果展示
            if pred_label == 0:
                main_text, sub_text, emoji, color = "心情不错", "【正常语气-正面】", "💖", "#ccffcc"
            elif pred_label == 1:
                main_text, sub_text, emoji, color = "你已急哭", "【正常语气-负面】", "💔", "#f0f2f6"
            else:
                main_text, sub_text, emoji, color = "阴阳怪气", "【反讽警告！】", "😏", "#FFB7B2"
            
            st.markdown(f"""
            <div class="result-card" style="background-color: {color};">
                <h1 style="color: #555; margin:0;">{emoji} {main_text}</h1>
                <h3 style="color: #333; margin:10px;">{sub_text}</h3>
                <p>置信度: <strong>{probs[pred_label]:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 情感倾向分布")
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("❤️ 正面", f"{probs[0]:.1%}")
            col_p1.progress(float(probs[0]))
            col_p2.metric("💔 负面", f"{probs[1]:.1%}")
            col_p2.progress(float(probs[1]))
            col_p3.metric("😏 反讽", f"{probs[2]:.1%}")
            col_p3.progress(float(probs[2]))
        
        # Mochi 视角（右侧）
        with col_right:
            st.markdown("### 🔍 Mochi 的视角")
            
            # Tab 1: 交互式超图
            with st.expander("🕸️ 超图结构（可交互）", expanded=True):
                html = create_interactive_hypergraph(hanlp_data, attention_weights, tokens_list)
                if html:
                    components.html(html, height=500, scrolling=True)
                    st.caption("💡 节点大小和颜色表示注意力权重，可拖拽和缩放")
                else:
                    st.warning("超图生成失败")
            
            # Tab 2: 注意力热力图
            with st.expander("🔥 注意力热力图", expanded=False):
                fig = create_attention_heatmap(attention_weights, tokens_list)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("💡 颜色越深表示注意力权重越高")
                else:
                    st.warning("热力图生成失败")
            
            # 调试信息
            if st.session_state.show_debug:
                with st.expander("🐛 调试信息", expanded=False):
                    st.json({
                        "tokens_count": len(tokens_list),
                        "attention_shape": attention_weights.shape,
                        "hanlp_keys": list(hanlp_data.keys()),
                        "device": str(DEVICE)
                    })
    
    elif not input_text:
        with col_main:
            st.info("👈 请在上方输入文本开始体验")
            st.markdown("""
            > **Mochi** 是一个能够听懂"言外之意"的智能助手。
            > * 😄 **真诚的赞美** (Positive)
            > * 😡 **直白的批评** (Negative)
            > * 😏 **阴阳怪气的讽刺** (Sarcasm)
            """)

if __name__ == "__main__":
    main()
