import streamlit as st
import torch
import os
import time
import pandas as pd
import random
import graphviz # 引入画图工具
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

# ================= 样式注入 (CSS) =================
def set_style():
    st.markdown("""
    <style>
        /* 全局字体设置 */
        html, body, [class*="css"] {
            font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif;
            color: #555555;
        }
        
        /* 标题样式 */
        h1 {
            color: #FFB7B2; /* 柔和粉 */
            font-weight: 700;
            text-shadow: 1px 1px 2px #eee;
        }
        h2, h3 {
            color: #AEC6CF; /* 雾霾蓝 */
        }
        
        /* 按钮样式 */
        div.stButton > button {
            background-color: #AEC6CF;
            color: white;
            border-radius: 20px;
            border: none;
            height: 50px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            background-color: #FFB7B2;
            transform: scale(1.02);
        }
        
        /* 结果卡片 */
        .result-card {
            background-color: #FDFD96;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
            margin-bottom: 20px;
            animation: fadeIn 0.8s;
        }
        
        /* 动画定义 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ================= 核心逻辑 =================
@st.cache_resource
def load_model():
    # 模拟加载进度
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
            return None, None
            
        model.to(DEVICE)
        model.eval()
        
        status_text.empty()
        progress_bar.empty()
        return model, preprocessor
    except Exception as e:
        st.error(f"Mochi 生病了: {str(e)}")
        return None, None

def predict(model, preprocessor, text, topic):
    if not topic or not topic.strip():
        topic = "网友评论"
    bert_tokens = preprocessor.process_text_pair(text, topic, 256)
    hanlp_result = preprocessor.process_text_with_hanlp(text, topic)
    hypergraph_matrix = preprocessor._create_single_hypergraph_matrix(hanlp_result, 256, 50)
    
    input_ids = bert_tokens['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = bert_tokens['attention_mask'].unsqueeze(0).to(DEVICE)
    token_type_ids = bert_tokens['token_type_ids'].unsqueeze(0).to(DEVICE)
    hg_mat_tensor = torch.tensor(hypergraph_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, hg_mat_tensor, token_type_ids)
        probs = F.softmax(logits, dim=1)
        
    return probs.cpu().numpy()[0], hanlp_result

# ================= 页面主结构 =================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
set_style()

# 侧边栏
with st.sidebar:
    st.image("https://api.dicebear.com/7.x/notionists/svg?seed=Mochi&backgroundColor=ffb7b2", width=100)
    st.title("Mochi 控制台")
    st.caption("Ver 1.0.0 (Thesis Build)")
    menu = st.radio("功能导航", ["🚀 情感分析实验室", "📊 批量数据处理", "🔌 API 接口文档", "⚙️ 系统设置", "👤 用户中心"])
    if menu != "🚀 情感分析实验室":
        st.info("⚠️ 该模块仅对 VIP 开放，演示版请使用【情感分析实验室】。")
    st.markdown("---")
    st.markdown("#### 🛠️ 模型参数")
    st.slider("置信度阈值", 0.0, 1.0, 0.85)
    st.toggle("开启句法去噪", value=True)

# 主界面
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

# 逻辑处理
if analyze_btn and input_text:
    model, preprocessor = load_model()
    if model:
        with col_main:
            with st.status("🧠 Mochi 正在思考...", expanded=True) as status:
                st.write("🔍 分词与词性标注 (HanLP)...")
                time.sleep(0.3)
                st.write("🕸️ 构建依存句法超图 (Dependency Hypergraph)...")
                time.sleep(0.3)
                probs, hanlp_data = predict(model, preprocessor, input_text, input_topic)
                status.update(label="✅ 分析完成!", state="complete", expanded=False)
            
            # --- 结果文案定制 ---
            pred_label = probs.argmax()
            
            if pred_label == 0:
                main_text = "心情不错"
                sub_text = "【正常语气-正面】 (Positive)"
                emoji = "💖"
                color = "#ccffcc" 
            elif pred_label == 1:
                main_text = "你已急哭" 
                sub_text = "【正常语气-负面】 (Negative)"
                emoji = "💔"
                color = "#f0f2f6" 
            else:
                main_text = "阴阳怪气"
                sub_text = "【反讽警告！】 (Sarcasm)"
                emoji = "😏"
                color = "#FFB7B2" 
            
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

        # --- Mochi 视角 (Graphviz 超图可视化 - 修复版) ---
        with col_right:
            st.markdown("### 🔍 Mochi 的视角")
            st.info("这是 HGNN 眼中的句子结构（依存句法超图）")
            
            try:
                # 1. 严格使用你环境返回的键名
                words = hanlp_data.get('tokens') # 之前是 tok
                deps = hanlp_data.get('dependencies') # 之前是 dep
                
                # 有些 HanLP 版本会嵌套一层列表，做个防御性处理
                if words and isinstance(words[0], list):
                    words = words[0]
                if deps and isinstance(deps[0], list) and isinstance(deps[0][0], tuple) == False:
                     # 这种情况下 deps 可能是 [[head, rel], [head, rel]...] 或者是 [[[head, rel]...]]
                    if isinstance(deps[0][0], list):
                        deps = deps[0]

                if words and deps:
                    # 使用 Graphviz 画出超图结构
                    graph = graphviz.Digraph()
                    graph.attr(rankdir='LR', size='8,5', bgcolor='transparent')
                    graph.attr('node', shape='ellipse', style='filled', fillcolor='#f0f2f6', fontname='Microsoft YaHei')
                    graph.attr('edge', fontname='Microsoft YaHei', fontsize='10', color='#AEC6CF')
                    
                    # 先画所有节点
                    for i, w in enumerate(words):
                        graph.node(str(i), w)
                    
                    # 再画边
                    edge_count = 0
                    for i, item in enumerate(deps):
                        # 不同的 HanLP 版本，deps 的格式不同，这里做兼容
                        # 格式A: (head_word, relation) -> 字符串
                        # 格式B: (head_index, relation) -> 数字索引 (最常见)
                        head = None
                        rel = None
                        
                        if isinstance(item, (list, tuple)):
                            if len(item) >= 2:
                                head = item[0]
                                rel = item[1]
                        
                        # 如果 head 是索引 (数字)
                        if isinstance(head, int):
                            head_idx = head - 1 # HanLP 索引通常从1开始，Graphviz从0开始
                            if head_idx >= 0 and head_idx < len(words):
                                # 过滤掉不重要的边，只展示核心句法，避免图太乱
                                if rel in ['nsubj', 'obj', 'dobj', 'advmod', 'root', 'att', 'punct']:
                                    graph.edge(str(i), str(head_idx), label=rel)
                                    edge_count += 1
                                    
                        # 如果 head 已经是词 (字符串)，说明 HanLP 直接返回了词
                        elif isinstance(head, str):
                             # 这种比较难对应索引，简化处理，直接画虚节点（较少见）
                             pass

                    if edge_count > 0:
                        st.graphviz_chart(graph)
                        st.caption(f"✨ 依存句法可视化成功！共捕获 {edge_count} 条核心语义超边。")
                        st.markdown("""
                        > **图例解释：**
                        > * `nsubj`: 名词性主语 (谁?)
                        > * `dobj`: 直接宾语 (什么?)
                        > * `advmod`: 状语修饰 (怎么样?)
                        """)
                    else:
                        st.warning("句子结构简单或解析格式不匹配，未画出连线。")
                        st.write(f"Raw Deps: {deps[:2]}") # 调试用
                else:
                    st.error("数据解析失败，键名匹配但内容为空。")
                    st.write(f"Available keys: {list(hanlp_data.keys())}")
                        
            except Exception as e:
                st.error("可视化渲染遭遇未知错误")
                st.caption(f"Error Details: {str(e)}")
            
            st.markdown("---")
            st.caption("Generated by Mochi Engine v1.0")

else:
    # 欢迎页
    with col_main:
        st.info("👈 请在左侧选择模式，或直接在上方输入文本开始体验。")
        st.markdown("""
        > **Mochi** 是一个能够听懂“言外之意”的智能助手。
        > 它可以区分：
        > * 😄 **真诚的赞美** (Happy)
        > * 😡 **直白的批评** (Angry)
        > * 😏 **阴阳怪气的讽刺** (Sarcasm - 最难的！)
        """)