"""
inference.py
三分类最终演示版：正面 vs 负面 vs 阴阳怪气
"""
import torch
from data_preprocess import DataPreprocessor
from model import BertHGNNModel
import os

# ================= 配置区 =================
# 必须指向你刚训练好的那个文件夹
MODEL_DIR = "checkpoints/3class_final" 
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
BERT_MODEL = "bert-base-chinese"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================

def load_trained_model():
    print("⏳ 正在加载预处理工具...")
    preprocessor = DataPreprocessor(bert_model_name=BERT_MODEL)
    
    print(f"⏳ 正在加载模型: {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

    # 注意：这里的参数必须和你 main.py 里的完全一致
    model = BertHGNNModel(
        bert_model_name=BERT_MODEL,
        hgnn_hidden_dims=[256, 128], 
        num_attention_heads=4,
        num_classes=3,  # <--- 关键：现在是 3 分类
        dropout=0
    )
    
    # 加载权重
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    print("✅ 模型加载成功！准备起飞！")
    return model, preprocessor

def predict(model, preprocessor, text, topic=""):
    # 1. 预处理
    # 如果用户没输 Topic，我们给一个中性的 Topic，防止模型因为空 Topic 乱猜
    if not topic: 
        topic = "网友评论" 
        
    bert_tokens = preprocessor.process_text_pair(text, topic, MAX_LEN)
    hanlp_result = preprocessor.process_text_with_hanlp(text, topic)
    
    # 2. 构建超图
    max_edges = 50 
    hypergraph_matrix = preprocessor._create_single_hypergraph_matrix(
        hanlp_result, MAX_LEN, max_edges
    )
    
    # 3. 转 Tensor
    input_ids = bert_tokens['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = bert_tokens['attention_mask'].unsqueeze(0).to(DEVICE)
    token_type_ids = bert_tokens['token_type_ids'].unsqueeze(0).to(DEVICE)
    hg_mat_tensor = torch.tensor(hypergraph_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 4. 预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask, hg_mat_tensor, token_type_ids)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        
        # 获取每个类别的概率
        prob_0 = probs[0][0].item() # 正面
        prob_1 = probs[0][1].item() # 负面
        prob_2 = probs[0][2].item() # 反讽
        
    return pred_label, (prob_0, prob_1, prob_2)

if __name__ == "__main__":
    try:
        model, preprocessor = load_trained_model()
        
        print("\n" + "="*40)
        print("🤖 AI 情感与反讽分析终端 (H100驱动版)")
        print("标签定义: [0]正面夸奖  [1]正常差评  [2]阴阳怪气")
        print("输入 'q' 退出")
        print("="*40 + "\n")
        
        while True:
            print("-" * 30)
            topic = input("场景/标题 【可回车跳过】: ").strip()
            if topic == 'q': break
            
            text = input("文本内容: ").strip()
            if text == 'q': break
            if not text: continue
            
            try:
                label, probs = predict(model, preprocessor, text, topic)
                
                # 结果美化
                if label == 0:
                    result = "❤️  心情不错 【正常语气-正面】"
                elif label == 1:
                    result = "💔  你已急哭 【正常语气-负面】"
                else:
                    result = "😏  阴阳怪气 【反讽警告!】"
                
                print(f"\n分析结论: {result}")
                print(f"详细概率: 正面[{probs[0]:.2%}]  负面[{probs[1]:.2%}]  反讽[{probs[2]:.2%}]")
                
                # 强撑检测逻辑（彩蛋）
                if label == 0 and probs[2] > 0.3:
                    print("💡 洞察: 听起来心情不错，但有一丝强撑的味道...")
                    
            except Exception as e:
                print(f"推理出错: {e}")
    except Exception as e:
        print(f"启动失败: {e}")