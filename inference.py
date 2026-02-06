"""
inference.py
加载训练好的模型，进行单句测试
"""
import torch
from data_preprocess import DataPreprocessor
from model import BertHGNNModel
import os

# ================= 配置 =================
MODEL_PATH = "checkpoints/best_model.pth"  # 确保这个路径对
BERT_MODEL = "bert-base-chinese"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================

def load_trained_model():
    print("正在加载预处理工具...")
    preprocessor = DataPreprocessor(bert_model_name=BERT_MODEL)
    
    print(f"正在加载模型: {MODEL_PATH} ...")
    # 注意：这里的参数必须和你训练时 main.py 里的完全一致！
    # 如果你训练时用的 hidden_dims=[256, 128]，这里也要改
    model = BertHGNNModel(
        bert_model_name=BERT_MODEL,
        hgnn_hidden_dims=[256, 128],  # <--- 关键：和你训练参数保持一致
        num_attention_heads=4,        # <--- 关键：和你训练参数保持一致
        num_classes=2,
        dropout=0
    )
    
    # 加载权重
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # 兼容处理：有些保存方式会多一层 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    print("模型加载成功！")
    return model, preprocessor

def predict(model, preprocessor, text, topic):
    # 1. 预处理
    bert_tokens = preprocessor.process_text_pair(text, topic, MAX_LEN)
    hanlp_result = preprocessor.process_text_with_hanlp(text, topic)
    
    # 2. 构建超图 (这里有些繁琐，因为推理时是单条数据)
    # 我们借用一下 DataPreprocessor 里的私有方法，虽然不优雅但管用
    max_edges = 50 # 推理时估算一个即可
    hypergraph_matrix = preprocessor._create_single_hypergraph_matrix(
        hanlp_result, MAX_LEN, max_edges
    )
    
    # 3. 转 Tensor 并扩充 Batch 维度
    input_ids = bert_tokens['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = bert_tokens['attention_mask'].unsqueeze(0).to(DEVICE)
    token_type_ids = bert_tokens['token_type_ids'].unsqueeze(0).to(DEVICE)
    hg_mat_tensor = torch.tensor(hypergraph_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 4. 预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask, hg_mat_tensor, token_type_ids)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
        
    return pred_label, confidence

if __name__ == "__main__":
    model, preprocessor = load_trained_model()
    
    print("\n" + "="*30)
    print("AI 阴阳怪气检测器 (MVP版)")
    print("输入 'q' 退出")
    print("="*30 + "\n")
    
    while True:
        topic = input("请输入背景/新闻标题 (直接回车可跳过): ").strip()
        if topic == 'q': break
        text = input("请输入评论内容: ").strip()
        if text == 'q': break
        
        if not text: continue
        
        try:
            label, conf = predict(model, preprocessor, text, topic)
            result = "⚠️ 阴阳怪气" if label == 1 else "✅ 正常评论"
            print(f"\n分析结果: {result}")
            print(f"置信度:   {conf:.2%}\n")
            print("-" * 30)
        except Exception as e:
            print(f"出错了: {e}")