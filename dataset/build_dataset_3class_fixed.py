"""
数据清洗脚本 - 三分类版本 (修复版)
修复内容：
1. Topic 填充：防止模型通过"有无 Topic"作弊
2. 降采样平衡：解决 3.56% 的极端不平衡
3. 文本长度过滤：去除过短和过长的文本
"""

import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter


# ================= 配置区 =================
# 降采样目标数量：让 Label 0/1 的数量接近 Label 2
# Label 2 大约有 4800 条，我们把 0 和 1 也限制在 6000 条左右
TARGET_SAMPLE_NUM = 6000

# 文本长度过滤
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 200
# ==========================================


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    if not os.path.exists(file_path):
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except:
                    pass
    return data


def load_json(file_path: str) -> List[Dict]:
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except:
            return []


def load_csv(file_path: str) -> List[Dict]:
    """加载 CSV 文件"""
    data = []
    if not os.path.exists(file_path):
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_random_topic(source_type: str) -> str:
    """
    【关键修复】生成随机通用 Topic，防止模型根据 Topic 是否为空来作弊
    
    Args:
        source_type: 'chn' 或 'weibo'
        
    Returns:
        随机选择的通用 Topic
    """
    if source_type == 'chn':
        topics = [
            "用户评价", "购物心得", "酒店入住体验", 
            "产品反馈", "服务点评", "买家秀",
            "商品评论", "消费体验", "使用感受"
        ]
    else:  # weibo
        topics = [
            "微博热搜", "心情记录", "每日吐槽", 
            "网友热议", "生活点滴", "吃瓜现场",
            "今日话题", "随手一拍", "日常分享"
        ]
    return random.choice(topics)


def process_chnsenticorp(raw_dir: str = "raw_source/ChnSentiCorp") -> List[Dict]:
    """
    处理 ChnSentiCorp 数据
    - 重新标注：原标签 1(好评) -> Label 0(正面), 原标签 0(差评) -> Label 1(负面)
    - 填充 Topic：防止数据泄露
    - 降采样：控制数量在 TARGET_SAMPLE_NUM
    """
    print("=" * 60)
    print(f"📥 处理 ChnSentiCorp (目标采样: {TARGET_SAMPLE_NUM} 条)...")
    print("=" * 60)
    
    all_data = []
    
    # 读取所有数据
    temp_data = []
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(raw_dir, f"{split}.jsonl")
        temp_data.extend(load_jsonl(file_path))
    
    print(f"原始数据: {len(temp_data)} 条")
    
    # 随机打乱
    random.shuffle(temp_data)
    
    count_0, count_1 = 0, 0
    
    for item in temp_data:
        text = item.get('text', '').strip()
        original_label = item.get('label', 0)
        
        # 文本长度过滤
        if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
            continue
        
        # 重新标注：0: 正面 (原label 1), 1: 负面 (原label 0)
        new_label = 0 if original_label == 1 else 1
        
        # 降采样控制
        if new_label == 0 and count_0 >= TARGET_SAMPLE_NUM:
            continue
        if new_label == 1 and count_1 >= TARGET_SAMPLE_NUM:
            continue
        
        all_data.append({
            'text': text,
            'topic': get_random_topic('chn'),  # 【修复】填充 Topic
            'label': new_label,
            'source': 'ChnSentiCorp'
        })
        
        if new_label == 0:
            count_0 += 1
        else:
            count_1 += 1
        
        # 两个类别都达到目标数量，停止
        if count_0 >= TARGET_SAMPLE_NUM and count_1 >= TARGET_SAMPLE_NUM:
            break
    
    print(f"✅ ChnSentiCorp 处理完成:")
    print(f"   - Label 0 (正面): {count_0} 条")
    print(f"   - Label 1 (负面): {count_1} 条")
    
    return all_data


def process_weibo(raw_file: str = "raw_source/Weibo/weibo_senti_100k.csv") -> List[Dict]:
    """
    处理 Weibo 数据
    - 重新标注：原标签 1(正向) -> Label 0(正面), 原标签 0(负向) -> Label 1(负面)
    - 填充 Topic：防止数据泄露
    - 降采样：控制数量在 TARGET_SAMPLE_NUM
    """
    print("\n" + "=" * 60)
    print(f"📥 处理 Weibo (目标采样: {TARGET_SAMPLE_NUM} 条)...")
    print("=" * 60)
    
    raw_data = load_csv(raw_file)
    print(f"原始数据: {len(raw_data)} 条")
    
    # 随机打乱
    random.shuffle(raw_data)
    
    processed_data = []
    count_0, count_1 = 0, 0
    
    for item in raw_data:
        text = item.get('review', '').strip()
        
        try:
            original_label = int(item.get('label', 0))
        except:
            continue
        
        # 文本长度过滤（微博限制 140 字）
        if len(text) < MIN_TEXT_LENGTH or len(text) > 140:
            continue
        
        # 重新标注：0: 正面 (原label 1), 1: 负面 (原label 0)
        new_label = 0 if original_label == 1 else 1
        
        # 降采样控制
        if new_label == 0 and count_0 >= TARGET_SAMPLE_NUM:
            continue
        if new_label == 1 and count_1 >= TARGET_SAMPLE_NUM:
            continue
        
        processed_data.append({
            'text': text,
            'topic': get_random_topic('weibo'),  # 【修复】填充 Topic
            'label': new_label,
            'source': 'Weibo'
        })
        
        if new_label == 0:
            count_0 += 1
        else:
            count_1 += 1
        
        # 两个类别都达到目标数量，停止
        if count_0 >= TARGET_SAMPLE_NUM and count_1 >= TARGET_SAMPLE_NUM:
            break
    
    print(f"✅ Weibo 处理完成:")
    print(f"   - Label 0 (正面): {count_0} 条")
    print(f"   - Label 1 (负面): {count_1} 条")
    
    return processed_data


def process_tosarcasm(raw_dir: str = "raw_source/ToSarcasm") -> List[Dict]:
    """
    处理 ToSarcasm 数据
    - 全部标注为 Label 2 (反讽)
    - 保留真实的 Topic (新闻标题)
    """
    print("\n" + "=" * 60)
    print("📥 处理 ToSarcasm (全部保留)...")
    print("=" * 60)
    
    all_data = []
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(raw_dir, f"{split}.json")
        for item in load_json(file_path):
            text = item.get('text', '').strip()
            topic = item.get('topic', '').strip()
            
            if text:
                all_data.append({
                    'text': text,
                    'topic': topic,  # 保留真实新闻标题
                    'label': 2,  # 阴阳怪气
                    'source': 'ToSarcasm'
                })
    
    print(f"✅ ToSarcasm 处理完成: {len(all_data)} 条")
    print(f"   - Label 2 (反讽): {len(all_data)} 条")
    
    return all_data


def save_and_split(data: List[Dict], output_dir: str, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    保存并划分数据集
    
    Args:
        data: 所有数据
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    print("\n" + "=" * 60)
    print("📊 划分数据集...")
    print("=" * 60)
    
    # 打乱数据
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    def _save(d, name):
        """保存数据并打印分布"""
        # 移除 source 字段
        clean_data = []
        for item in d:
            clean_data.append({
                'text': item['text'],
                'topic': item['topic'],
                'label': item['label']
            })
        
        path = os.path.join(output_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
        
        # 打印分布
        cnt = Counter([x['label'] for x in d])
        total_count = len(d)
        print(f"\n💾 {name}:")
        print(f"   总数: {total_count}")
        print(f"   Label 0 (正面): {cnt[0]:5d} ({cnt[0]/total_count*100:5.2f}%)")
        print(f"   Label 1 (负面): {cnt[1]:5d} ({cnt[1]/total_count*100:5.2f}%)")
        print(f"   Label 2 (反讽): {cnt[2]:5d} ({cnt[2]/total_count*100:5.2f}%)")
    
    _save(train_data, "train.json")
    _save(val_data, "dev.json")
    _save(test_data, "test.json")


def main():
    """主函数"""
    print("\n" + "🔧" * 30)
    print("数据清洗脚本 - 三分类版本 (修复版)")
    print("🔧" * 30 + "\n")
    
    print("🔴 修复内容:")
    print("   1. Topic 填充：防止模型通过'有无 Topic'作弊")
    print("   2. 降采样平衡：解决 3.56% 的极端不平衡")
    print("   3. 文本长度过滤：去除过短和过长的文本")
    print()
    
    # 设置随机种子
    random.seed(42)
    
    # 切换到 dataset 目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📂 工作目录: {os.getcwd()}\n")
    
    # 1. 获取数据 (带降采样和 Topic 填充)
    d1 = process_chnsenticorp()
    d2 = process_weibo()
    d3 = process_tosarcasm()
    
    # 2. 合并
    final_data = d1 + d2 + d3
    
    print("\n" + "=" * 60)
    print("📊 最终数据分布")
    print("=" * 60)
    
    cnt = Counter([x['label'] for x in final_data])
    total = len(final_data)
    
    print(f"总数据量: {total} 条\n")
    print(f"Label 0 (正面): {cnt[0]:5d} ({cnt[0]/total*100:5.2f}%)")
    print(f"Label 1 (负面): {cnt[1]:5d} ({cnt[1]/total*100:5.2f}%)")
    print(f"Label 2 (反讽): {cnt[2]:5d} ({cnt[2]/total*100:5.2f}%)")
    print("=" * 60)
    
    # 检查平衡性
    min_count = min(cnt.values())
    max_count = max(cnt.values())
    ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n📈 类别平衡性:")
    print(f"   最小类别: {min_count} 条")
    print(f"   最大类别: {max_count} 条")
    print(f"   不平衡比例: {ratio:.2f}:1")
    
    if ratio < 2.0:
        print("   ✅ 类别相对平衡（< 2:1）")
    elif ratio < 5.0:
        print("   ⚠️ 类别略有不平衡（2:1 ~ 5:1）")
    else:
        print("   ❌ 类别严重不平衡（> 5:1）")
    
    # 3. 保存
    save_and_split(final_data, "processed")
    
    print("\n" + "=" * 60)
    print("✅ 数据处理完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {os.path.abspath('processed')}")
    print("\n⚠️ 重要提示:")
    print("   1. 所有数据都有 Topic（防止数据泄露）")
    print("   2. 类别已平衡（降采样）")
    print("   3. 文本长度已过滤")
    print("\n💡 下一步:")
    print("   cd ..")
    print("   python main.py --dataset_dir dataset/processed --num_classes 3")


if __name__ == "__main__":
    main()
