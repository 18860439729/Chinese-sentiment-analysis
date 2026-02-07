"""
数据清洗脚本 - 三分类版本
实现带情感极性的反讽检测

标签体系：
- Label 0: 正常-正面 (Normal-Positive) - 真正的夸奖
- Label 1: 正常-负面 (Normal-Negative) - 真正的批评
- Label 2: 阴阳怪气 (Sarcastic) - 反讽/讽刺
"""

import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {line_num} 行 JSON 解析错误: {e}")
                    continue
    
    return data


def load_json(file_path: str) -> List[Dict]:
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析错误: {e}")
            return []


def load_csv(file_path: str) -> List[Dict]:
    """加载 CSV 文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


def process_chnsenticorp(raw_dir: str = "raw_source/ChnSentiCorp") -> List[Dict]:
    """
    处理 ChnSentiCorp 数据
    原标签 0 -> Label 1 (负面)
    原标签 1 -> Label 0 (正面)
    
    Returns:
        处理后的数据列表
    """
    print("=" * 60)
    print("📥 处理 ChnSentiCorp 数据集...")
    print("=" * 60)
    
    all_data = []
    
    # 处理所有分割
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(raw_dir, f"{split}.jsonl")
        if not os.path.exists(file_path):
            continue
        
        raw_data = load_jsonl(file_path)
        print(f"✅ 加载 {split}: {len(raw_data)} 样本")
        
        for item in raw_data:
            text = item.get('text', '').strip()
            original_label = item.get('label', 0)
            
            if text:
                # 重新标注：原标签 1(好评) -> Label 0(正面)
                #          原标签 0(差评) -> Label 1(负面)
                new_label = 0 if original_label == 1 else 1
                
                all_data.append({
                    'text': text,
                    'topic': '',
                    'label': new_label,
                    'source': 'ChnSentiCorp'
                })
    
    print(f"✅ ChnSentiCorp 处理完成: {len(all_data)} 样本")
    
    # 统计标签分布
    label_counts = Counter(item['label'] for item in all_data)
    print(f"   - Label 0 (正面): {label_counts[0]} 条")
    print(f"   - Label 1 (负面): {label_counts[1]} 条")
    
    return all_data


def process_weibo(raw_file: str = "raw_source/Weibo/weibo_senti_100k.csv") -> List[Dict]:
    """
    处理 Weibo 数据
    原标签 1 -> Label 0 (正面)
    原标签 0 -> Label 1 (负面)
    
    Returns:
        处理后的数据列表
    """
    print("\n" + "=" * 60)
    print("📥 处理 Weibo 数据集...")
    print("=" * 60)
    
    if not os.path.exists(raw_file):
        print(f"⚠️ Weibo 数据文件不存在: {raw_file}")
        return []
    
    raw_data = load_csv(raw_file)
    print(f"✅ 加载 Weibo 数据: {len(raw_data)} 样本")
    
    processed_data = []
    for item in raw_data:
        text = item.get('review', '').strip()
        original_label = int(item.get('label', 0))
        
        if text:
            # 重新标注：原标签 1(正向) -> Label 0(正面)
            #          原标签 0(负向) -> Label 1(负面)
            new_label = 0 if original_label == 1 else 1
            
            processed_data.append({
                'text': text,
                'topic': '',
                'label': new_label,
                'source': 'Weibo'
            })
    
    print(f"✅ Weibo 处理完成: {len(processed_data)} 样本")
    
    # 统计标签分布
    label_counts = Counter(item['label'] for item in processed_data)
    print(f"   - Label 0 (正面): {label_counts[0]} 条")
    print(f"   - Label 1 (负面): {label_counts[1]} 条")
    
    return processed_data


def process_tosarcasm(raw_dir: str = "raw_source/ToSarcasm") -> List[Dict]:
    """
    处理 ToSarcasm 数据
    所有数据 -> Label 2 (反讽)
    
    Returns:
        处理后的数据列表
    """
    print("\n" + "=" * 60)
    print("📥 处理 ToSarcasm 数据集...")
    print("=" * 60)
    
    all_data = []
    
    # 处理所有分割
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(raw_dir, f"{split}.json")
        if not os.path.exists(file_path):
            continue
        
        raw_data = load_json(file_path)
        print(f"✅ 加载 {split}: {len(raw_data)} 样本")
        
        for item in raw_data:
            text = item.get('text', '').strip()
            topic = item.get('topic', '').strip()
            
            if text:
                # 所有 ToSarcasm 数据标注为 Label 2 (反讽)
                all_data.append({
                    'text': text,
                    'topic': topic,
                    'label': 2,
                    'source': 'ToSarcasm'
                })
    
    print(f"✅ ToSarcasm 处理完成: {len(all_data)} 样本")
    print(f"   - Label 2 (反讽): {len(all_data)} 条")
    
    return all_data


def balance_and_split_data(data: List[Dict], 
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    平衡数据并划分训练/验证/测试集
    
    Args:
        data: 所有数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (train_data, val_data, test_data)
    """
    print("\n" + "=" * 60)
    print("📊 数据平衡和划分...")
    print("=" * 60)
    
    # 按标签分组
    label_groups = {0: [], 1: [], 2: []}
    for item in data:
        label_groups[item['label']].append(item)
    
    print(f"原始数据分布:")
    for label, items in label_groups.items():
        label_name = ['正面', '负面', '反讽'][label]
        print(f"   Label {label} ({label_name}): {len(items)} 条")
    
    # 打乱每个标签的数据
    for label in label_groups:
        random.shuffle(label_groups[label])
    
    # 划分每个标签的数据
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in label_groups.items():
        total = len(items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    # 打乱混合后的数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"\n划分后数据集:")
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(val_data)} 条")
    print(f"   测试集: {len(test_data)} 条")
    
    # 统计每个数据集的标签分布
    for dataset_name, dataset in [('训练集', train_data), ('验证集', val_data), ('测试集', test_data)]:
        label_counts = Counter(item['label'] for item in dataset)
        print(f"\n{dataset_name}标签分布:")
        for label in [0, 1, 2]:
            label_name = ['正面', '负面', '反讽'][label]
            count = label_counts[label]
            percentage = count / len(dataset) * 100 if dataset else 0
            print(f"   Label {label} ({label_name}): {count} ({percentage:.2f}%)")
    
    return train_data, val_data, test_data


def save_json(data: List[Dict], output_file: str):
    """保存为 JSON 格式"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 移除 source 字段（仅用于调试）
    clean_data = []
    for item in data:
        clean_data.append({
            'text': item['text'],
            'topic': item['topic'],
            'label': item['label']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存到: {output_file}")


def main():
    """主函数"""
    print("\n" + "🔧" * 30)
    print("数据清洗脚本 - 三分类版本")
    print("🔧" * 30 + "\n")
    
    print("📋 标签体系:")
    print("   Label 0: 正常-正面 (Normal-Positive)")
    print("   Label 1: 正常-负面 (Normal-Negative)")
    print("   Label 2: 阴阳怪气 (Sarcastic)")
    print()
    
    # 切换到 dataset 目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📂 工作目录: {os.getcwd()}\n")
    
    # 设置随机种子
    random.seed(42)
    
    # 处理各个数据集
    chnsenticorp_data = process_chnsenticorp()
    weibo_data = process_weibo()
    tosarcasm_data = process_tosarcasm()
    
    # 合并所有数据
    all_data = chnsenticorp_data + weibo_data + tosarcasm_data
    
    print("\n" + "=" * 60)
    print(f"📊 总数据量: {len(all_data)} 条")
    print("=" * 60)
    
    # 平衡和划分数据
    train_data, val_data, test_data = balance_and_split_data(all_data)
    
    # 创建输出目录
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    print("\n" + "=" * 60)
    print("💾 保存处理后的数据...")
    print("=" * 60)
    
    save_json(train_data, os.path.join(output_dir, "train.json"))
    save_json(val_data, os.path.join(output_dir, "dev.json"))
    save_json(test_data, os.path.join(output_dir, "test.json"))
    
    print("\n" + "=" * 60)
    print("✅ 数据处理完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {os.path.abspath(output_dir)}")
    print("\n⚠️ 重要提示:")
    print("   模型需要修改为 3 分类:")
    print("   - 在 main.py 中设置 --num_classes 3")
    print("   - 或修改 model.py 中的 num_classes 默认值为 3")
    print("\n💡 下一步:")
    print("   cd ..")
    print("   python main.py --dataset_dir dataset/processed --num_classes 3")


if __name__ == "__main__":
    main()
