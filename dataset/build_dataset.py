"""
数据集处理脚本
将 raw_source 的原始数据转换为 processed 的训练数据
支持 ChnSentiCorp 和 Weibo 数据集
"""

import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import random

def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载 JSONL 文件
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        数据列表
    """
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON 解析错误: {e}")
                    continue
    
    return data


def load_csv(file_path: str) -> List[Dict]:
    """
    加载 CSV 文件
    
    Args:
        file_path: CSV 文件路径
        
    Returns:
        数据列表
    """
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


def process_chnsenticorp(raw_dir: str = "raw_source/ChnSentiCorp") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    处理 ChnSentiCorp 情感分析数据
    
    Args:
        raw_dir: 原始数据目录
        
    Returns:
        (train_data, dev_data, test_data)
    """
    print("=" * 60)
    print("📥 处理 ChnSentiCorp 数据集...")
    print("=" * 60)
    
    train_data = []
    dev_data = []
    test_data = []
    
    # 加载训练集
    train_file = os.path.join(raw_dir, "train.jsonl")
    if os.path.exists(train_file):
        raw_train = load_jsonl(train_file)
        print(f"✅ 加载训练集: {len(raw_train)} 样本")
        
        for item in raw_train:
            # 提取文本和标签
            text = item.get('text', '').strip()
            label = item.get('label', 0)
            
            if text:  # 过滤空文本
                train_data.append({
                    'text': text,
                    'topic': '',  # ChnSentiCorp 没有 topic
                    'label': str(label)
                })
    
    # 加载验证集
    val_file = os.path.join(raw_dir, "validation.jsonl")
    if os.path.exists(val_file):
        raw_val = load_jsonl(val_file)
        print(f"✅ 加载验证集: {len(raw_val)} 样本")
        
        for item in raw_val:
            text = item.get('text', '').strip()
            label = item.get('label', 0)
            
            if text:
                dev_data.append({
                    'text': text,
                    'topic': '',
                    'label': str(label)
                })
    
    # 加载测试集
    test_file = os.path.join(raw_dir, "test.jsonl")
    if os.path.exists(test_file):
        raw_test = load_jsonl(test_file)
        print(f"✅ 加载测试集: {len(raw_test)} 样本")
        
        for item in raw_test:
            text = item.get('text', '').strip()
            label = item.get('label', 0)
            
            if text:
                test_data.append({
                    'text': text,
                    'topic': '',
                    'label': str(label)
                })
    
    print(f"✅ ChnSentiCorp 处理完成:")
    print(f"   - 训练集: {len(train_data)} 样本")
    print(f"   - 验证集: {len(dev_data)} 样本")
    print(f"   - 测试集: {len(test_data)} 样本")
    
    return train_data, dev_data, test_data


def process_weibo(raw_file: str = "raw_source/Weibo/weibo_senti_100k.csv", 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    处理 Weibo 情感分析数据
    
    Args:
        raw_file: 原始 CSV 文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (train_data, dev_data, test_data)
    """
    print("\n" + "=" * 60)
    print("📥 处理 Weibo 数据集...")
    print("=" * 60)
    
    if not os.path.exists(raw_file):
        print(f"⚠️ Weibo 数据文件不存在: {raw_file}")
        return [], [], []
    
    # 加载 CSV 数据
    raw_data = load_csv(raw_file)
    print(f"✅ 加载 Weibo 数据: {len(raw_data)} 样本")
    
    # 转换格式并过滤
    processed_data = []
    for item in raw_data:
        text = item.get('review', '').strip()
        label = item.get('label', '0')
        
        if text:  # 过滤空文本
            processed_data.append({
                'text': text,
                'topic': '',
                'label': str(label)
            })
    
    print(f"✅ 有效样本: {len(processed_data)} 条")
    
    # 打乱数据
    random.shuffle(processed_data)
    
    # 划分数据集
    total = len(processed_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = processed_data[:train_end]
    dev_data = processed_data[train_end:val_end]
    test_data = processed_data[val_end:]
    
    print(f"✅ Weibo 处理完成:")
    print(f"   - 训练集: {len(train_data)} 样本")
    print(f"   - 验证集: {len(dev_data)} 样本")
    print(f"   - 测试集: {len(test_data)} 样本")
    
    return train_data, dev_data, test_data


def save_json(data: List[Dict], output_file: str):
    """
    保存为 JSON 格式
    
    Args:
        data: 数据列表
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存到: {output_file}")


def main():
    """主函数"""
    print("\n" + "🔧" * 30)
    print("数据集处理工具")
    print("🔧" * 30 + "\n")
    
    # 切换到 dataset 目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📂 工作目录: {os.getcwd()}")
    
    # 设置随机种子
    random.seed(42)
    
    # 处理 ChnSentiCorp
    train_data, dev_data, test_data = process_chnsenticorp()
    
    # 处理 Weibo（可选，作为补充数据）
    try:
        weibo_train, weibo_dev, weibo_test = process_weibo()
        
        if weibo_train:
            print(f"\n📊 混合 Weibo 数据...")
            train_data.extend(weibo_train)
            dev_data.extend(weibo_dev)
            test_data.extend(weibo_test)
            
            # 打乱混合后的数据
            random.shuffle(train_data)
            random.shuffle(dev_data)
            random.shuffle(test_data)
            
            print(f"✅ 混合后总样本数:")
            print(f"   - 训练集: {len(train_data)}")
            print(f"   - 验证集: {len(dev_data)}")
            print(f"   - 测试集: {len(test_data)}")
    except Exception as e:
        print(f"⚠️ Weibo 处理失败，跳过: {e}")
    
    # 创建输出目录
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    print("\n" + "=" * 60)
    print("💾 保存处理后的数据...")
    print("=" * 60)
    
    save_json(train_data, os.path.join(output_dir, "train.json"))
    save_json(dev_data, os.path.join(output_dir, "dev.json"))
    save_json(test_data, os.path.join(output_dir, "test.json"))
    
    # 统计信息
    print("\n" + "=" * 60)
    print("📊 数据集统计:")
    print("=" * 60)
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(dev_data)} 样本")
    print(f"测试集: {len(test_data)} 样本")
    
    # 标签分布
    if train_data:
        label_counts = {}
        for item in train_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n训练集标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  Label {label}: {count} ({count/len(train_data)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ 数据处理完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {os.path.abspath(output_dir)}")
    print("\n💡 下一步:")
    print("   cd ..")
    print("   python main.py --dataset_dir dataset/processed")


if __name__ == "__main__":
    main()
