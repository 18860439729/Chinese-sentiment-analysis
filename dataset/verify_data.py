"""
验证三分类数据的格式和分布
"""

import json
from collections import Counter


def verify_dataset(file_path: str):
    """验证数据集"""
    print(f"\n{'='*60}")
    print(f"📊 验证: {file_path}")
    print('='*60)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 检查格式
    if data:
        sample = data[0]
        print(f"\n样本格式:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"\n示例数据:")
        print(f"  text: {sample['text'][:50]}...")
        print(f"  topic: {sample['topic']}")
        print(f"  label: {sample['label']}")
    
    # 统计标签分布
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    
    print(f"\n标签分布:")
    label_names = {0: '正面', 1: '负面', 2: '反讽'}
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = count / len(data) * 100
        name = label_names.get(label, f'未知({label})')
        print(f"  Label {label} ({name}): {count:6d} ({percentage:5.2f}%)")
    
    # 检查是否有 topic
    has_topic = sum(1 for item in data if item['topic'])
    print(f"\n包含 topic 的样本: {has_topic} ({has_topic/len(data)*100:.2f}%)")
    
    # 检查文本长度
    text_lengths = [len(item['text']) for item in data]
    print(f"\n文本长度统计:")
    print(f"  最小: {min(text_lengths)}")
    print(f"  最大: {max(text_lengths)}")
    print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f}")
    
    return True


def main():
    """主函数"""
    print("\n" + "🔍" * 30)
    print("三分类数据验证工具")
    print("🔍" * 30)
    
    files = [
        'processed/train.json',
        'processed/dev.json',
        'processed/test.json'
    ]
    
    for file_path in files:
        try:
            verify_dataset(file_path)
        except Exception as e:
            print(f"\n❌ 验证失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 验证完成！")
    print("="*60)


if __name__ == "__main__":
    main()
