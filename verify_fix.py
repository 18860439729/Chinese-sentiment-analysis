"""
快速验证数据泄露修复
"""
import json

print("=" * 60)
print("🔍 数据泄露修复验证")
print("=" * 60)

# 加载训练数据
with open('dataset/processed/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 找到每个类别的样本
samples = {}
for item in data:
    label = item['label']
    if label not in samples:
        samples[label] = item

# 显示样本
label_names = {0: '正面', 1: '负面', 2: '反讽'}
print("\n📊 每个类别的样本示例：\n")

for label in sorted(samples.keys()):
    sample = samples[label]
    name = label_names[label]
    print(f"Label {label} ({name}):")
    print(f"  text: {sample['text'][:60]}...")
    print(f"  topic: '{sample['topic']}'")
    print(f"  label: {sample['label']}")
    print()

# 验证所有样本都有 topic
has_topic_count = sum(1 for item in data if item['topic'])
total_count = len(data)

print("=" * 60)
print(f"✅ 包含 topic 的样本: {has_topic_count}/{total_count} ({has_topic_count/total_count*100:.2f}%)")
print("=" * 60)

if has_topic_count == total_count:
    print("\n🎉 修复成功！所有样本都有 topic，数据泄露问题已解决！")
else:
    print(f"\n⚠️ 警告：还有 {total_count - has_topic_count} 个样本没有 topic")
