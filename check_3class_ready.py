"""
检查三分类模型是否准备就绪
"""

import re


def check_file(file_path, checks):
    """检查文件中的关键代码"""
    print(f"\n{'='*60}")
    print(f"📄 检查: {file_path}")
    print('='*60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_passed = True
        for check_name, pattern, expected in checks:
            if re.search(pattern, content):
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name} - 未找到")
                if expected:
                    print(f"   期望: {expected}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "🔍" * 30)
    print("三分类模型准备检查")
    print("🔍" * 30)
    
    all_checks_passed = True
    
    # 检查 main.py
    main_checks = [
        ("num_classes 默认值为 3", r"--num_classes.*default=3", "default=3"),
        ("使用加权损失函数", r"class_weights.*=.*torch\.tensor.*2\.5", "torch.tensor([1.0, 1.0, 2.5])"),
        ("CrossEntropyLoss 使用权重", r"CrossEntropyLoss\(weight=class_weights\)", "weight=class_weights"),
        ("添加三分类标签名称", r"label_names.*=.*\['正面'.*'负面'.*'反讽'\]", "['正面', '负面', '反讽']"),
    ]
    
    if not check_file('main.py', main_checks):
        all_checks_passed = False
    
    # 检查 utils.py
    utils_checks = [
        ("f1_score 使用 weighted", r"average='weighted'", "average='weighted'"),
        ("添加 zero_division 参数", r"zero_division=0", "zero_division=0"),
        ("添加 f1_macro 指标", r"f1_macro", "f1_macro"),
        ("多分类 AUC 支持", r"multi_class='ovr'", "multi_class='ovr'"),
    ]
    
    if not check_file('utils.py', utils_checks):
        all_checks_passed = False
    
    # 检查数据文件
    print(f"\n{'='*60}")
    print(f"📊 检查数据文件")
    print('='*60)
    
    import os
    import json
    
    data_files = [
        'dataset/processed/train.json',
        'dataset/processed/dev.json',
        'dataset/processed/test.json'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查标签范围
                labels = set(item['label'] for item in data)
                if labels == {0, 1, 2}:
                    print(f"✅ {file_path}: 包含三个类别 {labels}")
                elif labels.issubset({0, 1, 2}):
                    print(f"⚠️ {file_path}: 只包含部分类别 {labels}")
                else:
                    print(f"❌ {file_path}: 标签异常 {labels}")
                    all_checks_passed = False
            except Exception as e:
                print(f"❌ {file_path}: 读取失败 - {e}")
                all_checks_passed = False
        else:
            print(f"❌ {file_path}: 文件不存在")
            all_checks_passed = False
    
    # 总结
    print("\n" + "="*60)
    if all_checks_passed:
        print("✅ 所有检查通过！")
        print("="*60)
        print("\n🎉 模型已准备就绪，可以开始训练！")
        print("\n训练命令:")
        print("   python main.py --dataset_dir dataset/processed --num_classes 3")
    else:
        print("❌ 部分检查未通过")
        print("="*60)
        print("\n⚠️ 请根据上述提示修改代码")
    print()


if __name__ == "__main__":
    main()
