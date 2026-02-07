"""
从缓存的数据集文件转换为 JSONL
"""

import os
import json
from pathlib import Path
from datasets import Dataset


def convert_dataset_cache(cache_dir: str, output_dir: str):
    """
    从缓存目录加载数据集并转换为 JSONL
    
    Args:
        cache_dir: 缓存目录
        output_dir: 输出目录
    """
    print(f"📁 处理目录: {cache_dir}")
    
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找 arrow 文件
    arrow_files = list(cache_path.glob("*.arrow"))
    
    if not arrow_files:
        print(f"   ⚠️ 没有找到 Arrow 文件")
        return False
    
    print(f"   找到 {len(arrow_files)} 个文件")
    
    success_count = 0
    
    for arrow_file in arrow_files:
        try:
            # 确定分割名称
            # chn_senti_corp-train.arrow -> train
            base_name = arrow_file.stem.replace("chn_senti_corp-", "")
            
            print(f"\n📄 转换: {arrow_file.name} -> {base_name}.jsonl")
            
            # 使用 datasets 库加载
            dataset = Dataset.from_file(str(arrow_file))
            
            print(f"   - 样本数: {len(dataset)}")
            print(f"   - 特征: {list(dataset.features.keys())}")
            
            # 保存为 JSONL
            output_file = output_path / f"{base_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"   ✅ 已保存到: {output_file}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
    
    return success_count > 0


def main():
    """主函数"""
    print("\n" + "🔄" * 30)
    print("缓存数据集转换工具")
    print("🔄" * 30 + "\n")
    
    # ChnSentiCorp
    chn_cache = "raw_source/ChnSentiCorp"
    chn_output = "raw_source/ChnSentiCorp"
    
    print("=" * 60)
    print("📥 转换 ChnSentiCorp 数据集")
    print("=" * 60)
    
    if convert_dataset_cache(chn_cache, chn_output):
        print("\n✅ ChnSentiCorp 转换完成！")
        
        # 保存数据集信息
        info_file = Path(chn_output) / "dataset_info.txt"
        jsonl_files = list(Path(chn_output).glob("*.jsonl"))
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("ChnSentiCorp 数据集信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"来源: HuggingFace - seamew/ChnSentiCorp\n")
            f.write(f"任务: 情感分析（正面/负面）\n")
            f.write(f"文件: {[f.name for f in jsonl_files]}\n")
        
        print(f"📄 数据集信息已保存到: {info_file}")
    else:
        print("\n❌ ChnSentiCorp 转换失败")
    
    print("\n" + "=" * 60)
    print("💡 下一步:")
    print("   python build_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
