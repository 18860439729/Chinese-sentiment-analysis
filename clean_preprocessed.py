"""
清理预处理数据脚本
删除之前生成的错误 .pkl 文件
"""

import os
import shutil

def clean_preprocessed_data(preprocessed_dir: str = "preprocessed_data"):
    """
    清理预处理数据目录
    
    Args:
        preprocessed_dir: 预处理数据目录
    """
    if os.path.exists(preprocessed_dir):
        print(f"🗑️  删除目录: {preprocessed_dir}")
        shutil.rmtree(preprocessed_dir)
        print("✅ 清理完成")
    else:
        print(f"⚠️  目录不存在: {preprocessed_dir}")
    
    # 重新创建空目录
    os.makedirs(preprocessed_dir, exist_ok=True)
    print(f"📁 重新创建目录: {preprocessed_dir}")

if __name__ == "__main__":
    clean_preprocessed_data()
    print("💡 现在可以重新运行: python preprocess_offline.py")