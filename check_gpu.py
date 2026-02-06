import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("环境检查通过：准备起飞。")
else:
    print("环境检查失败：你正在使用CPU，这会导致训练极慢！请检查CUDA驱动。")
