"""
工具函数模块
存放各种工具函数
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import logging
from datetime import datetime
import random


def set_seed(seed: int = 42):
    """设置随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        
    Returns:
        配置好的logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            是否应该早停
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int], 
                         labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签名称
            
        Returns:
            包含各种指标的字典
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        unique_labels = sorted(list(set(y_true + y_pred)))
        for i, label in enumerate(unique_labels):
            label_name = labels[label] if labels and label < len(labels) else f'class_{label}'
            metrics[f'{label_name}_precision'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
            metrics[f'{label_name}_recall'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
            metrics[f'{label_name}_f1'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true: List[int], y_pred: List[int], 
                                  labels: Optional[List[str]] = None):
        """打印分类报告"""
        print("\n" + "="*50)
        print("Classification Report")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=labels))


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_training_history(train_losses: List[float], val_losses: List[float],
                            train_accuracies: List[float], val_accuracies: List[float],
                            save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_accuracies: 训练准确率列表
            val_accuracies: 验证准确率列表
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                            labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签名称
            save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_attention_weights(attention_weights: torch.Tensor, tokens: List[str],
                             save_path: Optional[str] = None):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重矩阵
            tokens: token列表
            save_path: 保存路径
        """
        # 转换为numpy数组
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # 如果是批次数据，取第一个样本
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights[0]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_weights, 
                   xticklabels=tokens[:attention_weights.shape[1]], 
                   yticklabels=tokens[:attention_weights.shape[0]],
                   cmap='Blues', cbar=True)
        plt.title('Attention Weights Visualization')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, 
               epoch: int, loss: float, save_path: str):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, 
               checkpoint_path: str) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        加载后的模型、优化器、轮数和损失
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss


def get_device() -> torch.device:
    """获取可用设备 - 硬件无关性标准写法"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    return device


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"