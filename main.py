"""
主程序入口
实验的入口文件
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from typing import Dict, Any, List, Tuple

from data_preprocess import (
    DataPreprocessor, 
    load_all_datasets, 
    create_data_loaders, 
    SarcasmDataset,
    create_hypergraph_collate_fn  # 新增：用于测试集
)
from model import BertHGNNModel
from utils import (
    set_seed, setup_logging, save_config, load_config, EarlyStopping,
    MetricsCalculator, Visualizer, count_parameters, save_model, load_model,
    get_device, AverageMeter, format_time
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BERT + HGNN + Attention 文本分类')
    
    # 数据相关参数
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='数据集目录路径')
    parser.add_argument('--cache_dir', type=str, default='cache', help='缓存目录路径')
    parser.add_argument('--train_data', type=str, help='训练数据路径（可选，默认使用dataset/train.json）')
    parser.add_argument('--val_data', type=str, help='验证数据路径（可选，默认使用dataset/dev.json）')
    parser.add_argument('--test_data', type=str, help='测试数据路径（可选，默认使用dataset/test.json）')
    
    # 模型相关参数
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese', 
                       help='BERT模型名称')
    parser.add_argument('--hgnn_hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='HGNN隐藏层维度')
    parser.add_argument('--num_attention_heads', type=int, default=8, 
                       help='注意力头数')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--freeze_bert', action='store_true', help='是否冻结BERT参数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500, help='预热步数')
    
    # 早停相关参数
    parser.add_argument('--patience', type=int, default=7, help='早停容忍轮数')
    parser.add_argument('--min_delta', type=float, default=0.001, help='早停最小改善幅度')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--config_file', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--evaluate_only', action='store_true', help='仅进行评估')
    
    return parser.parse_args()


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                logger) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # 硬件无关性：数据已经在collate_fn中移到设备上
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'] 
        token_type_ids = batch['token_type_ids']
        hypergraph_matrix = batch['hypergraph_matrix']
        labels = batch['labels']
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, attention_mask, hypergraph_matrix, token_type_ids)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # 更新统计信息
        losses.update(loss.item(), labels.size(0))
        accuracies.update(accuracy.item(), labels.size(0))
        
        # 打印进度
        if batch_idx % 100 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                       f'Loss: {losses.avg:.4f}, Acc: {accuracies.avg:.4f}, '
                       f'Time: {format_time(elapsed_time)}')
    
    return losses.avg, accuracies.avg


def validate_epoch(model: nn.Module, 
                  val_loader: DataLoader, 
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    """验证一个epoch"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 硬件无关性：数据已经在collate_fn中移到设备上
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids'] 
            hypergraph_matrix = batch['hypergraph_matrix']
            labels = batch['labels']
            
            # 前向传播
            outputs = model(input_ids, attention_mask, hypergraph_matrix, token_type_ids)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()
            
            # 更新统计信息
            losses.update(loss.item(), labels.size(0))
            accuracies.update(accuracy.item(), labels.size(0))
            
            # 收集预测结果
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return losses.avg, accuracies.avg, all_predictions, all_labels


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("开始训练...")
    
    # 获取设备 - 硬件无关性标准写法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载配置文件（如果提供）
    if args.config_file and os.path.exists(args.config_file):
        config = load_config(args.config_file)
        logger.info(f"从配置文件加载参数: {args.config_file}")
        # 更新args中的参数
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # 保存当前配置
    config_save_path = os.path.join(args.save_dir, 'config.json')
    save_config(vars(args), config_save_path)
    
    # 初始化数据预处理器 - 传入BERT模型名称
    logger.info("初始化数据预处理器...")
    preprocessor = DataPreprocessor(bert_model_name=args.bert_model)
    
    # 加载数据 - 使用内置缓存机制
    logger.info("📥 加载数据集...")
    
    if args.train_data and args.val_data:
        # 使用指定的数据文件路径
        from data_preprocess import load_dataset
        train_data = load_dataset(args.train_data)
        val_data = load_dataset(args.val_data)
        test_data = load_dataset(args.test_data) if args.test_data else []
    else:
        # 使用默认的数据集目录
        train_data, val_data, test_data = load_all_datasets(args.dataset_dir)
    
    logger.info(f"训练数据: {len(train_data)} 样本")
    logger.info(f"验证数据: {len(val_data)} 样本")
    if test_data:
        logger.info(f"测试数据: {len(test_data)} 样本")
    
    # 创建数据加载器（内置缓存机制，第一次运行会慢，之后会很快）
    logger.info("🔧 创建数据加载器（内置缓存机制）...")
    logger.info("💡 第一次运行会进行HanLP预处理并缓存，之后启动会很快")
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, preprocessor, args.batch_size, max_length=256, cache_dir=args.cache_dir
    )
    
    # 初始化模型 - 硬件无关性标准写法
    logger.info("初始化模型...")
    model = BertHGNNModel(
        bert_model_name=args.bert_model,
        hgnn_hidden_dims=args.hgnn_hidden_dims,
        num_attention_heads=args.num_attention_heads,
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    ).to(device)  # 务必加上 .to(device)
    
    # 打印模型信息
    num_params = count_parameters(model)
    logger.info(f"模型参数数量: {num_params:,}")
    
    # 定义损失函数和优化器 - 分层学习率策略
    criterion = nn.CrossEntropyLoss()
    
    # 分层学习率：BERT用小学习率微调，HGNN+Attention用大学习率训练
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},  # BERT: 2e-5
        {'params': other_params, 'lr': args.learning_rate * 50, 'weight_decay': args.weight_decay}  # HGNN+Attention: 1e-3
    ])
    
    logger.info(f"优化器配置:")
    logger.info(f"  BERT参数: {len(bert_params)} 个, 学习率: {args.learning_rate}")
    logger.info(f"  HGNN+Attention参数: {len(other_params)} 个, 学习率: {args.learning_rate * 50}")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # 恢复训练（如果提供检查点）
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复训练: {args.resume}")
        model, optimizer, start_epoch, _ = load_model(model, optimizer, args.resume)
    
    # 如果只进行评估
    if args.evaluate_only:
        if not args.resume:
            logger.error("评估模式需要提供模型检查点")
            return
        
        logger.info("开始评估...")
        val_loss, val_acc, predictions, true_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 计算详细指标
        metrics = MetricsCalculator.calculate_metrics(true_labels, predictions)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # 打印分类报告
        MetricsCalculator.print_classification_report(true_labels, predictions)
        
        # 绘制混淆矩阵
        Visualizer.plot_confusion_matrix(true_labels, predictions)
        
        return
    
    # 训练循环
    logger.info("开始训练...")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        
        # 验证
        val_loss, val_acc, predictions, true_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f'Epoch {epoch+1}/{args.epochs} - '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                   f'Time: {format_time(epoch_time)}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_model(model, optimizer, epoch, val_loss, best_model_path)
            logger.info(f"保存最佳模型: {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_model(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # 早停检查
        if early_stopping(val_loss, model):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 训练完成
    logger.info("训练完成!")
    logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
    
    # 绘制训练历史
    history_plot_path = os.path.join(args.save_dir, 'training_history.png')
    Visualizer.plot_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies, history_plot_path
    )
    
    # 最终评估
    logger.info("进行最终评估...")
    
    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model, _, _, _ = load_model(model, optimizer, best_model_path)
    
    # 在验证集上评估
    val_loss, val_acc, predictions, true_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # 计算详细指标
    metrics = MetricsCalculator.calculate_metrics(true_labels, predictions)
    logger.info("最终验证结果:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 打印分类报告
    MetricsCalculator.print_classification_report(true_labels, predictions)
    
    # 绘制混淆矩阵
    confusion_matrix_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    # 告诉画图工具，0是正常，1是讽刺
    Visualizer.plot_confusion_matrix(true_labels, predictions, labels=['Normal', 'Sarcastic'], save_path=confusion_matrix_path)
    
    # 如果有测试数据，进行测试
    if test_data and len(test_data) > 0:
        logger.info("在测试集上评估...")
        
        # 修复：创建 collate_fn 和测试数据加载器
        logger.info("🔧 创建测试数据加载器...")
        
        # 1. 创建 collate_fn（使用与训练相同的配置）
        test_collate_fn = create_hypergraph_collate_fn(preprocessor, max_length=256)
        
        # 2. 创建测试数据集（带缓存）
        test_cache_file = os.path.join(args.cache_dir, 'test_cache.pkl')
        test_dataset = SarcasmDataset(test_data, preprocessor, 256, cache_file=test_cache_file)
        
        # 3. 创建 DataLoader（注意：num_workers=0 因为 collate_fn 里用了 .to(device)）
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=test_collate_fn,
            num_workers=0  # 重要：避免多进程与GPU冲突
        )
        
        logger.info(f"✅ 测试数据加载器创建完成: {len(test_dataset)} 样本")
        
        # 评估测试集
        test_loss, test_acc, test_predictions, test_labels = validate_epoch(model, test_loader, criterion, device)
        logger.info(f"测试准确率: {test_acc:.4f}")
        
        # 测试集详细指标
        test_metrics = MetricsCalculator.calculate_metrics(test_labels, test_predictions)
        logger.info("测试集结果:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main()