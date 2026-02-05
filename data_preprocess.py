"""
数据预处理模块
负责 HanLP 处理、生成超图结构
"""

import hanlp
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import networkx as nx
from collections import defaultdict
from transformers import BertTokenizer


class DataPreprocessor:
    """数据预处理器，负责文本处理和超图构建"""
    
    def __init__(self, bert_model_name: str = 'bert-base-chinese', hanlp_model_path: str = None):
        """
        初始化预处理器
        
        Args:
            bert_model_name: BERT模型名称，用于初始化tokenizer
            hanlp_model_path: HanLP模型路径，如果为None则使用默认模型
        """
        # 初始化BERT tokenizer - 绝对不能自己构建词汇表！
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 初始化HanLP模型 - 只在init中加载一次，避免重复加载
        self.hanlp_pipeline = self._load_hanlp_model(hanlp_model_path)
        
    def _load_hanlp_model(self, model_path: str = None):
        """加载HanLP模型 - 只加载一次"""
        if model_path:
            return hanlp.load(model_path)
        else:
            # 使用默认的中文模型
            return hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    
    def process_text_pair(self, text: str, topic: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        处理文本对（评论+主题），用于讽刺检测任务
        
        Args:
            text: 评论文本
            topic: 主题文本
            max_length: 最大序列长度
            
        Returns:
            包含input_ids, attention_mask, token_type_ids的字典
        """
        # 使用BERT tokenizer处理文本对
        encoded = self.tokenizer(
            text,
            topic,  # 第二个句子
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0)
        }
    
    def process_text_with_hanlp(self, text: str, topic: str = None) -> Dict[str, Any]:
        """
        使用HanLP处理文本，获取语言学特征用于构建超图
        优化：处理文本对（评论+主题）以获得更丰富的语言学特征
        
        Args:
            text: 评论文本
            topic: 主题文本（可选）
            
        Returns:
            包含分词、词性标注、命名实体识别等结果的字典
        """
        # 合并文本和主题进行HanLP处理，获得更全面的语言学特征
        combined_text = text
        if topic:
            combined_text = f"{topic} {text}"  # 主题在前，评论在后
            
        result = self.hanlp_pipeline(combined_text)
        
        # 分离主题和评论的token索引
        topic_tokens = self.hanlp_pipeline(topic)['tok'] if topic else []
        text_tokens = self.hanlp_pipeline(text)['tok']
        
        return {
            'tokens': result.get('tok', []),
            'pos_tags': result.get('pos', []),
            'ner_tags': result.get('ner', []),
            'dependencies': result.get('dep', []),
            'semantic_roles': result.get('srl', []),
            'topic_length': len(topic_tokens),  # 用于区分主题和评论部分
            'text_length': len(text_tokens)
        }
    
    def build_hypergraph_structure(self, text_pairs: List[Tuple[str, str]], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建超图结构 - 针对文本对（评论+主题）优化
        
        Args:
            text_pairs: (评论, 主题) 文本对列表
            max_length: BERT最大序列长度
            
        Returns:
            input_ids: BERT输入token IDs
            attention_mask: BERT注意力掩码
            token_type_ids: BERT token类型IDs
            hypergraph_matrix: 超图关联矩阵 H ∈ R^{N×M} (N个节点，M条超边)
        """
        # 使用BERT tokenizer处理所有文本对
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_hanlp_results = []
        
        for text, topic in text_pairs:
            # BERT tokenization - 处理文本对
            bert_tokens = self.process_text_pair(text, topic, max_length)
            all_input_ids.append(bert_tokens['input_ids'])
            all_attention_masks.append(bert_tokens['attention_mask'])
            all_token_type_ids.append(bert_tokens['token_type_ids'])
            
            # HanLP处理用于构建超图
            hanlp_result = self.process_text_with_hanlp(text, topic)
            all_hanlp_results.append(hanlp_result)
        
        # 构建超图关联矩阵 H ∈ R^{N×M}
        hypergraph_matrix = self._create_hypergraph_incidence_matrix(all_hanlp_results, max_length)
        
        # 硬件无关性：自动检测设备并移动张量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return (
            torch.stack(all_input_ids).to(device),
            torch.stack(all_attention_masks).to(device), 
            torch.stack(all_token_type_ids).to(device),
            torch.tensor(hypergraph_matrix, dtype=torch.float32).to(device)
        )
    
    def _create_single_hypergraph_matrix(self, hanlp_result: Dict[str, Any], max_length: int, max_edges: int) -> np.ndarray:
        """
        为单个样本创建独立的超图关联矩阵 - 修正：避免样本间数据泄露！
        
        Args:
            hanlp_result: 单个样本的HanLP处理结果
            max_length: 最大序列长度（节点数N）
            max_edges: 最大超边数（M，用于padding）
            
        Returns:
            超图关联矩阵 H ∈ R^{N×M}，单个样本独立
        """
        tokens = hanlp_result['tokens']
        dependencies = hanlp_result.get('dependencies', [])
        pos_tags = hanlp_result.get('pos_tags', [])
        ner_tags = hanlp_result.get('ner_tags', [])
        
        # 收集当前样本的超边
        sample_hyperedges = []
        
        # 超边1: 依存句法簇
        if dependencies:
            for dep in dependencies:
                if len(dep) >= 3:
                    head_idx, tail_idx = dep[0], dep[1]
                    if head_idx < len(tokens) and tail_idx < len(tokens) and head_idx < max_length and tail_idx < max_length:
                        hyperedge = set([head_idx, tail_idx])
                        sample_hyperedges.append(hyperedge)
        
        # 超边2: 词性标注簇
        pos_groups = defaultdict(set)
        for i, pos in enumerate(pos_tags):
            if i < len(tokens) and i < max_length:
                pos_groups[pos].add(i)
        
        for pos, indices in pos_groups.items():
            if len(indices) > 1:
                sample_hyperedges.append(indices)
        
        # 超边3: 命名实体簇
        if ner_tags:
            current_entity = None
            current_indices = set()
            
            for i, ner in enumerate(ner_tags):
                if i < len(tokens) and i < max_length:
                    if ner.startswith('B-'):
                        if current_entity and len(current_indices) > 1:
                            sample_hyperedges.append(current_indices)
                        current_entity = ner[2:]
                        current_indices = {i}
                    elif ner.startswith('I-') and current_entity:
                        current_indices.add(i)
                    else:
                        if current_entity and len(current_indices) > 1:
                            sample_hyperedges.append(current_indices)
                        current_entity = None
                        current_indices = set()
            
            if current_entity and len(current_indices) > 1:
                sample_hyperedges.append(current_indices)
        
        # 超边4: 滑动窗口
        window_size = 3
        for i in range(min(len(tokens), max_length) - window_size + 1):
            window_indices = set(range(i, i + window_size))
            sample_hyperedges.append(window_indices)
        
        # 超边5: 主题-评论关联
        topic_length = hanlp_result.get('topic_length', 0)
        text_length = hanlp_result.get('text_length', 0)
        
        if topic_length > 0 and text_length > 0:
            topic_indices = set(range(1, min(topic_length + 1, len(tokens), max_length)))
            text_start = topic_length + 1
            text_indices = set(range(text_start, min(text_start + text_length, len(tokens), max_length)))
            
            if len(topic_indices) > 0 and len(text_indices) > 0:
                sample_hyperedges.append(topic_indices.union(text_indices))
                if len(topic_indices) > 1:
                    sample_hyperedges.append(topic_indices)
                if len(text_indices) > 1:
                    sample_hyperedges.append(text_indices)
        
        # 构建关联矩阵并进行padding
        num_actual_edges = len(sample_hyperedges)
        incidence_matrix = np.zeros((max_length, max_edges))
        
        # 填充实际的超边（限制在max_edges范围内）
        for j, hyperedge in enumerate(sample_hyperedges[:max_edges]):
            for node_idx in hyperedge:
                if node_idx < max_length:
                    incidence_matrix[node_idx, j] = 1.0
        
        return incidence_matrix
    
    def estimate_max_edges(self, hanlp_results: List[Dict[str, Any]]) -> int:
        """
        估算批次中的最大超边数，用于padding
        
        Args:
            hanlp_results: HanLP处理结果列表
            
        Returns:
            最大超边数
        """
        max_edges = 0
        
        for result in hanlp_results:
            tokens = result['tokens']
            dependencies = result.get('dependencies', [])
            pos_tags = result.get('pos_tags', [])
            ner_tags = result.get('ner_tags', [])
            
            # 粗略估算当前样本的超边数
            edge_count = 0
            
            # 依存关系超边
            edge_count += len(dependencies) if dependencies else 0
            
            # 词性超边（估算）
            unique_pos = len(set(pos_tags)) if pos_tags else 0
            edge_count += unique_pos
            
            # 实体超边（估算）
            entity_count = sum(1 for tag in (ner_tags or []) if tag.startswith('B-'))
            edge_count += entity_count
            
            # 滑动窗口超边
            window_edges = max(0, len(tokens) - 2) if tokens else 0
            edge_count += window_edges
            
            # 主题-评论超边
            edge_count += 3  # 交互+主题+评论
            
            max_edges = max(max_edges, edge_count)
        
        # 添加一些缓冲
        return min(max_edges + 10, 200)  # 限制最大值避免内存爆炸


def load_dataset(file_path: str) -> List[Tuple[str, str, int]]:
    """
    加载JSON格式的数据集
    
    Args:
        file_path: JSON数据文件路径
        
    Returns:
        (文本, 主题, 标签) 的列表
    """
    import json
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        for item in json_data:
            text = item.get('text', '').strip()
            topic = item.get('topic', '').strip()
            label = int(item.get('label', '0'))  # 转换字符串标签为整数
            
            if text:  # 确保文本不为空
                data.append((text, topic, label))
                
    except Exception as e:
        print(f"加载数据集时出错: {file_path}, 错误: {e}")
        
    return data


def load_all_datasets(dataset_dir: str = "dataset") -> Tuple[List[Tuple[str, str, int]], 
                                                           List[Tuple[str, str, int]], 
                                                           List[Tuple[str, str, int]]]:
    """
    加载所有数据集文件
    
    Args:
        dataset_dir: 数据集目录路径
        
    Returns:
        (训练数据, 验证数据, 测试数据) 的元组
    """
    import os
    
    train_path = os.path.join(dataset_dir, 'train.json')
    dev_path = os.path.join(dataset_dir, 'dev.json')
    test_path = os.path.join(dataset_dir, 'test.json')
    
    train_data = load_dataset(train_path) if os.path.exists(train_path) else []
    dev_data = load_dataset(dev_path) if os.path.exists(dev_path) else []
    test_data = load_dataset(test_path) if os.path.exists(test_path) else []
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(dev_data)} 样本") 
    print(f"  测试集: {len(test_data)} 样本")
    
    return train_data, dev_data, test_data


class SarcasmDataset(torch.utils.data.Dataset):
    """讽刺检测数据集类 - 移到全局作用域"""
    def __init__(self, data, preprocessor, max_length):
        self.data = data
        self.preprocessor = preprocessor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, topic, label = self.data[idx]
        
        # 使用BERT tokenizer处理文本对
        bert_tokens = self.preprocessor.process_text_pair(text, topic, self.max_length)
        
        # 使用HanLP处理用于超图构建
        hanlp_result = self.preprocessor.process_text_with_hanlp(text, topic)
        
        return {
            'input_ids': bert_tokens['input_ids'],
            'attention_mask': bert_tokens['attention_mask'],
            'token_type_ids': bert_tokens['token_type_ids'],
            'hanlp_result': hanlp_result,
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,  # 保留原始文本用于调试
            'topic': topic  # 保留原始主题用于调试
        }


def create_hypergraph_collate_fn(preprocessor, max_length):
    """创建超图批处理函数的工厂函数"""
    def collate_fn(batch):
        """自定义批处理函数，处理超图矩阵构建 - 修正：每个样本独立的超图矩阵"""
        # 提取批次数据
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # 构建批次的超图矩阵 - 关键修正：每个样本独立！
        hanlp_results = [item['hanlp_result'] for item in batch]
        
        # 估算最大超边数用于padding
        max_edges = preprocessor.estimate_max_edges(hanlp_results)
        
        # 为每个样本创建独立的超图矩阵
        batch_hypergraph_matrices = []
        for hanlp_result in hanlp_results:
            single_matrix = preprocessor._create_single_hypergraph_matrix(hanlp_result, max_length, max_edges)
            batch_hypergraph_matrices.append(single_matrix)
        
        # 堆叠成 [batch_size, max_length, max_edges] 的3D张量
        hypergraph_matrix = torch.tensor(np.stack(batch_hypergraph_matrices), dtype=torch.float32)
        
        # 硬件无关性
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'token_type_ids': token_type_ids.to(device),
            'hypergraph_matrix': hypergraph_matrix.to(device),  # [batch_size, N, M]
            'labels': labels.to(device)
        }
    
    return collate_fn


def create_data_loaders(train_data: List[Tuple[str, str, int]], 
                       val_data: List[Tuple[str, str, int]], 
                       preprocessor: DataPreprocessor,
                       batch_size: int = 32,
                       max_length: int = 512) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建数据加载器 - 针对JSON格式数据集优化
    
    Args:
        train_data: 训练数据 [(text, topic, label), ...]
        val_data: 验证数据 [(text, topic, label), ...]
        preprocessor: 数据预处理器
        batch_size: 批次大小
        max_length: BERT最大序列长度
        
    Returns:
        训练和验证数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 创建批处理函数
    collate_fn = create_hypergraph_collate_fn(preprocessor, max_length)
    
    train_dataset = SarcasmDataset(train_data, preprocessor, max_length)
    val_dataset = SarcasmDataset(val_data, preprocessor, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader