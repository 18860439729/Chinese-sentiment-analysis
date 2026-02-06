"""
数据预处理模块 (修复版)
负责 HanLP 处理、生成超图结构
包含：智能键名匹配、缓存机制
"""

import hanlp
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from transformers import BertTokenizer
import os
import pickle
from tqdm import tqdm

class DataPreprocessor:
    """数据预处理器，负责文本处理和超图构建"""
    
    def __init__(self, bert_model_name: str = 'bert-base-chinese', hanlp_model_path: str = None):
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # 初始化HanLP模型
        self.hanlp_pipeline = self._load_hanlp_model(hanlp_model_path)
        
    def _load_hanlp_model(self, model_path: str = None):
        if model_path:
            return hanlp.load(model_path)
        else:
            # 使用默认的中文多任务模型
            return hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

    def _get_feature(self, doc, prefix):
        """
        [关键修复] 安全获取 HanLP 结果中的字段 
        自动兼容 tok/fine, pos/ctb 等不同键名
        """
        # 1. 尝试直接匹配 (e.g., 'tok')
        if prefix in doc:
            return doc[prefix]
        
        # 2. 尝试常见变体 (e.g., 'tok/fine')
        variants = {
            'tok': ['tok/fine', 'tok/coarse'],
            'pos': ['pos/ctb', 'pos/pku', 'pos/863'],
            'ner': ['ner/msra', 'ner/pku', 'ner/ontonotes'],
            'dep': ['dep/ctb', 'dep/pmt'],
            'srl': ['srl']
        }
        for v in variants.get(prefix, []):
            if v in doc:
                return doc[v]
                
        # 3. 暴力搜索前缀 (只要是 tok/ 开头的都算)
        for k in doc:
            if str(k).startswith(prefix + '/'):
                return doc[k]
        
        # 4. 如果没找到，返回空列表，防止报错
        return []
    
    def process_text_pair(self, text: str, topic: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """处理BERT输入"""
        encoded = self.tokenizer(
            text,
            topic,
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
        使用HanLP处理文本，获取语言学特征
        [已修复 KeyError: 'tok' 问题]
        """
        combined_text = text
        if topic:
            combined_text = f"{topic} {text}"
            
        # 1. 处理整个拼接文本
        result = self.hanlp_pipeline(combined_text)
        
        # 2. 安全提取特征
        tokens = self._get_feature(result, 'tok')
        pos_tags = self._get_feature(result, 'pos')
        ner_tags = self._get_feature(result, 'ner')
        dependencies = self._get_feature(result, 'dep')
        semantic_roles = self._get_feature(result, 'srl')

        # 3. 计算 topic 和 text 的长度 (用于区分节点归属)
        # 为了稳健性，我们单独分词一次计算长度，虽然稍微慢点，但不会出错
        topic_len = 0
        if topic:
            topic_res = self.hanlp_pipeline(topic)
            topic_tokens = self._get_feature(topic_res, 'tok')
            topic_len = len(topic_tokens)
            
        text_res = self.hanlp_pipeline(text)
        text_tokens = self._get_feature(text_res, 'tok')
        text_len = len(text_tokens)
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'ner_tags': ner_tags,
            'dependencies': dependencies,
            'semantic_roles': semantic_roles,
            'topic_length': topic_len,
            'text_length': text_len
        }
    
    def _create_single_hypergraph_matrix(self, hanlp_result: Dict[str, Any], max_length: int, max_edges: int) -> np.ndarray:
        """为单个样本创建超图关联矩阵"""
        tokens = hanlp_result['tokens']
        dependencies = hanlp_result.get('dependencies', [])
        pos_tags = hanlp_result.get('pos_tags', [])
        ner_tags = hanlp_result.get('ner_tags', [])
        
        sample_hyperedges = []
        
        # 超边1: 依存句法
        if dependencies:
            for dep in dependencies:
                # dep 格式通常是 (head_idx, rel, tail_idx) 或类似
                # 这里假设是 [head, tail] 或 (head, tail, rel)
                # HanLP 2.x 的 dep 格式通常是 [(id, head, rel)...] 或者 (head_index, rel) 列表
                # 我们做个简单兼容:
                try:
                    # 假设 dep[i] = (head_index, relation)
                    for i, item in enumerate(dependencies):
                        head_idx = -1
                        if isinstance(item, (list, tuple)):
                            # 如果是元组，通常第一个或者是head
                            head_idx = item[0] 
                            # 注意 HanLP head 索引可能是 1-based，如果是 0 表示 ROOT
                            # 这里需要根据具体输出调试，暂且假设 1-based (HanLP标准)
                            if head_idx > 0:
                                head_idx -= 1 # 转 0-based
                        elif hasattr(item, 'head'): # 或者是对象
                            head_idx = item.head - 1

                        tail_idx = i
                        
                        if head_idx >= 0 and head_idx < len(tokens) and tail_idx < len(tokens):
                            if head_idx < max_length and tail_idx < max_length:
                                sample_hyperedges.append({head_idx, tail_idx})
                except:
                    pass # 依存解析格式复杂，若出错则跳过，保证不崩

        # 超边2: 词性簇
        pos_groups = defaultdict(set)
        for i, pos in enumerate(pos_tags):
            if i < len(tokens) and i < max_length:
                pos_groups[pos].add(i)
        for indices in pos_groups.values():
            if len(indices) > 1:
                sample_hyperedges.append(indices)
        
        # 超边3: 命名实体
        if ner_tags:
            # HanLP NER 格式可能是 [('Entity', 'Type', begin, end)...]
            # 或者是 BIO 标签列表
            # 这里做个简单的 BIO 处理兼容
            if isinstance(ner_tags[0], str) and (ner_tags[0].startswith('B-') or ner_tags[0] == 'O'):
                # BIO 格式
                curr_indices = set()
                for i, tag in enumerate(ner_tags):
                    if i >= max_length: break
                    if tag.startswith('B-'):
                        if len(curr_indices) > 1: sample_hyperedges.append(curr_indices)
                        curr_indices = {i}
                    elif tag.startswith('I-') and curr_indices:
                        curr_indices.add(i)
                    else:
                        if len(curr_indices) > 1: sample_hyperedges.append(curr_indices)
                        curr_indices = set()
                if len(curr_indices) > 1: sample_hyperedges.append(curr_indices)
            else:
                # 假设是 Span 格式 (entity, type, start, end)
                for item in ner_tags:
                    try:
                        # 尝试解包，不确定具体格式，通常最后两个是 start, end
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            start, end = item[-2], item[-1] # 假设最后是起止
                            if isinstance(start, int) and isinstance(end, int):
                                indices = set(range(start, end))
                                valid_indices = {idx for idx in indices if idx < max_length}
                                if len(valid_indices) > 1:
                                    sample_hyperedges.append(valid_indices)
                    except: pass

        # 超边4: 滑动窗口
        window_size = 3
        for i in range(min(len(tokens), max_length) - window_size + 1):
            sample_hyperedges.append(set(range(i, i + window_size)))
        
        # 超边5: 主题-评论交互
        topic_len = hanlp_result.get('topic_length', 0)
        text_len = hanlp_result.get('text_length', 0)
        if topic_len > 0 and text_len > 0:
            t_start = 0
            t_end = min(topic_len, max_length)
            c_start = topic_len
            c_end = min(topic_len + text_len, max_length)
            
            if c_end > c_start:
                topic_idxs = set(range(t_start, t_end))
                comment_idxs = set(range(c_start, c_end))
                # 强行把它们连起来
                if topic_idxs and comment_idxs:
                    sample_hyperedges.append(topic_idxs | comment_idxs)

        # 构建矩阵
        incidence_matrix = np.zeros((max_length, max_edges), dtype=np.float32)
        for j, edge in enumerate(sample_hyperedges[:max_edges]):
            for node_idx in edge:
                if node_idx < max_length:
                    incidence_matrix[node_idx, j] = 1.0
                    
        return incidence_matrix

    def estimate_max_edges(self, hanlp_results: List[Dict[str, Any]]) -> int:
        """估算最大边数"""
        max_e = 0
        for res in hanlp_results:
            # 粗略估算：依存(N) + 窗口(N) + 词性(N/2) + NER(N/10) + Topic(1)
            # 安全起见，给个较大的倍数
            n = len(res.get('tokens', []))
            est = n * 3 + 10
            max_e = max(max_e, est)
        return min(max_e + 20, 500) # 上限 500

def load_dataset(file_path: str) -> List[Tuple[str, str, int]]:
    import json
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for item in items:
                data.append((item.get('text',''), item.get('topic',''), int(item.get('label',0))))
    return data

def load_all_datasets(dataset_dir: str = "dataset"):
    train = load_dataset(os.path.join(dataset_dir, 'train.json'))
    dev = load_dataset(os.path.join(dataset_dir, 'dev.json'))
    test = load_dataset(os.path.join(dataset_dir, 'test.json'))
    return train, dev, test

class SarcasmDataset(torch.utils.data.Dataset):
    """带缓存的数据集类"""
    def __init__(self, data, preprocessor, max_length, cache_file=None):
        self.data = data
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.cached_data = []
        
        # 缓存逻辑
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cache from {cache_file} ...")
            try:
                with open(cache_file, 'rb') as f:
                    self.cached_data = pickle.load(f)
            except:
                print("Cache broken, re-processing...")
                self.cached_data = []
        
        if not self.cached_data:
            print("Processing data with HanLP (this happens once)...")
            for text, topic, label in tqdm(self.data):
                bert_tokens = self.preprocessor.process_text_pair(text, topic, self.max_length)
                hanlp_result = self.preprocessor.process_text_with_hanlp(text, topic)
                self.cached_data.append({
                    'input_ids': bert_tokens['input_ids'],
                    'attention_mask': bert_tokens['attention_mask'],
                    'token_type_ids': bert_tokens['token_type_ids'],
                    'hanlp_result': hanlp_result,
                    'label': torch.tensor(label, dtype=torch.long)
                })
            
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cached_data, f)
                    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

def create_hypergraph_collate_fn(preprocessor, max_length):
    def collate_fn(batch):
        input_ids = torch.stack([x['input_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        token_type_ids = torch.stack([x['token_type_ids'] for x in batch])
        labels = torch.stack([x['label'] for x in batch])
        
        hanlp_results = [x['hanlp_result'] for x in batch]
        max_edges = preprocessor.estimate_max_edges(hanlp_results)
        
        matrices = []
        for res in hanlp_results:
            mat = preprocessor._create_single_hypergraph_matrix(res, max_length, max_edges)
            matrices.append(mat)
            
        hypergraph_matrix = torch.tensor(np.stack(matrices), dtype=torch.float32)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'token_type_ids': token_type_ids.to(device),
            'hypergraph_matrix': hypergraph_matrix.to(device),
            'labels': labels.to(device)
        }
    return collate_fn

def create_data_loaders(train_data, val_data, preprocessor, batch_size, max_length, cache_dir='cache'):
    collate_fn = create_hypergraph_collate_fn(preprocessor, max_length)
    
    # 使用 hash 或文件名确保缓存唯一
    train_cache = os.path.join(cache_dir, f'train_cache_{max_length}.pkl')
    val_cache = os.path.join(cache_dir, f'val_cache_{max_length}.pkl')
    
    train_dataset = SarcasmDataset(train_data, preprocessor, max_length, train_cache)
    val_dataset = SarcasmDataset(val_data, preprocessor, max_length, val_cache)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader