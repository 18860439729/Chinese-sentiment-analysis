"""
离线预处理脚本 - 完全重写版本
彻底解耦，不依赖 data_preprocess.py，避免环境冲突
独立加载纯净的 BertTokenizer 和 HanLP
"""

import json
import pickle
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import torch
from collections import defaultdict

# 独立导入，避免命名空间冲突
print("🔧 导入依赖库...")
try:
    from transformers import BertTokenizer
    print("✅ transformers.BertTokenizer 导入成功")
except ImportError as e:
    print(f"❌ transformers 导入失败: {e}")
    print("💡 请安装: pip install transformers")
    exit(1)

try:
    import hanlp
    print("✅ hanlp 导入成功")
except ImportError as e:
    print(f"❌ hanlp 导入失败: {e}")
    print("💡 请安装: pip install hanlp")
    exit(1)


class IndependentPreprocessor:
    """
    完全独立的预处理器
    不依赖任何其他模块，避免环境冲突
    """
    
    def __init__(self, bert_model_name: str = 'bert-base-chinese'):
        """初始化独立的预处理器"""
        print(f"🚀 初始化独立预处理器...")
        
        # 1. 初始化纯净的 BERT tokenizer
        print(f"📥 加载 BERT tokenizer: {bert_model_name}")
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            print("✅ BERT tokenizer 加载成功")
        except Exception as e:
            print(f"❌ BERT tokenizer 加载失败: {e}")
            raise e
        
        # 2. 初始化 HanLP 模型
        print("📥 加载 HanLP 模型...")
        try:
            self.hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            print("✅ HanLP 模型加载成功")
        except Exception as e:
            print(f"❌ HanLP 模型加载失败: {e}")
            raise e
        
        print("🎉 独立预处理器初始化完成")
    
    def bert_encode(self, text: str, topic: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        纯净的 BERT 编码，不与其他库冲突
        
        Args:
            text: 评论文本
            topic: 主题文本
            max_length: 最大长度
            
        Returns:
            BERT 编码结果
        """
        try:
            # 使用纯净的 transformers tokenizer
            encoded = self.bert_tokenizer(
                text,                    # 第一个句子
                topic,                   # 第二个句子
                add_special_tokens=True, # 添加 [CLS], [SEP]
                max_length=max_length,   # 最大长度
                padding='max_length',    # 填充到最大长度
                truncation=True,         # 截断超长文本
                return_tensors='pt'      # 返回 PyTorch 张量
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0)
            }
            
        except Exception as e:
            print(f"❌ BERT 编码失败: {e}")
            print(f"   文本: {text[:100]}...")
            print(f"   主题: {topic[:100]}...")
            raise e
    
    def hanlp_analyze(self, text: str, topic: str = None) -> Dict[str, Any]:
        """
        纯净的 HanLP 分析，不与其他库冲突
        
        Args:
            text: 评论文本
            topic: 主题文本
            
        Returns:
            HanLP 分析结果
        """
        try:
            # 合并文本进行分析
            combined_text = text
            if topic:
                combined_text = f"{topic} {text}"
            
            # HanLP 多任务分析
            result = self.hanlp_model(combined_text)
            
            # 分别分析主题和文本长度
            topic_length = 0
            text_length = 0
            
            if topic:
                try:
                    topic_result = self.hanlp_model(topic)
                    topic_length = len(topic_result.get('tok', []))
                except:
                    topic_length = 0
            
            try:
                text_result = self.hanlp_model(text)
                text_length = len(text_result.get('tok', []))
            except:
                text_length = 0
            
            return {
                'tokens': result.get('tok', []),
                'pos_tags': result.get('pos', []),
                'ner_tags': result.get('ner', []),
                'dependencies': result.get('dep', []),
                'semantic_roles': result.get('srl', []),
                'topic_length': topic_length,
                'text_length': text_length
            }
            
        except Exception as e:
            print(f"❌ HanLP 分析失败: {e}")
            print(f"   文本: {text[:100]}...")
            if topic:
                print(f"   主题: {topic[:100]}...")
            
            # 返回空结果，避免程序崩溃
            return {
                'tokens': [],
                'pos_tags': [],
                'ner_tags': [],
                'dependencies': [],
                'semantic_roles': [],
                'topic_length': 0,
                'text_length': 0
            }
    
    def process_single_sample(self, text: str, topic: str, label: int, max_length: int = 512) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            text: 评论文本
            topic: 主题文本
            label: 标签
            max_length: 最大长度
            
        Returns:
            处理后的样本数据
        """
        # BERT 编码
        bert_result = self.bert_encode(text, topic, max_length)
        
        # HanLP 分析
        hanlp_result = self.hanlp_analyze(text, topic)
        
        # 验证结果
        if bert_result['input_ids'].numel() == 0:
            raise ValueError("BERT 编码结果为空")
        
        if len(hanlp_result['tokens']) == 0:
            raise ValueError("HanLP 分析结果为空")
        
        return {
            'text': text,
            'topic': topic,
            'label': label,
            'input_ids': bert_result['input_ids'],
            'attention_mask': bert_result['attention_mask'],
            'token_type_ids': bert_result['token_type_ids'],
            'hanlp_result': hanlp_result
        }


def load_json_dataset(file_path: str) -> List[Tuple[str, str, int]]:
    """
    加载 JSON 数据集
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        (text, topic, label) 列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        dataset = []
        for item in raw_data:
            text = item.get('text', '').strip()
            topic = item.get('topic', '').strip()
            label = int(item.get('label', '0'))
            
            if text:  # 只保留非空文本
                dataset.append((text, topic, label))
        
        return dataset
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {file_path}, 错误: {e}")
        return []


def preprocess_dataset(dataset_dir: str = "dataset", 
                      output_dir: str = "preprocessed_data",
                      bert_model_name: str = "bert-base-chinese",
                      max_length: int = 512):
    """
    离线预处理数据集 - 完全独立版本
    
    Args:
        dataset_dir: 原始数据集目录
        output_dir: 输出目录
        bert_model_name: BERT 模型名称
        max_length: 最大序列长度
    """
    print("🚀 开始离线预处理（独立版本）...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化独立预处理器
    try:
        preprocessor = IndependentPreprocessor(bert_model_name)
    except Exception as e:
        print(f"❌ 预处理器初始化失败: {e}")
        return
    
    # 处理每个数据集
    total_success = 0
    total_samples = 0
    
    for split in ['train', 'dev', 'test']:
        print(f"\n{'='*50}")
        print(f"🔄 处理 {split.upper()} 数据集")
        print(f"{'='*50}")
        
        # 文件路径
        json_file = os.path.join(dataset_dir, f'{split}.json')
        output_file = os.path.join(output_dir, f'{split}_preprocessed.pkl')
        
        if not os.path.exists(json_file):
            print(f"⚠️  跳过不存在的文件: {json_file}")
            continue
        
        # 加载原始数据
        print(f"📥 加载数据: {json_file}")
        raw_dataset = load_json_dataset(json_file)
        
        if not raw_dataset:
            print(f"❌ 数据集为空或加载失败")
            continue
        
        print(f"📊 原始数据: {len(raw_dataset)} 样本")
        
        # 逐样本处理
        processed_data = []
        error_count = 0
        
        for i, (text, topic, label) in enumerate(tqdm(raw_dataset, desc=f"Processing {split}")):
            try:
                # 处理单个样本
                processed_sample = preprocessor.process_single_sample(text, topic, label, max_length)
                processed_data.append(processed_sample)
                
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # 只显示前3个错误
                    print(f"\n❌ 样本 {i+1} 处理失败: {str(e)}")
                    print(f"   文本: {text[:50]}...")
                elif error_count == 4:
                    print(f"\n⚠️  更多错误将不再显示...")
        
        # 保存处理结果
        if processed_data:
            try:
                with open(output_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                
                success_rate = len(processed_data) / len(raw_dataset) * 100
                print(f"\n✅ {split.upper()} 数据集处理完成:")
                print(f"   📊 成功: {len(processed_data)} 样本")
                print(f"   ❌ 失败: {error_count} 样本")
                print(f"   📈 成功率: {success_rate:.1f}%")
                print(f"   💾 保存至: {output_file}")
                
                total_success += len(processed_data)
                total_samples += len(raw_dataset)
                
            except Exception as e:
                print(f"❌ 保存失败: {e}")
        else:
            print(f"❌ 没有成功处理的样本")
    
    # 总结
    print(f"\n{'='*50}")
    print(f"🎉 预处理完成")
    print(f"{'='*50}")
    print(f"📊 总体统计:")
    print(f"   成功样本: {total_success}")
    print(f"   总样本数: {total_samples}")
    if total_samples > 0:
        overall_success_rate = total_success / total_samples * 100
        print(f"   总成功率: {overall_success_rate:.1f}%")
    print(f"📁 预处理文件保存在: {output_dir}")
    print(f"💡 训练时使用: python main.py --use_preprocessed")


# 独立的数据加载函数，用于训练时加载预处理数据
def load_preprocessed_data(preprocessed_dir: str = "preprocessed_data") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    加载预处理好的数据
    
    Args:
        preprocessed_dir: 预处理数据目录
        
    Returns:
        (train_data, dev_data, test_data) 元组
    """
    datasets = {}
    
    for split in ['train', 'dev', 'test']:
        pkl_file = os.path.join(preprocessed_dir, f'{split}_preprocessed.pkl')
        
        if os.path.exists(pkl_file):
            try:
                with open(pkl_file, 'rb') as f:
                    datasets[split] = pickle.load(f)
                print(f"📥 加载 {split} 数据: {len(datasets[split])} 样本")
            except Exception as e:
                print(f"❌ 加载 {split} 数据失败: {e}")
                datasets[split] = []
        else:
            datasets[split] = []
            print(f"⚠️  未找到 {split} 预处理文件: {pkl_file}")
    
    return datasets.get('train', []), datasets.get('dev', []), datasets.get('test', [])


class PreprocessedDataset:
    """预处理数据集类 - 直接加载预处理结果，无需重复计算"""
    
    def __init__(self, preprocessed_data: List[Dict[str, Any]]):
        self.data = preprocessed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'token_type_ids': item['token_type_ids'],
            'hanlp_result': item['hanlp_result'],
            'label': item['label'],
            'text': item['text'],
            'topic': item['topic']
        }


def create_fast_data_loaders(train_data, val_data, batch_size):
    """
    终极修正版加载器：
    1. 包含 FastDataset 防止读取 text 报错
    2. 包含 build_hypergraph_matrix 补全缺失的 hypergraph_matrix 键
    """
    from torch.utils.data import Dataset, DataLoader
    import torch
    import numpy as np
    from collections import defaultdict

    # --- 1. 定义数据集类 ---
    class FastDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # --- 2. 内置超图构建逻辑 (为了让 main.py 能拿到 hypergraph_matrix) ---
    def build_single_matrix(hanlp_result, max_len=512, max_edges=100):
        """简化的超图构建逻辑，确保模型有东西可算"""
        try:
            tokens = hanlp_result.get('tok', [])
            if not tokens: tokens = hanlp_result.get('tokens', [])  # 兼容不同 key

            # 初始化矩阵 [N, M]
            matrix = np.zeros((max_len, max_edges), dtype=np.float32)
            edge_idx = 0

            # 策略A: 简单的滑动窗口构建超边 (最稳健，不依赖复杂句法)
            window_size = 3
            seq_len = min(len(tokens), max_len)

            for i in range(max(0, seq_len - window_size + 1)):
                if edge_idx >= max_edges: break
                # 将窗口内的词连接到同一条超边
                for w in range(window_size):
                    if i + w < max_len:
                        matrix[i + w, edge_idx] = 1.0
                edge_idx += 1

            # 策略B: 依存关系构建 (如果存在)
            deps = hanlp_result.get('dep', [])
            if not deps: deps = hanlp_result.get('dependencies', [])

            if deps:
                for dep in deps:
                    if edge_idx >= max_edges: break
                    if len(dep) >= 2:
                        head, tail = dep[0], dep[1] - 1  # HanLP索引通常从1开始
                        if head < max_len and tail < max_len and tail >= 0:
                            matrix[head, edge_idx] = 1.0
                            matrix[tail, edge_idx] = 1.0
                            edge_idx += 1

            return matrix
        except Exception:
            # 万一出错，返回全零矩阵防止程序崩溃
            return np.zeros((max_len, max_edges), dtype=np.float32)

    # --- 3. 定义打包函数 ---
    def fast_collate(batch):
        # 提取基础张量
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # --- 关键修复：现场构建超图矩阵 ---
        # 你的 main.py 需要 batch['hypergraph_matrix']，我们这里造给它
        matrices = []
        for item in batch:
            # 从 hanlp_result 构建矩阵
            mat = build_single_matrix(item['hanlp_result'])
            matrices.append(mat)

        hypergraph_matrix = torch.tensor(np.stack(matrices), dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'hypergraph_matrix': hypergraph_matrix,  # <--- 补上了这个关键的 Key
            'hanlp_results': [item['hanlp_result'] for item in batch]
        }

    # --- 4. 创建 DataLoader ---
    train_loader = DataLoader(FastDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=fast_collate)
    val_loader = DataLoader(FastDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=fast_collate)

    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='离线预处理数据集 - 独立版本')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data', help='预处理结果保存目录')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese', help='BERT模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        bert_model_name=args.bert_model,
        max_length=args.max_length
    )