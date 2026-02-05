"""
模型定义模块
定义 BERT + HGNN + Attention 的网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Optional, Tuple
import math


class HypergraphConvolution(nn.Module):
    """
    超图卷积层 - 修正：实现真正的超图卷积公式
    X^{(l+1)} = σ(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} Θ)
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """
        初始化超图卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout率
        """
        super(HypergraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重矩阵 Θ
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, hypergraph_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 修正：支持批处理的3D超图矩阵
        
        Args:
            x: 输入特征 [batch_size, num_nodes, in_features]
            hypergraph_matrix: 超图关联矩阵 H [batch_size, num_nodes, num_hyperedges]
            
        Returns:
            输出特征 [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.size()
        num_hyperedges = hypergraph_matrix.size(2)
        
        # 批处理计算度矩阵
        # D_v: 节点度矩阵 (每个节点连接的超边数) [batch_size, num_nodes]
        d_v = torch.sum(hypergraph_matrix, dim=2)  # [batch_size, num_nodes]
        d_v = torch.clamp(d_v, min=1e-8)  # 避免除零
        d_v_inv_sqrt = torch.pow(d_v, -0.5)  # D_v^{-1/2} [batch_size, num_nodes]
        
        # D_e: 超边度矩阵 (每条超边连接的节点数) [batch_size, num_hyperedges]
        d_e = torch.sum(hypergraph_matrix, dim=1)  # [batch_size, num_hyperedges]
        d_e = torch.clamp(d_e, min=1e-8)  # 避免除零
        d_e_inv = torch.pow(d_e, -1.0)  # D_e^{-1} [batch_size, num_hyperedges]
        
        # 批处理超图卷积: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        H = hypergraph_matrix  # [batch_size, num_nodes, num_hyperedges]
        
        # 第一步: D_v^{-1/2} H - 使用广播机制
        step1 = d_v_inv_sqrt.unsqueeze(2) * H  # [batch_size, num_nodes, num_hyperedges]
        
        # 第二步: (D_v^{-1/2} H) D_e^{-1} - 使用广播机制
        step2 = step1 * d_e_inv.unsqueeze(1)  # [batch_size, num_nodes, num_hyperedges]
        
        # 第三步: (D_v^{-1/2} H D_e^{-1}) H^T - 使用批矩阵乘法
        step3 = torch.bmm(step2, H.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]
        
        # 第四步: (D_v^{-1/2} H D_e^{-1} H^T) D_v^{-1/2} - 使用广播机制
        normalized_matrix = step3 * d_v_inv_sqrt.unsqueeze(1)  # [batch_size, num_nodes, num_nodes]
        
        # 应用到输入特征
        # X^{(l)} Θ
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # 超图卷积: normalized_matrix @ support - 使用批矩阵乘法
        output = torch.bmm(normalized_matrix, support)  # [batch_size, num_nodes, out_features]
        
        # 添加偏置
        output = output + self.bias
        
        # Dropout
        output = self.dropout(output)
        
        return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑并连接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        return output


class HGNN(nn.Module):
    """超图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        """
        初始化HGNN
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            dropout: dropout率
        """
        super(HGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(HypergraphConvolution(dims[i], dims[i+1], dropout))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hypergraph_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            hypergraph_matrix: 超图邻接矩阵
            
        Returns:
            输出特征
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, hypergraph_matrix)
            if i < len(self.layers) - 1:  # 最后一层不使用激活函数
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


class BertHGNNModel(nn.Module):
    """BERT + HGNN + Attention 融合模型"""
    
    def __init__(self, 
                 bert_model_name: str = 'bert-base-chinese',
                 hgnn_hidden_dims: list = [512, 256],
                 num_attention_heads: int = 8,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 freeze_bert: bool = False):
        """
        初始化模型
        
        Args:
            bert_model_name: BERT模型名称
            hgnn_hidden_dims: HGNN隐藏层维度
            num_attention_heads: 注意力头数
            num_classes: 分类类别数
            dropout: dropout率
            freeze_bert: 是否冻结BERT参数
        """
        super(BertHGNNModel, self).__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # 超图神经网络
        self.hgnn = HGNN(
            input_dim=bert_hidden_size,
            hidden_dims=hgnn_hidden_dims,
            dropout=dropout
        )
        
        # 多头注意力
        final_dim = hgnn_hidden_dims[-1] if hgnn_hidden_dims else bert_hidden_size
        self.attention = MultiHeadAttention(
            d_model=final_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # 特征融合层
        self.fusion_layer = nn.Linear(bert_hidden_size + final_dim, final_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                hypergraph_matrix: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            hypergraph_matrix: 超图关联矩阵 H [num_nodes, num_hyperedges]
            token_type_ids: token类型IDs [batch_size, seq_len]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        bert_sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        bert_pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        
        # HGNN处理 - 使用修正后的超图卷积
        hgnn_output = self.hgnn(bert_sequence_output, hypergraph_matrix)  # [batch_size, seq_len, hgnn_dim]
        
        # 多头注意力
        attention_output = self.attention(hgnn_output, hgnn_output, hgnn_output, 
                                        mask=attention_mask.unsqueeze(1).unsqueeze(2))
        
        # 池化操作（取[CLS]位置或平均池化）
        attention_pooled = attention_output[:, 0, :]  # 取[CLS]位置
        
        # 特征融合
        fused_features = self.fusion_layer(
            torch.cat([bert_pooled_output, attention_pooled], dim=-1)
        )
        fused_features = self.dropout(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_attention_weights(self, 
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            hypergraph_matrix: torch.Tensor,
                            token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取注意力权重用于可视化"""
        with torch.no_grad():
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            bert_sequence_output = bert_outputs.last_hidden_state
            hgnn_output = self.hgnn(bert_sequence_output, hypergraph_matrix)
            
            # 计算注意力权重
            batch_size, seq_len = input_ids.size()
            Q = self.attention.w_q(hgnn_output)
            K = self.attention.w_k(hgnn_output)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention.d_k)
            attention_weights = F.softmax(scores, dim=-1)
            
            return attention_weights