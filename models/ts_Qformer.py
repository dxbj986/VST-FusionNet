import torch
import torch.nn as nn
import torch.nn.functional as F
from .T6_infer import TPA, TPACrossAttention
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 64
    max_seq_len: int = 256
    head_dim: int = -1
    q_rank: int = 6
    rank: int = 2
    using_groupnorm: bool = False

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, tsf_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.tsf_dim = tsf_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q 的投影（来自 embed_dim）
        self.query = nn.Linear(embed_dim, embed_dim)
        # K 和 V 的投影（来自 tsf_dim）
        self.key = nn.Linear(tsf_dim, embed_dim)
        self.value = nn.Linear(tsf_dim, embed_dim)

        # 输出层
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        输入: q.shape = [B, seq_len_q, embed_dim] （来自 Self-Attention）
              k.shape = [B, seq_len_k, tsf_dim]   （来自 x_fea）
              v.shape = [B, seq_len_k, tsf_dim]   （来自 x_fea）
        输出: out.shape = [B, seq_len_q, embed_dim]
        """
        B, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        # 线性投影
        Q = self.query(q)  # [B, seq_len_q, embed_dim]
        K = self.key(k)  # [B, seq_len_k, embed_dim]
        V = self.value(v)  # [B, seq_len_k, embed_dim]

        # 多头注意力机制
        Q = Q.view(B, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [B, num_heads, seq_len_q, head_dim]
        K = K.view(B, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [B, num_heads, seq_len_k, head_dim]
        V = V.view(B, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [B, num_heads, seq_len_k, head_dim]

        # 计算注意力权重
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, seq_len_q, seq_len_k]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权聚合
        out = torch.matmul(attn, V)  # [B, num_heads, seq_len_q, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, seq_len_q, num_heads, head_dim]
        out = out.view(B, seq_len_q, self.embed_dim)  # [B, seq_len_q, embed_dim]

        # 输出变换
        out = self.out(out)  # [B, seq_len_q, embed_dim]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, tsf_dim, num_heads, dropout=0.1):
        super().__init__()
        # Self-Attention（输入输出均为 embed_dim）
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.t_self_attn = TPA(ModelArgs)

        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-Attention（处理 tsf_dim -> embed_dim）
        self.cross_attn = MultiHeadCrossAttention(embed_dim, tsf_dim, num_heads, dropout)
        #self.cross_attn = TPACrossAttention(embed_dim,tsf_dim,num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward（保持 embed_dim）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_fea):
        """
        输入: x.shape = [B, seq_len, embed_dim]       （拼接后的序列）
              x_fea.shape = [B, N, tsf_dim]           （外部特征）
        输出: out.shape = [B, seq_len, embed_dim]
        """
        # Self-Attention
        #attn_out, _ = self.self_attn(x, x, x)  # [B, seq_len, embed_dim]
        attn_out=self.t_self_attn(x,x,x,0,None)

        x = x + self.dropout(self.norm1(attn_out))

        # Cross-Attention
        cross_attn_out = self.cross_attn(x, x_fea, x_fea)  # [B, seq_len, embed_dim]
        x = x + self.dropout(self.norm2(cross_attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        out = x + self.dropout(self.norm3(ffn_out))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, tsf_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, tsf_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, x_fea):
        for layer in self.layers:
            x = layer(x, x_fea)
        return x