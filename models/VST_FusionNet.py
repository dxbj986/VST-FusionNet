import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
#from util.adj_metric import *
#from .my_layers import *
from models.img_transformations import STFTEmbedder, DelayEmbedder
from models.ViEncoder import ViTFeatureExtractor
from models.ts_Qformer import TransformerEncoder
from models.pretrained_ViT import QueryAwareTransformer, EnhancedViT
from util.data_vi import data_visual, visualize_attention, visualize_attention2, visualize_channel_heatmaps
from pyts.image import GramianAngularField

class MTSPatch(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """
        输入: x.shape = [Batch_size, node_num, slide_window_size]
        输出: patches.shape = [Batch_size, num_patches, node_num, patch_size]
        """
        B, N, T = x.shape  # Batch_size, node_num, slide_window_size
        assert T % self.patch_size == 0, "slide_window_size 必须能被 patch_size 整除"

        # 划分 patch
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # patches.shape = [B, N, num_patches, patch_size]

        # 调整维度顺序
        patches = patches.permute(0, 2, 1, 3)
        # patches.shape = [B, num_patches, N, patch_size]

        return patches


class PatchFeatureExtractor(nn.Module):
    def __init__(self, patch_size, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # 特征提取模块（1D CNN）
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化，压缩时间维度
        )

        # 特征映射层
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, patches):
        """
        输入: patches.shape = [B, num_patches, N, patch_size]
        输出: features.shape = [B, num_patches, N, feature_dim]
        """
        B, num_patches, N, patch_size = patches.shape

        # 将 patches 展平为 [B * num_patches * N, 1, patch_size]
        patches_flat = patches.reshape(-1, 1, patch_size)

        # 提取特征
        features_flat = self.conv(patches_flat)  # [B * num_patches * N, 64, 1]
        features_flat = features_flat.squeeze(-1)  # [B * num_patches * N, 64]
        features_flat = self.fc(features_flat)  # [B * num_patches * N, feature_dim]

        # 恢复原始形状
        features = features_flat.reshape(B, num_patches, N, self.feature_dim)
        return features


class SpatialAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 线性变换层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # 输出层
        self.out = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        """
        输入: features.shape = [B, num_patches, N, feature_dim]
        输出: spatial_features.shape = [B, num_patches, N, feature_dim]
        """
        B, num_patches, N, _ = features.shape

        # 线性变换
        Q = self.query(features)  # [B, num_patches, N, feature_dim]
        K = self.key(features)  # [B, num_patches, N, feature_dim]
        V = self.value(features)  # [B, num_patches, N, feature_dim]

        # 多头注意力机制
        Q = Q.view(B, num_patches, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [B, num_patches, num_heads, N, head_dim]
        K = K.view(B, num_patches, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [B, num_patches, num_heads, N, head_dim]
        V = V.view(B, num_patches, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [B, num_patches, num_heads, N, head_dim]

        # 计算注意力权重
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_patches, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn1=attn
        attn = self.dropout(attn)

        # 加权聚合
        out = torch.matmul(attn, V)  # [B, num_patches, num_heads, N, head_dim]
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # [B, num_patches, N, num_heads, head_dim]
        out = out.view(B, num_patches, N, self.feature_dim)  # [B, num_patches, N, feature_dim]

        # 输出变换
        spatial_features = self.out(out)  # [B, num_patches, N, feature_dim]
        return spatial_features, attn1


class PatchGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # GRU 模块
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, spatial_features):
        """
        输入: spatial_features.shape = [B, num_patches, N, feature_dim]
        输出: temporal_features.shape = [B, N, hidden_dim]
        """
        B, num_patches, N, _ = spatial_features.shape

        # 将变量维度展平为 [B * N, num_patches, feature_dim]
        features_flat = spatial_features.permute(0, 2, 1, 3)  # [B, N, num_patches, feature_dim]
        features_flat = features_flat.reshape(B * N, num_patches, -1)  # [B * N, num_patches, feature_dim]

        # GRU 时序建模
        gru_out, _ = self.gru(features_flat)  # [B * N, num_patches, hidden_dim * num_directions]

        # 如果是双向 GRU，合并前后向特征
        if self.bidirectional:
            gru_out = gru_out.view(B * N, num_patches, 2, self.hidden_dim)  # [B * N, num_patches, 2, hidden_dim]
            gru_out = torch.sum(gru_out, dim=2)  # [B * N, num_patches, hidden_dim]

        # 只保留最后一个时间步的特征
        last_step_features = gru_out[:, -1, :]  # [B * N, hidden_dim]

        # 恢复原始形状
        temporal_features = last_step_features.reshape(B, N, self.hidden_dim)  # [B, N, hidden_dim]
        return temporal_features

class OuterProductAggregation(nn.Module):
    def __init__(self, query_num: int,  emb_dim: int):
        super().__init__()
        self.query_num = query_num
        self.emb_dim = emb_dim
        self.proj_q = nn.Linear(emb_dim, 64)
        self.proj_p = nn.Linear(emb_dim, 64)
        # 定义聚合层（用于将外积结果映射回目标维度）
        self.aggregate_fc = nn.Linear(64 * 64, emb_dim)

    def forward(self, qvi_embeddings: torch.Tensor) -> torch.Tensor:
        """
        输入:
            qvi_embeddings: [B, query_num + patch_num, emb_dim]
        输出:
            patch_output: [B, patch_num, emb_dim]
        """
        # Step 1: 分割张量为查询和图像块部分
        queries = qvi_embeddings[:, :self.query_num, :]  # [B, query_num, emb_dim]
        patches = qvi_embeddings[:, self.query_num:, :]  # [B, patch_num, emb_dim]
        q_low = self.proj_q(queries)
        p_low = self.proj_p(patches)
        # Step 2: 计算外积（Query与Patch的交互）
        # 使用爱因斯坦求和公式高效计算
        # 输出形状: [B, query_num, patch_num, emb_dim, emb_dim]
        outer_product = torch.einsum("bqe,bpd->bqpde", q_low, p_low)

        # Step 3: 聚合操作（沿Query维度）
        # 展开外积结果 -> [B, query_num, patch_num, emb_dim*emb_dim]
        outer_flatten = outer_product.reshape(*outer_product.shape[:-2], -1)

        # 对Query维度求平均 -> [B, patch_num, emb_dim*emb_dim]
        aggregated = outer_flatten.mean(dim=1)  # 也可以改用sum或max

        # 通过全连接层压缩回目标维度 -> [B, patch_num, emb_dim]
        patch_output = self.aggregate_fc(aggregated)

        return patch_output


def random_mask(x_fea, vi_fea, mask_ratio_x=0.15, mask_ratio_vi=0.15):
    """
    对两个特征张量进行独立随机掩码
    Args:
        x_fea: 形状为 [B, N, tsf_dim] 的特征张量
        vi_fea: 形状为 [B, patch_num, embed_dim] 的特征张量
        mask_ratio_x: x_fea的掩码比例 (0-1)
        mask_ratio_vi: vi_fea的掩码比例 (0-1)
    Returns:
        掩码后的两个特征张量
    """
    # 确保两个张量在同一个设备上
    device = x_fea.device

    # 对x_fea进行掩码
    B, N, tsf_dim = x_fea.shape
    mask_x = torch.rand(B, N, device=device) < mask_ratio_x
    # 使用0值进行掩码，unsqueeze扩展维度以匹配特征维度
    x_masked = x_fea.masked_fill(mask_x.unsqueeze(-1), 0.0)

    # 对vi_fea进行掩码
    B_vi, patch_num, embed_dim = vi_fea.shape
    assert B == B_vi, "Batch size mismatch between two inputs"
    mask_vi = torch.rand(B, patch_num, device=device) < mask_ratio_vi
    # 使用随机噪声进行掩码（这里用正态分布，可根据需求修改）
    vi_masked = vi_fea.masked_fill(mask_vi.unsqueeze(-1), 0.0)

    return x_masked, vi_masked

class VST_FusionNet(nn.Module):
    def __init__(self,batch=32,node_num=44,hidden_dim=64,slide_win=56,ts_patch=8,slide_stride=8,tsf_dim=64,use_stft=False,delay=3,
                 embedding=32,n_fft=63,hop_length=23,img_size=224,patch_size=8,embed_dim=256,en_depth=2,en_num_heads=4,num_query_token=32,
                 freeze_vit=True,out_layer_num=1,out_layer_inter_dim=256,device='cuda'):
        super(VST_FusionNet, self).__init__()

        #MTS feature extractor
        self.patchify=MTSPatch(patch_size=ts_patch)
        self.extra_patch_fea=PatchFeatureExtractor(patch_size=ts_patch,feature_dim=tsf_dim)
        self.spatial_feature=SpatialAttention(feature_dim=tsf_dim, num_heads=en_num_heads)#要能整除
        self.s_t_feature=PatchGRU(feature_dim=tsf_dim,hidden_dim=tsf_dim)

        #Query
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, embed_dim)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

        #MTS2Image
        if not use_stft:
            self.delay = delay
            self.embedding = embedding
            self.seq_len = slide_win
            self.ts_img = DelayEmbedder(device, self.seq_len, self.delay, self.embedding)#[B,N,self.embedding,self.embedding]
        else:
            self.n_fft=n_fft
            self.hop_length=hop_length
            self.ts_img = STFTEmbedder(device, self.seq_len, self.n_fft, self.hop_length)

        #Image encoder
        self.vit_extractor = ViTFeatureExtractor(
            image_size=embedding,
            patch_size=patch_size,
            in_channels=node_num,
            patch_embedding_dim=embed_dim,
            vit_model_name="vit_large_patch32_384",  #vit_base_patch32_384 vit_tiny_patch16_224
            num_heads=en_num_heads

        )

        #MTS_Qformer
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            tsf_dim=tsf_dim,
            num_heads=en_num_heads,
            num_layers=en_depth
        )

        #pre_trained ViT
        #self.pretrained=QueryAwareTransformer(
        #    emb_dim=embed_dim,
        #    hidden_dim=out_layer_inter_dim,
        #    num_layers=en_depth,
        #    num_heads=en_num_heads,
        #    output_dim=node_num
        #)

        self.TensorFusion=OuterProductAggregation(query_num=num_query_token,emb_dim=embed_dim)

        self.pretrained_2=EnhancedViT(embed_dim=embed_dim,depth=en_depth,num_heads=en_num_heads,output_dim=node_num)


    def forward(self,x,test=False):
        #x.shape=[B,N,T]
        x_patch=self.patchify(x)
        x_patch_fea=self.extra_patch_fea(x_patch)
        x_spatial_fea, attn=self.spatial_feature(x_patch_fea)
        x_fea=self.s_t_feature(x_spatial_fea)#[B,N,tsf_dim]

        query_tokens = self.query_tokens.expand(x_fea.shape[0], -1, -1)

        print(x[0])
        img=self.ts_img.ts_to_img(x)#[B,N,embedding,embedding]
        visualize_channel_heatmaps(img)

        vi_fea=self.vit_extractor(img)#[B, patch_num, embed_dim]
        patch_num=vi_fea.shape[1]

        #x_masked, vi_masked=random_mask(x_fea,vi_fea,0.3,0.6)

        q_vi=torch.cat([query_tokens, vi_fea], dim=1)
        qvi_embeddings=self.transformer_encoder(q_vi, x_fea)#[B,query_num+patch_num, emb_dim]

        #q_vi = torch.cat([query_tokens, vi_masked], dim=1)
        #qvi_embeddings = self.transformer_encoder(q_vi, x_masked)  # [B,query_num+patch_num, emb_dim]

        #qvi_embeddings_Fusion=self.TensorFusion(qvi_embeddings)

        #output=self.pretrained(qvi_embeddings)
        output = self.pretrained_2(qvi_embeddings)
        #output = self.pretrained_2(qvi_embeddings_Fusion)
        #print(output.shape)

        if test:
            #print(attn.shape)
            visualize_attention2(attn=attn, batch_idx=0, use_first_head=True)


        return output




