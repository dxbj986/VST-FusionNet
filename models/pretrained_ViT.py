import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.models.layers import DropPath


class QueryAwareTransformer(nn.Module):
    def __init__(self,
                 emb_dim=768,
                 hidden_dim=1024,
                 num_layers=6,
                 num_heads=16,
                 output_dim=128,  # 最终输出的维度N
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()

        # 特征增强的全连接层
        self.fc = nn.Linear(emb_dim, hidden_dim)

        # Transformer块堆叠
        self.blocks = nn.Sequential(*[
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(num_layers)])

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 输出投影层
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, qvi_embeddings):
        """
        输入形状: [B, seq_len, emb_dim]
        输出形状: [B, N]
        """
        # 特征增强
        x = self.fc(qvi_embeddings)  # [B, seq_len, hidden_dim]

        # 通过Transformer层
        x = self.blocks(x)  # [B, seq_len, hidden_dim]

        # 全局平均池化
        x = x.mean(dim=1)  # [B, hidden_dim]

        # 层归一化 + 投影
        x = self.norm(x)
        x = self.proj(x)  # [B, output_dim]

        return x


class EnhancedViT(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            depth=12,
            num_heads=12,
            output_dim=128,  # 新增的目标输出维度
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.2, #0
            attn_drop_rate=0.2,#0
            drop_path_rate=0.5,#0
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            init_values=1e-4,
            use_conv_proj=True,
    ):
        super().__init__()

        # 增强的投影层（保持维度不变）
        self.proj = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2, bias=False),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim, bias=False),
            )
            if use_conv_proj
            else nn.Identity()
        )

        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 全局位置偏置

        # 深度可分离卷积增强
        self.depthwise_conv = (
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
            )
            if use_conv_proj
            else None
        )

        # 构建Transformer块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减

        self.blocks = nn.ModuleList([
            BlockWithConv(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                init_values=init_values,
                use_conv=(i % 4 == 0) and use_conv_proj,
            )
            for i in range(depth)
        ])

        # 输出规范化
        self.norm = norm_layer(embed_dim)

        # 新增输出层：将embed_dim映射到output_dim
        self.output_layer = nn.Linear(embed_dim, output_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.depthwise_conv is not None:
            nn.init.kaiming_normal_(self.depthwise_conv.weight)
        # 初始化输出层
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        B, N, D = x.shape  # 输入形状: [Batch, Sequence_Length, embed_dim]

        # 特征增强投影
        x = self.proj(x)  # 保持形状: [B, N, embed_dim]

        # 深度可分离卷积处理
        if self.depthwise_conv is not None:
            x = x + self.depthwise_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 添加全局位置偏置
        x = x + self.pos_embed.expand(B, -1, -1)  # 形状: [B, N, embed_dim]

        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x)  # 每个块的输出形状保持: [B, N, embed_dim]

        # 规范化
        x = self.norm(x)  # 形状: [B, N, embed_dim]

        # 全局平均池化 + 输出层
        x = x.mean(dim=1)  # 形状: [B, embed_dim]
        x = self.output_layer(x)  # 形状: [B, output_dim]

        return x


class BlockWithConv(Block):
    """增强的Transformer块，带卷积增强"""

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init_values=1e-4,
                 use_conv=False):
        # 调用父类构造函数并传递init_values参数
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            init_values=init_values  # 确保父类接收到正确的init_values
        )

        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

        # 卷积增强模块
        self.conv = None
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
                nn.BatchNorm1d(dim),
                nn.GELU()
            )
            self.conv_gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # 卷积增强分支
        if self.conv is not None:
            conv_x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + self.conv_gamma * conv_x

        # 原始Transformer分支
        x = x + self.drop_path1(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x