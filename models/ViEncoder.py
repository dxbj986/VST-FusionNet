import torch
import torch.nn as nn
import timm


class ViTFeatureExtractor(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, patch_embedding_dim, vit_model_name="vit_tiny_patch16_224",num_heads=4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embedding_dim = patch_embedding_dim

        # 计算 patch 数量
        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
        self.num_patches = (image_size // patch_size) ** 2

        # 使用 timm 的 ViT
        self.vit = timm.create_model(
            vit_model_name,
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=patch_embedding_dim,
            pretrained=False,# 是否加载预训练权重
            num_heads=num_heads
        )

    def forward(self, x):
        """
        输入: x.shape = [B, N, embedding, embedding]
        输出: features.shape = [B, num_patches, patch_embedding_dim]
        """
        B, N, H, W = x.shape
        assert H == W == self.image_size, "输入图像大小与设定不符"

        # 将输入转换为 [B, N, H, W]
        x = x.reshape(B, N, H, W)

        # 使用 ViT 提取特征
        features = self.vit.forward_features(x)  # [B, num_patches + 1, patch_embedding_dim]

        return features