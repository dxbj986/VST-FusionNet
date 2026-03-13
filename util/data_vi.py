import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from math import ceil, sqrt
import seaborn as sns

def data_visual(X):
    # # 数据集导入
    # X, y = datasets.load_digits(return_X_y=True)

    # t-SNE降维处理
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    result = tsne.fit_transform(X)
    yy=KMeans(n_clusters=10).fit(X)

    # 归一化处理
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    result = scaler.fit_transform(result)
    #print(result.shape)

    # 颜色设置
    #color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400',
    #         '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']
    color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400']
    # 可视化展示
    plt.figure(figsize=(7, 7))
    plt.title('t-SNE process')
    # plt.xlim((-1.1, 1.1))
    # plt.ylim((-1.1, 1.1))
    # for i in range(len(result)):
    #     plt.text(result[i,0], result[i,1], str(y[i]),
    #              color=color[y[i]], fontdict={'weight': 'bold', 'size': 9})

    plt.scatter(result[:, 0], result[:, 1], c=yy.labels_,cmap='plasma' ,s=40)
    plt.show()
    return

def visualize_attention(attn, batch_idx=0, use_first_head=False, normalize_per_map=True):
    """
    可视化注意力热力图（适配PyTorch/TensorFlow张量输入）

    Parameters:
        attn (Tensor): 输入张量，形状为 [B, num_patches, num_heads, N, N]
        batch_idx (int): 选择要可视化的批次样本索引
        use_first_head (bool): 若为True则只使用第一个注意力头，否则平均所有头
        normalize_per_map (bool): 是否对每个热力图独立归一化
    """
    # 将张量转换为NumPy数组（适配PyTorch和TensorFlow）
    if 'torch' in str(type(attn)):  # PyTorch张量
        attn_np = attn.detach().cpu().numpy()
    elif 'tensorflow' in str(type(attn)):  # TensorFlow张量
        attn_np = attn.numpy()
    else:
        attn_np = attn  # 假设已经是NumPy数组

    # 选择批次样本并处理多头
    sample_attn = attn_np[batch_idx]  # [num_patches, num_heads, N, N]

    if use_first_head:
        processed_attn = sample_attn[:, 0, :, :]  # 取第一个头
        title_suffix = " (First Head)"
    else:
        processed_attn = np.mean(sample_attn, axis=1)  # 平均所有头
        title_suffix = " (Averaged Heads)"

    num_patches, N, _ = processed_attn.shape

    # 创建子图布局
    n_rows = int(np.sqrt(num_patches))
    n_cols = int(np.ceil(num_patches / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    fig.suptitle(f'Attention Maps for Batch Sample {batch_idx}' + title_suffix, fontsize=14)

    # 全局归一化参数
    vmin = processed_attn.min() if not normalize_per_map else None
    vmax = processed_attn.max() if not normalize_per_map else None

    # 绘制每个patch的热力图
    for i, ax in enumerate(axes.flat):
        if i < num_patches:
            attn_map = processed_attn[i]
            if normalize_per_map:
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            im = ax.imshow(attn_map, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Patch {i}', fontsize=6)
            ax.axis('off')
        else:
            ax.axis('off')

    # 添加颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.show()

# 示例用法（PyTorch）
# import torch
# attn_tensor = torch.randn(2, 16, 8, 14, 14)  # 示例张量
# visualize_attention(attn_tensor, batch_idx=0, use_first_head=True)

def visualize_attention2(attn,
                        batch_idx=0,
                        use_first_head=False,
                        normalize_per_map=False,
                        base_figsize=4,
                        max_cols=6,
                        dpi=100):
    """
    完整版注意力热力图可视化函数

    Parameters:
        attn (Tensor/np.ndarray): 输入张量 [B, num_patches, num_heads, N, N]
        batch_idx (int): 选择要可视化的批次索引
        use_first_head (bool): 是否只使用第一个注意力头
        normalize_per_map (bool): 是否独立归一化每个热力图
        base_figsize (int): 单个子图的基础尺寸（控制整体缩放）
        max_cols (int): 每行最大显示列数（防止过多导致重叠）
        dpi (int): 输出分辨率（建议100-300）
    """
    # 张量转换处理
    if 'torch' in str(type(attn)):
        attn_np = attn.detach().cpu().numpy()
    elif 'tensorflow' in str(type(attn)):
        attn_np = attn.numpy()
    else:
        attn_np = attn

    # 选择批次样本并处理多头
    sample_attn = attn_np[batch_idx]
    num_patches = sample_attn.shape[0]

    # 处理多头
    if use_first_head:
        processed_attn = sample_attn[:, 0, :, :]
        title_suffix = " (First Head)"
    else:
        processed_attn = np.mean(sample_attn, axis=1)
        title_suffix = " (Averaged Heads)"

    # 自动布局计算
    n_cols = min(num_patches, max_cols)
    n_rows = ceil(num_patches / n_cols)

    # 创建画布（动态尺寸计算）
    fig_width = n_cols * base_figsize * 1.1  # 10%边距
    fig_height = n_rows * base_figsize * 1.2  # 20%标题空间
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_width, fig_height),
                             dpi=dpi,
                             gridspec_kw={'wspace': 0.4, 'hspace': 0.6})

    # 处理子图对象
    axes = axes.flat if n_rows > 1 else [axes]

    # 绘制每个patch
    for i, ax in enumerate(axes):
        if i < num_patches:
            attn_map = processed_attn[i]

            # 归一化处理
            if normalize_per_map:
                attn_map = (attn_map - attn_map.min()) / \
                           (attn_map.max() - attn_map.min() + 1e-8)

            # 可视化主逻辑
            im = ax.imshow(attn_map, cmap='viridis')
            ax.set_title(f'Patch {i}\nSize: {attn_map.shape}',
                         fontsize=base_figsize * 2,
                         pad=base_figsize * 0.5)
            ax.axis('off')
        else:
            ax.axis('off')  # 隐藏多余子图

    # 颜色条动态定位
    cbar_left = 0.85 if n_cols <= 4 else 0.92
    cbar_ax = fig.add_axes([cbar_left, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # 主标题和布局优化
    fig.suptitle(f'Attention Maps (Batch {batch_idx}){title_suffix}\n'
                 f'Total Patches: {num_patches}, Map Size: {attn_map.shape}',
                 y=0.95, fontsize=base_figsize * 4)
    plt.tight_layout(rect=[0, 0, cbar_left - 0.05, 0.95])
    plt.show()


def visualize_channel_heatmaps(input_tensor):
    """
    可视化输入张量的每个通道
    参数:
        input_tensor: 形状为 [B, N, embedding, embedding] 的四维张量
    """
    # 确保输入是NumPy数组
    if not isinstance(input_tensor, np.ndarray):
        input_tensor = np.array(input_tensor.cpu())

    # 取批次中的第一个样本 (批量索引=0)
    batch_sample = input_tensor[0]
    num_channels = batch_sample.shape[0]

    for channel_idx in range(num_channels):
        # 获取当前通道的数据
        channel_data = batch_sample[channel_idx]

        # 创建新图形
        plt.figure(figsize=(8, 6))

        # 绘制热力图
        ax = sns.heatmap(
            channel_data,
            cmap='viridis',
            annot=True,  # 在单元格中显示数值
            fmt=".2f",  # 数值格式
            annot_kws={"size": 6},  # 数值字体大小
            cbar=True,  # 显示颜色条
            square=True  # 保持单元格为正方形
        )

        # 设置标题和标签
        plt.title(f'Channel {channel_idx + 1}/{num_channels} Heatmap')
        plt.xlabel('Width Position')
        plt.ylabel('Height Position')

        # 显示图形
        plt.show()