import matplotlib.pyplot as plt
import pandas as pd

# 创建数据框
data = {
    "w": [8,16,24,32,40,48,56,64,72,80],
    "SWaT": [87.96,89.48,91.02,92.01,94.88,95.24,95.67,93.35,90.58,89.84],
    "HAI": [85.24,87.91,90.12,91.87,92.53,94.05,95.12,95.17,94.68,92.94],
    "SKAB": [89.45,91.67,93.83,97.96,96.12,94.35,93.64,92.41,91.06,89.75],
    "SMD": [90.38,92.13,93.65,96.24,96.81,96.41,95.85,94.67,92.83,91.72],
    "PUMP": [90.74,92.48,94.72,95.67,96.12,96.01,96.38,95.91,94.83,93.92],
    "SMAP": [91.52,93.47,95.39,97.24,98.1,97.93,97.56,96.72,95.43,94.18]
}
df = pd.DataFrame(data)

# 创建可视化
plt.figure(figsize=(13,7), dpi=100)
color_palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

# 绘制主趋势线
for idx, col in enumerate(df.columns[1:]):
    plt.plot(df["w"], df[col],  # 使用"w"而不是"g"
             marker='o',
             markersize=7,
             linewidth=2.5,
             color=color_palette[idx],
             label=col)

# 设置图表元素
plt.title("F1-Score Trends with Window Size (w)", pad=20, fontsize=14)  # 更新标题
plt.xlabel("Window Size (w)", fontsize=12)  # 更新x轴标签
plt.ylabel("F1-Score (%)", fontsize=12)
plt.xticks(range(0, 81, 10))  # 设置x轴刻度从0到80，间隔为10
plt.yticks(range(85, 101, 2))
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.12, 1), fontsize=10)
plt.xlim(5, 85)  # 设置x轴范围，稍微超出数据范围
plt.ylim(85, 100)

plt.tight_layout()
plt.show()