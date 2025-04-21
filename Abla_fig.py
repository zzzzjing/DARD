import numpy as np
import matplotlib.pyplot as plt

# 横坐标标签
labels = ['FGSM', 'PGD20', 'T-PGD', 'BIM', 'AutoAttack']

# 三组数据
pgdard = [69.80, 66.83, 80.96, 66.83, 64.84]
onlyadvlaard = [70.16, 67.64, 82.17, 67.61, 65.13]
dard = [70.55, 68.10, 82.34, 68.32, 65.66]

# 柱子的宽度（较宽）
width = 0.4

# 每组之间需要足够的间距，这里定义每组的起始 x 坐标
n_groups = len(labels)
group_spacing = 1.4  # 每组的间隔
x = np.arange(n_groups) * group_spacing  # 每组的起始 x 坐标

# 定义每组内部三个柱子的 x 坐标
x1 = x            # 第一组柱子位置
x2 = x + width    # 第二组柱子位置
x3 = x + 2 * width  # 第三组柱子位置

# 颜色设置（深色）
colors = {
    "PGDARD": "#1f77b4",      # 深蓝色
    "OnlyadvlaARD": "#ff7f0e", # 深橙色
    "DARD": "#2ca02c"          # 深绿色
}

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x1, pgdard, width=width, label='PGDARD', color=colors["PGDARD"])
bars2 = plt.bar(x2, onlyadvlaard, width=width, label='OnlyadvlaARD', color=colors["OnlyadvlaARD"])
bars3 = plt.bar(x3, dard, width=width, label='DARD', color=colors["DARD"])

# 在每个柱子上显示准确率
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)

# 设置 x 轴刻度为每组柱子的中心位置
plt.xticks(x + width, labels, fontsize=12)

# 设置纵坐标范围和刻度
plt.ylim(62, 84)
plt.yticks(np.arange(62, 85, 2), fontsize=12)

# 添加图例、标题和坐标轴标签
plt.xlabel("Attack Type", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
#plt.title("Comparison of Adversarial Training Methods", fontsize=14)
plt.legend(fontsize=12)

# 保存图像（不显示图片）
plt.savefig("Abla_fig4.png", dpi=300, bbox_inches='tight')
