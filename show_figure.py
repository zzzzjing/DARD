import numpy as np
import matplotlib.pyplot as plt

# 攻击方法作为 x 轴标签
attack_methods = ['Clean', 'FGSM', 'PGD20', 'T-PGD', 'BIM', 'AutoAttack']

# 各防御方法在不同攻击下的准确率数据
natural = [75.12, 7.10, 0.08, 0.18, 0.00, 0.00]
sat     = [58.29, 28.08, 22.89, 54.94, 22.99, 20.50]
ard     = [59.23, 29.43, 24.69, 53.24, 21.46, 20.78]
cad     = [60.33, 31.36, 26.77, 55.68, 26.78, 22.08]

# x 轴位置
x = np.arange(len(attack_methods))
width = 0.2  # 每个柱子的宽度

plt.figure(figsize=(10, 6))  # 设置图表大小

# 绘制分组柱状图，每个柱子代表一种防御方法
plt.bar(x - 1.5 * width, natural, width, label='Natural')
plt.bar(x - 0.5 * width, sat, width, label='SAT')
plt.bar(x + 0.5 * width, ard, width, label='ARD')
plt.bar(x + 1.5 * width, cad, width, label='CAD')

# 设置 x 轴刻度标签
plt.xticks(x, attack_methods)
plt.ylabel('Accuracy (%)')
plt.title('White-box Robustness of ResNet-18 on CIFAR-100')

# 显示图例
plt.legend()

# 自动调整布局
plt.tight_layout()

# 保存图表为图片文件（PNG格式）
plt.savefig('bar_chart_1.png', dpi=300)

# 显示图表
plt.show()
