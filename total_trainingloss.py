import pickle
import matplotlib.pyplot as plt
import numpy as np

# ======== 1. 读取三个.pkl文件并加载loss列表 ========
with open("PGDARD_epoch_losses_e100.pkl", "rb") as f:
    baseline_anl_loss = pickle.load(f)

with open("DARD_epoch_losses_e100.pkl", "rb") as f:
    baseline_loss = pickle.load(f)

with open("Onlyadv_epoch_loss_list.pkl_e100", "rb") as f:
    baseline_mt_loss = pickle.load(f)



# ======== 2. 画图并保存 ========
plt.figure(figsize=(8, 6))
epochs = range(1, len(baseline_loss) + 1)

# 直接使用原始数据，不做任何平滑
plt.plot(epochs, baseline_anl_loss, label="PGDARD", color="blue")
plt.plot(epochs, baseline_mt_loss, label="OnlyadvlaARD", color="orange")
plt.plot(epochs, baseline_loss, label="DARD", color="green")

plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)

# 保存图片（可根据需要修改 dpi、文件名、格式等）
plt.savefig("loss_comparison_3.png", dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
