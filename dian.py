import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. Prepare the data
# -------------------------
# List of attack methods
attack_methods = ["Clean", "FGSM", "PGD-20", "T-PGD", "BIM", "AutoAttack"]

# Replace these values with the actual data from your tables:
# Table II: Clean Teacher ResNet-56 on CIFAR-100
resnet56 = [71.72, 6.35, 0.00, 0.08, 0.00, 0.00]

# Table III: Adversarial Teacher WideResNet-34-10 on CIFAR-100
wide34_10 = [68.66, 56.11, 63.83, 64.12, 55.23, 26.80]

# Table IV: Clean Student ResNet-18 on CIFAR-100
resnet18 = [75.12, 7.10, 0.00, 0.18, 0.00, 0.00]

# Create x-axis positions for the attack methods
x = np.arange(len(attack_methods))

# -------------------------
# 2. Plotting
# -------------------------
plt.figure(figsize=(8, 5))  # Adjust the figure size as needed

# Plot each model's performance using lines and markers
plt.plot(x, resnet56, marker='o', label='ResNet-56 (Clean Teacher)')
plt.plot(x, wide34_10, marker='^', label='WideResNet-34-10 (Adv Teacher)')
plt.plot(x, resnet18, marker='s', label='ResNet-18 (Clean Student)')

# Set the x-axis ticks to be the attack methods
plt.xticks(x, attack_methods)

# Add grid, legend, and title
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.title("Model Performance Comparison under Various Attacks")

# Adjust the y-axis range if needed, for example from 0 to 85
plt.ylim([0, 85])

# Add labels for x and y axes
plt.xlabel("Attack Methods")
plt.ylabel("Accuracy (%)")

plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("model_performance.png", dpi=300)

plt.show()
