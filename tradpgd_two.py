import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import time
import torchattacks
import torchvision.utils as vutils
import os
from PIL import Image, ImageDraw, ImageFont
# 设置随机种子，保证实验可重复
def set_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # 多GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # Python随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 传统 PGD 攻击,调用 torchattacks

def traditional_pgd(model, images, labels, epsilon, alpha, num_iter, random_start=True):
    atk = torchattacks.PGD(
        model,
        eps=epsilon,
        alpha=alpha,
        steps=num_iter,
        random_start=random_start
    )
    adv_images = atk(images, labels)
    return adv_images


# def save_adversarial_images(adv_images, labels, attack_name, num_samples=10):
#     save_dir = f"adv_samples/{attack_name}/"
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 取前 num_samples 张图片
#     for i in range(min(num_samples, adv_images.size(0))):
#         img_path = os.path.join(save_dir, f"{attack_name}_sample_{i}_label_{labels[i].item()}.png")
#         vutils.save_image(adv_images[i].cpu(), img_path)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean


def save_adversarial_images_with_original(orig_images, adv_images, labels, predicted_labels, attack_name,
                                          num_samples=10):
    save_dir = f"adv_samples/{attack_name}/"
    os.makedirs(save_dir, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    for i in range(min(num_samples, adv_images.size(0))):
        orig_img = denormalize(orig_images[i].cpu(), mean, std).clamp(0, 1)
        adv_img = denormalize(adv_images[i].cpu(), mean, std).clamp(0, 1)
        true_label = labels[i].item()
        adv_label = predicted_labels[i].item()

        combined_img = torch.cat((orig_img, adv_img), dim=2)
        img_path = os.path.join(save_dir, f"{attack_name}_sample_{i}_true_{true_label}_adv_{adv_label}.png")
        vutils.save_image(combined_img, img_path, normalize=False)

        img = Image.open(img_path)
        img = img.resize((128, 64), Image.BICUBIC)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        text = f"True: {true_label} | Adv: {adv_label}"
        draw.text((5, 5), text, fill=(255, 255, 255), font=font)
        img.save(img_path)
# 将标签转换为 one-hot 格式
def to_onehot(labels, num_classes):

    if labels.dim() == 1:
        onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    elif labels.dim() == 4:
        B, dim, H, W = labels.shape
        assert dim == 1, f"Invalid 'labels' shape. Expected [B,1,H,W] but got {labels.shape}"
        onehot = F.one_hot(labels[:, 0].long(), num_classes=num_classes)
        onehot = onehot.permute(0, 3, 1, 2).float()
    else:
        raise ValueError("Unsupported label dimensions")
    return onehot


# DiceLoss 用于计算对抗优化中的损失（这里用于分类任务）
def DiceLoss(input, target, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5):

    intersection = torch.sum(target * input)
    if squared_pred:
        ground_o = torch.sum(target ** 2)
        pred_o = torch.sum(input ** 2)
    else:
        ground_o = torch.sum(target)
        pred_o = torch.sum(input)
    denominator = ground_o + pred_o
    dice_loss = 1.0 - (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)
    return dice_loss



# mpgd_attack
def mpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes):

    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=images.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=images.device).view(1, 3, 1, 1)
    # epsilon 与 alpha 转换到归一化空间（每个通道除以对应的 std）
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=images.device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=images.device).view(1, 3, 1, 1)
    # 初始化对抗样本
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)  # 输出 shape 为 [B, num_classes]
        # labels 形状为 [B]
        pred_labels = torch.argmax(outputs, dim=1)  # [B]
        correct_mask = (labels == pred_labels)  # 布尔型张量
        onehot = to_onehot(labels, num_classes)  # shape [B, num_classes]
        adv_pred_softmax = F.softmax(outputs, dim=1)  # shape [B, num_classes]

        # 分别计算正确预测样本和错误预测样本的 DiceLoss
        if correct_mask.sum() > 0:
            loss_correct = DiceLoss(adv_pred_softmax[correct_mask],
                                    onehot[correct_mask],
                                    squared_pred=True)
        else:
            loss_correct = 0.0
        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask],
                                  onehot[~correct_mask],
                                  squared_pred=True)
        else:
            loss_wrong = 0.0

        # 动态权重：随着迭代步数逐渐增加，lambda 由 0 逐步上升
        lam = i / (2.0 * num_iter)
        loss = (1 - lam) * loss_correct + lam * loss_wrong

        model.zero_grad()
        loss.backward()

        # 根据梯度符号更新对抗样本（归一化空间下更新）
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()

        # 对扰动进行投影，确保每个像素扰动不超过 epsilon（归一化空间下）
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images


# 定义用于 CIFAR-10 的 ResNet18 模型
class ResNet18_CIFAR10(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR10, self).__init__()
        # 使用在 ImageNet 上预训练的 ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 修改最后一层，全连接输出类别数为10
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


# 超参数设置
batch_size = 128
learning_rate = 1e-3
num_epochs = 40

epsilon = 8 / 255.0
alpha = 2 / 255.0


# 数据预处理和加载
def get_train_transforms():
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(saturation=0.1, contrast=0.1, brightness=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return train_transform


def get_test_transforms():
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return test_transform


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=get_train_transforms())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=get_test_transforms())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型训练（若已存在权重则直接加载）
model = ResNet18_CIFAR10().to(device)
os.makedirs("checkpoints", exist_ok=True)
model_path = "checkpoints/modified_pgd_resnet18_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded pre-trained model from", model_path)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("=== Start Standard Training ===")
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    end_time = time.time()
    print("Training time: {:.2f} seconds".format(end_time - start_time))
    torch.save(model.state_dict(), model_path)
    print(">>> Standard Trained Model Saved To", model_path)


# 定义评估函数
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# 评估不同迭代次数下的攻击效果：传统 PGD 与 mpgd_attack
iterations_list = [1, 5, 10, 20, 30, 50, 60, 80, 100]


def evaluate_attack_with_saving(model, loader, attack_name, epsilon, alpha, iterations, **kwargs):
    model.eval()
    correct = 0
    total = 0
    saved = False

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_name.lower() == 'traditional_pgd':
            adv_images = traditional_pgd(model, images, labels, epsilon, alpha, iterations, random_start=True)
        elif attack_name.lower() == 'mpgd':
            mean = kwargs.get('mean', (0.4914, 0.4822, 0.4465))
            std = kwargs.get('std', (0.2023, 0.1994, 0.2010))
            num_classes = kwargs.get('num_classes', 10)
            adv_images = mpgd_attack(model, images, labels, epsilon, alpha, iterations, mean, std, num_classes)
        else:
            adv_images = images

        outputs = model(adv_images)
        _, predicted_labels = torch.max(outputs, 1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

        if not saved:
            save_adversarial_images_with_original(images, adv_images, labels, predicted_labels, attack_name,
                                                  num_samples=10)
            saved = True

    return 100.0 * correct / total



# 评估并保存对抗样本
print("\n=== Evaluating Traditional PGD Attack with Saving ===")
for iters in iterations_list:
    acc_trad = evaluate_attack_with_saving(model, test_loader, attack_name='traditional_pgd',
                                           epsilon=epsilon, alpha=alpha, iterations=iters)
    print(f"Traditional PGD - Iterations: {iters:<3d} | Accuracy: {acc_trad:.2f}%")

print("\n=== Evaluating mpgd Attack with Saving ===")
for iters in iterations_list:
    acc_mpgd = evaluate_attack_with_saving(model, test_loader, attack_name='mpgd',
                                           epsilon=epsilon, alpha=alpha, iterations=iters,
                                           mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                                           num_classes=10)
    print(f"mpgd Attack      - Iterations: {iters:<3d} | Accuracy: {acc_mpgd:.2f}%")

print("\n=== Done. Adversarial samples saved. ===")

# 对 clean 数据进行评估
acc_clean = evaluate(model, test_loader)
print("\nClean Test Accuracy: {:.2f}%".format(acc_clean))



print("\n=== Evaluating Traditional PGD Attack ===")
for iters in iterations_list:
    acc_trad = evaluate_attack_with_saving(model, test_loader, attack_name='traditional_pgd',
                               epsilon=epsilon, alpha=alpha, iterations=iters)
    print(f"Traditional PGD - Iterations: {iters:<3d} | Accuracy: {acc_trad:.2f}%")

print("\n=== Evaluating mpgd Attack ===")
for iters in iterations_list:
    acc_mpgd = evaluate_attack_with_saving(model, test_loader, attack_name='mpgd',
                               epsilon=epsilon, alpha=alpha, iterations=iters,
                               mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                               num_classes=10)
    print(f"mpgd Attack      - Iterations: {iters:<3d} | Accuracy: {acc_mpgd:.2f}%")

print("\n=== Done. ===")
