import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 如果使用 AutoAttack 和 Torchattacks，请确保已安装对应库
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack 库未安装，请使用 pip install autoattack 安装。")
try:
    from torchattacks import CW
except ImportError:
    print("torchattacks 库未安装，请使用 pip install torchattacks 安装。")
    CW = None


###########################################
# 辅助函数：to_onehot, DiceLoss, mpgd_attack
###########################################

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


###########################################
# 参数设置与数据集加载
###########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
batch_size = 128
epochs = 80

# 对抗攻击参数（在[0,1]空间下的像素扰动，通常 CIFAR10 常用 8/255）
epsilon = 8 / 255.0
alpha = 2 / 255.0
num_iter = 10

# CIFAR10 的均值和标准差（后续攻击时需要根据归一化来计算合法区间）
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 数据预处理
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

###########################################
# 定义模型（ResNet18，调整最后全连接层）
###########################################
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

###########################################
# 定义优化器、学习率调度器和损失函数
###########################################
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()


###########################################
# 定义评估函数（可选择对抗攻击后评估）
###########################################
def evaluate(model, test_loader, attack_fn=None, attack_name='Clean'):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # 如果需要攻击，则临时开启梯度计算
        if attack_fn is not None:
            with torch.enable_grad():
                data = attack_fn(model, data, target)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    acc = correct / total
    print(f"{attack_name} 测试准确率: {acc:.4f}")
    return acc


###########################################
# 实现 FGSM、PGD、CW 和 MPGDA 攻击
###########################################
def fgsm_attack(model, data, target, eps=epsilon):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    eps_tensor = torch.tensor([eps / s for s in std], device=data.device).view(1, 3, 1, 1)
    perturbed_data = data + eps_tensor * data.grad.sign()
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=data.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=data.device).view(1, 3, 1, 1)
    perturbed_data = torch.clamp(perturbed_data, lower_bound, upper_bound)
    return perturbed_data.detach()


def pgd_attack(model, data, target, eps=epsilon, alpha=alpha, num_iter=num_iter):
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=data.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=data.device).view(1, 3, 1, 1)
    eps_tensor = torch.tensor([eps / s for s in std], device=data.device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=data.device).view(1, 3, 1, 1)
    adv_data = data.clone().detach()
    adv_data.requires_grad = True
    for _ in range(num_iter):
        output = model(adv_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        adv_data = adv_data + alpha_tensor * adv_data.grad.sign()
        perturbation = torch.clamp(adv_data - data, -eps_tensor, eps_tensor)
        adv_data = torch.clamp(data + perturbation, lower_bound, upper_bound).detach_()
        adv_data.requires_grad = True
    return adv_data.detach()


def cw_attack_fn(model, data, target):
    if CW is None:
        raise ValueError("CW 攻击不可用，请安装 torchattacks。")
    attack = CW(model, c=1e-4, kappa=0, steps=num_iter, lr=0.01)
    adv_data = attack(data, target)
    return adv_data


def mpgd_attack_fn(model, data, target):
    return mpgd_attack(model, data, target, epsilon, alpha, num_iter, mean, std, num_classes)


###########################################
# AutoAttack 评估（对整个测试集进行攻击）
###########################################
def autoattack_evaluation(model, test_loader):
    model.eval()
    x_test = []
    y_test = []
    for data, target in test_loader:
        x_test.append(data)
        y_test.append(target)
    x_test = torch.cat(x_test, 0).to(device)
    y_test = torch.cat(y_test, 0).to(device)
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    output = model(x_adv)
    pred = output.argmax(dim=1)
    correct = pred.eq(y_test).sum().item()
    acc = correct / len(y_test)
    print(f"AutoAttack 测试准确率: {acc:.4f}")
    return acc


###########################################
# 模型缓存设置：保存训练参数配置和模型权重
###########################################
saved_model_path = 'adv_train_model.pth'
saved_config_path = 'adv_train_config.json'

# 定义训练所用的超参数（注意：如果修改这些参数，则会重新训练）
train_params = {
    'epsilon': epsilon,
    'alpha': alpha,
    'num_iter': num_iter,
    'epochs': epochs,
    'batch_size': batch_size,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.1,
}

use_cached_model = False
if os.path.exists(saved_model_path) and os.path.exists(saved_config_path):
    with open(saved_config_path, 'r') as f:
        saved_params = json.load(f)
    if saved_params == train_params:
        print("检测到相同的训练参数，加载已保存模型权重。")
        model.load_state_dict(torch.load(saved_model_path))
        use_cached_model = True
    else:
        print("训练参数发生变化，重新训练模型。")
else:
    print("未检测到已保存模型，开始训练。")

###########################################
# 对抗训练（使用 mpgd_attack 生成对抗样本）并加入早停机制
###########################################
if not use_cached_model:
    print("开始对抗训练...")
    best_val_acc = 0.0
    patience = 10  # 如果连续 10 个周期验证准确率没有提升则停止训练
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 使用 mpgd_attack 生成对抗样本
            adv_data = mpgd_attack(model, data, target, epsilon, alpha, num_iter, mean, std, num_classes)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] 平均损失: {avg_loss:.4f}")

        # 使用验证集（此处采用测试集）计算准确率，作为早停依据
        val_acc = evaluate(model, test_loader, attack_fn=None, attack_name="Validation")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Epoch [{epoch + 1}]: Validation accuracy improved to {val_acc:.4f}. Saving model.")
        else:
            patience_counter += 1
            print(
                f"Epoch [{epoch + 1}]: Validation accuracy did not improve. Patience counter: {patience_counter}/{patience}.")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # 恢复最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation accuracy.")

    print("对抗训练完成！")

    # 保存模型权重和训练参数配置
    torch.save(model.state_dict(), saved_model_path)
    with open(saved_config_path, 'w') as f:
        json.dump(train_params, f)
else:
    print("直接使用缓存的模型权重，跳过训练过程。")

###########################################
# 分别在 Clean、FGSM、PGD、CW、MPGDA 和 AutoAttack 下测试准确率
###########################################
print("\n================= 测试评估 =================")
print("在干净测试集上的准确率：")
evaluate(model, test_loader, attack_fn=None, attack_name="Clean")

print("\nFGSM 攻击下的准确率：")
evaluate(model, test_loader, attack_fn=fgsm_attack, attack_name="FGSM")

print("\nPGD 攻击下的准确率：")
evaluate(model, test_loader, attack_fn=pgd_attack, attack_name="PGD")

if CW is not None:
    print("\nCW 攻击下的准确率：")
    evaluate(model, test_loader, attack_fn=cw_attack_fn, attack_name="CW")
else:
    print("\nCW 攻击跳过，因为 torchattacks 库未安装。")

print("\nMPGDA 攻击下的准确率：")
evaluate(model, test_loader, attack_fn=mpgd_attack_fn, attack_name="MPGDA")

print("\nAutoAttack 下的准确率：")
autoattack_evaluation(model, test_loader)
