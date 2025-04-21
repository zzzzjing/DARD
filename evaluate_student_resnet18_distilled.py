import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

#############################
# 1. 定义学生模型结构（ResNet18 for CIFAR10）
#############################
def get_student_model(num_classes=10, device=None):
    """
    构造与训练时相同的学生模型结构：基于 ResNet18，适应 CIFAR-10。
    """
    model = torchvision.models.resnet18(weights=None)
    # CIFAR10 图像尺寸较小，修改第一层卷积并去掉 maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 修改全连接层输出为 num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

#############################
# 2. 定义对抗攻击方法
#############################

def fgsm_attack(model, images, labels, epsilon, mean, std):
    """
    FGSM 攻击：单步
    """
    device = images.device
    # 计算图像在归一化空间下的上下限
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    # 将 epsilon 转为归一化空间
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)

    images = images.clone().detach()
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # 沿着梯度符号方向迈一步
    adv_images = images + epsilon_tensor * images.grad.sign()
    # 约束到 [0,1] 归一化后的区间
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    return adv_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    """
    PGD 攻击：多步迭代 + 随机初始化
    """
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    # 复制原图
    adv_images = images.clone().detach()

    # ---- 修正：使用广播乘法进行随机初始化 ----
    # 先生成 [-1,1] 的随机噪声，再乘以 epsilon_tensor
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1)
    init_noise = init_noise * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    # ------------------------------------------

    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        # 沿着梯度符号方向更新
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 投影回 L∞ ball
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def tpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    """
    TPGD 攻击：目标设为模型最不可能预测的类别 (targeted attack)。
    同样使用随机初始化 + 多步迭代。
    """
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    # 找出每个样本最不可能的类别
    with torch.no_grad():
        outputs = model(images)
        target_labels = outputs.argmin(dim=1)

    adv_images = images.clone().detach()

    # ---- 修正：使用广播乘法进行随机初始化 ----
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1)
    init_noise = init_noise * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    # ------------------------------------------

    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        # 对目标类别做最小化损失 (targeted 攻击)
        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        # targeted 攻击，故反方向更新
        adv_images = adv_images - alpha_tensor * adv_images.grad.sign()
        # 投影回 L∞ ball
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def i_fgsm_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    """
    I-FGSM 攻击：与 PGD 类似，但不使用随机初始化。
    """
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1,3,1,1)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def cw_attack(model, images, labels, epsilon, num_iter, lr, kappa, mean, std):
    """
    简化版 CW 攻击：用 margin-based loss 并做 L∞ 范数约束。
    """
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1,3,1,1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1,3,1,1)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        # 真实标签对应的 logit
        onehot = F.one_hot(labels, outputs.shape[1]).float()
        correct_logit = (outputs * onehot).sum(dim=1)
        # 排除真实标签后的最大 logit
        inf_mask = torch.ones_like(outputs) * 1e4
        wrong_logits = outputs - onehot * inf_mask
        max_wrong_logit, _ = wrong_logits.max(dim=1)

        # CW 的 margin
        f = torch.clamp(max_wrong_logit - correct_logit, min=-kappa)
        loss = torch.mean(f)
        model.zero_grad()
        loss.backward()

        # 这里采用梯度下降来增大 margin（untargeted）
        adv_images = adv_images - lr * adv_images.grad
        # 投影回 L∞ ball
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

#############################
# AutoAttack 需要的模型包装器
#############################
class NormalizedModel(nn.Module):
    """
    将输入 [0,1] 范围图像进行归一化后再送入模型
    """
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1,3,1,1)
        self.std = torch.tensor(std).view(1,3,1,1)

    def forward(self, x):
        # x 假设已经在 [0,1]，这里进行 (x - mean)/std
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))

#############################
# 评估函数
#############################
def evaluate(model, test_loader, attack_fn, **attack_kwargs):
    """
    使用指定 attack_fn 对 test_loader 中的所有图像进行攻击，然后评估模型准确率。
    """
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in tqdm(test_loader, desc=attack_fn.__name__):
        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs = attack_fn(model, inputs, labels, **attack_kwargs)
        with torch.no_grad():
            outputs = model(adv_inputs)
            pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

#############################
# 4. 主流程：加载模型 -> 各种攻击评估
#############################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 均值和标准差
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # 攻击参数
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10
    cw_lr = 1e-2
    cw_kappa = 0.0  # 可调参数

    # 加载测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 加载已蒸馏好的学生模型参数
    student_model = get_student_model(num_classes=10, device=device)
    ckpt_path = "student_resnet18_distilled.pth"
    if not os.path.exists(ckpt_path):
        print(f"未找到模型权重文件 {ckpt_path}，请检查路径！")
        exit(0)
    student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    student_model.eval()
    print("成功加载学生模型参数，开始对抗评估 ...")

    # 1) FGSM
    acc_fgsm = evaluate(student_model, test_loader, fgsm_attack,
                        epsilon=epsilon, mean=cifar10_mean, std=cifar10_std)
    print(f"[FGSM] 测试准确率: {acc_fgsm:.2f}%")

    # 2) PGD
    acc_pgd = evaluate(student_model, test_loader, pgd_attack,
                       epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                       mean=cifar10_mean, std=cifar10_std)
    print(f"[PGD] 测试准确率: {acc_pgd:.2f}%")

    # 3) TPGD
    acc_tpgd = evaluate(student_model, test_loader, tpgd_attack,
                        epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                        mean=cifar10_mean, std=cifar10_std)
    print(f"[TPGD] 测试准确率: {acc_tpgd:.2f}%")

    # 4) I-FGSM
    acc_ifgsm = evaluate(student_model, test_loader, i_fgsm_attack,
                         epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                         mean=cifar10_mean, std=cifar10_std)
    print(f"[I-FGSM] 测试准确率: {acc_ifgsm:.2f}%")

    # 5) CW
    acc_cw = evaluate(student_model, test_loader, cw_attack,
                      epsilon=epsilon, num_iter=num_iter, lr=cw_lr, kappa=cw_kappa,
                      mean=cifar10_mean, std=cifar10_std)
    print(f"[CW] 测试准确率: {acc_cw:.2f}%")

    # 6) AutoAttack
    try:
        from autoattack import AutoAttack
        print("[AutoAttack] 开始评估 ...")
        # AutoAttack 需要输入图像在 [0,1] 范围，因此需反归一化
        all_data = []
        all_labels = []
        for imgs, lbls in test_loader:
            all_data.append(imgs)
            all_labels.append(lbls)
        all_data = torch.cat(all_data, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)

        # 反归一化到 [0,1]
        mean_tensor = torch.tensor(cifar10_mean).view(1,3,1,1).to(device)
        std_tensor = torch.tensor(cifar10_std).view(1,3,1,1).to(device)
        data_unnorm = all_data * std_tensor + mean_tensor  # 变回 [0,1]

        # 包装模型
        wrapped_model = NormalizedModel(student_model, cifar10_mean, cifar10_std).to(device)
        wrapped_model.eval()

        # 运行 AutoAttack
        adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard')
        adv_data = adversary.run_standard_evaluation(data_unnorm, all_labels, bs=128)

        # 评估：AutoAttack 返回的 adv_data 已在 [0,1]，需再归一化
        adv_data_norm = (adv_data - mean_tensor) / std_tensor
        with torch.no_grad():
            outputs = student_model(adv_data_norm)
            preds = outputs.argmax(dim=1)
        correct = (preds == all_labels).sum().item()
        acc_auto = 100.0 * correct / len(all_labels)
        print(f"[AutoAttack] 测试准确率: {acc_auto:.2f}%")
    except ImportError:
        print("未安装 autoattack 库，跳过 AutoAttack 评估。")
    except Exception as e:
        print("AutoAttack 评估时出错:", e)
