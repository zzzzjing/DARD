import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchattacks

# 定义学生模型结构 ResNet18 for CIFAR-100
def get_student_model(num_classes=100, device=None):
    model = torchvision.models.resnet18(weights=None)
    # CIFAR-100 图像尺寸较小，修改第一层卷积并去掉 maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 修改全连接层输出为 num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model


# 定义对抗攻击方法

def fgsm_attack(model, images, labels, epsilon, mean, std):
    device = images.device
    # 计算归一化空间下图像的上下界
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)

    images = images.clone().detach()
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # 沿梯度符号方向更新
    adv_images = images + epsilon_tensor * images.grad.sign()
    # 保证在合法归一化区间内
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    return adv_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

    adv_images = images.clone().detach()

    # 随机初始化，保证在 epsilon 范围内
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1) * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)

    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        # 沿梯度符号方向更新
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 投影回 L∞ ball
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def tpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

    # 选择每个样本最不可能的类别作为目标
    with torch.no_grad():
        outputs = model(images)
        target_labels = outputs.argmin(dim=1)

    adv_images = images.clone().detach()

    # 随机初始化
    init_noise = torch.empty_like(adv_images).uniform_(-1, 1) * epsilon_tensor
    adv_images = adv_images + init_noise
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)

    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        # 对目标类别最小化损失（targeted attack）
        loss = F.cross_entropy(outputs, target_labels)
        model.zero_grad()
        loss.backward()
        # 目标攻击：梯度反向更新
        adv_images = adv_images - alpha_tensor * adv_images.grad.sign()
        # 投影回 L∞ ball
        perturbation = torch.clamp(adv_images - images, -epsilon_tensor, epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images.detach()

def i_fgsm_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    device = images.device
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

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
    device = images.device
    # 计算归一化空间下图像的合法范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)

    # 初始化扰动 δ
    delta = torch.zeros_like(images, requires_grad=True)

    for _ in range(num_iter):
        adv_images = images + delta
        outputs = model(adv_images)
        onehot = F.one_hot(labels, outputs.shape[1]).float()
        # 真实类别对应的 logit
        true_logit = (outputs * onehot).sum(dim=1)
        # 除去真实类别后取最大 logit
        wrong_logit = (outputs - onehot * 1e4).max(dim=1)[0]
        # 损失：当真实类别 logit 超过其他类别过多时（未满足攻击目标）损失 > 0
        loss = torch.clamp(true_logit - wrong_logit + kappa, min=0).mean()

        model.zero_grad()
        loss.backward()

        # 使用真实梯度对扰动更新（梯度下降，目标是最小化损失，使其为 0）
        delta.data = delta.data - lr * delta.grad.data
        # 投影到 L∞ 球内
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad.data.zero_()

    adv_images = images + delta
    # 保证扰动后的图像仍在合法归一化范围内
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    return adv_images.detach()


# AutoAttack 需要的模型包装器
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        # x 假设在 [0,1] 范围内，先归一化再传入模型
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))


# 评估函数
def evaluate(model, test_loader, attack_fn, **attack_kwargs):
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


# 主流程：加载模型 , 各种攻击评估
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-100 均值和标准差（常用参数）
    cifar100_mean = (0.5071, 0.4865, 0.4409)
    cifar100_std = (0.2673, 0.2564, 0.2761)

    # 攻击参数
    epsilon = 8.0 / 255.0
    alpha = 2.0 / 255.0
    num_iter = 20
    cw_lr = 1e-2
    cw_kappa = 0.1 # 可调参数

    # 加载测试集（使用 CIFAR-100）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 加载已蒸馏好的学生模型参数
    student_model = get_student_model(num_classes=100, device=device)
    ckpt_path = "TDARD_cifar100.pth"
    if not os.path.exists(ckpt_path):
        print(f"未找到模型权重文件 {ckpt_path}，请检查路径！")
        exit(0)
    student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    student_model.eval()
    print("成功加载学生模型参数，开始对抗评估 ...")

    # FGSM
    acc_fgsm = evaluate(student_model, test_loader, fgsm_attack,
                        epsilon=epsilon, mean=cifar100_mean, std=cifar100_std)
    print(f"[FGSM] 测试准确率: {acc_fgsm:.2f}%")

    # PGD
    acc_pgd = evaluate(student_model, test_loader, pgd_attack,
                       epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                       mean=cifar100_mean, std=cifar100_std)
    print(f"[PGD] 测试准确率: {acc_pgd:.2f}%")

    # TPGD
    acc_tpgd = evaluate(student_model, test_loader, tpgd_attack,
                        epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                        mean=cifar100_mean, std=cifar100_std)
    print(f"[TPGD] 测试准确率: {acc_tpgd:.2f}%")

    # I-FGSM
    acc_ifgsm = evaluate(student_model, test_loader, i_fgsm_attack,
                         epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                         mean=cifar100_mean, std=cifar100_std)
    print(f"[I-FGSM] 测试准确率: {acc_ifgsm:.2f}%")

    # CW
    acc_cw = evaluate(student_model, test_loader, cw_attack,
                      epsilon=epsilon, num_iter=num_iter, lr=cw_lr, kappa=cw_kappa,
                      mean=cifar100_mean, std=cifar100_std)
    print(f"[CW] 测试准确率: {acc_cw:.2f}%")

    # AutoAttack
    try:
        from autoattack import AutoAttack
        print("[AutoAttack] 开始评估 ...")
        # 收集测试集所有数据
        all_data = []
        all_labels = []
        for imgs, lbls in test_loader:
            all_data.append(imgs)
            all_labels.append(lbls)
        all_data = torch.cat(all_data, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)

        # 反归一化到 [0,1]
        mean_tensor = torch.tensor(cifar100_mean).view(1, 3, 1, 1).to(device)
        std_tensor = torch.tensor(cifar100_std).view(1, 3, 1, 1).to(device)
        data_unnorm = all_data * std_tensor + mean_tensor  # 变回 [0,1]

        # 包装模型：输入 [0,1] 后归一化送入模型
        wrapped_model = NormalizedModel(student_model, cifar100_mean, cifar100_std).to(device)
        wrapped_model.eval()

        adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard')
        adv_data = adversary.run_standard_evaluation(data_unnorm, all_labels, bs=128)

        # AutoAttack 返回的 adv_data 在 [0,1]，需归一化后送入 student_model 评估
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
