import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
from tqdm import tqdm
import os

######################################
# 1. 定义学生模型（ResNet18 for CIFAR10）
######################################
def get_student_model(num_classes=10, device=None):
    """
    构造与训练时相同的学生模型结构：基于 ResNet18，适应 CIFAR-10。
    """
    model = torchvision.models.resnet18(weights=None)
    # 针对 CIFAR-10 修改第一层卷积（图像尺寸较小）并去掉 maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

######################################
# 2. 定义评估函数（用于 torchattacks 的攻击）
######################################
def test_attack(model, attack, test_loader, device):
    """
    对 test_loader 中每个批次生成对抗样本，并计算模型在对抗样本上的准确率。
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc=attack.__class__.__name__):
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

######################################
# 3. 定义 TPGD 攻击包装器（目标为最不可能的类别）
######################################
class TPGD(torchattacks.PGD):
    def __init__(self, model, eps, alpha, steps):
        # 设置 targeted=True，随机初始化
        super().__init__(model, eps=eps, alpha=alpha, steps=steps, random_start=True, targeted=True)
    def forward(self, images, labels):
        # 对于每个样本，计算模型预测中最不可能的类别作为攻击目标
        with torch.no_grad():
            outputs = self.model(images)
        target_labels = outputs.argmin(dim=1)
        # 此时传入的 target_labels 会作为目标类别
        return super().forward(images, target_labels)

######################################
# 4. 主流程：加载模型、数据集 & 各种攻击评估
######################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 归一化参数（与训练时保持一致）
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # 加载 CIFAR-10 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # 加载已训练好的学生模型参数
    student_model = get_student_model(num_classes=10, device=device)
    ckpt_path = "student_resnet18_distilled.pth"
    if not os.path.exists(ckpt_path):
        print(f"未找到模型权重文件 {ckpt_path}，请检查路径！")
        exit(0)
    student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    student_model.eval()
    print("成功加载学生模型参数，开始对抗评估 ...")

    # 1) FGSM 攻击（torchattacks.FGSM）
    atk_fgsm = torchattacks.FGSM(student_model, eps=8/255)
    acc_fgsm = test_attack(student_model, atk_fgsm, test_loader, device)
    print(f"[FGSM] 测试准确率: {acc_fgsm:.2f}%")

    # 2) PGD 攻击（随机初始化，等价于常规 PGD）
    atk_pgd = torchattacks.PGD(student_model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    acc_pgd = test_attack(student_model, atk_pgd, test_loader, device)
    print(f"[PGD] 测试准确率: {acc_pgd:.2f}%")

    # 3) I-FGSM 攻击（PGD 不使用随机初始化，即 iterative FGSM）
    atk_ifgsm = torchattacks.PGD(student_model, eps=8/255, alpha=2/255, steps=10, random_start=False)
    acc_ifgsm = test_attack(student_model, atk_ifgsm, test_loader, device)
    print(f"[I-FGSM] 测试准确率: {acc_ifgsm:.2f}%")

    # 4) TPGD 攻击（目标为最不可能类别，利用自定义 TPGD 包装器）
    atk_tpgd = TPGD(student_model, eps=8/255, alpha=2/255, steps=10)
    acc_tpgd = test_attack(student_model, atk_tpgd, test_loader, device)
    print(f"[TPGD] 测试准确率: {acc_tpgd:.2f}%")

    # 5) CW 攻击（torchattacks.CW，超参数可根据需要调整）
    atk_cw = torchattacks.CW(student_model, c=1, kappa=0, steps=10, lr=0.01)
    acc_cw = test_attack(student_model, atk_cw, test_loader, device)
    print(f"[CW] 测试准确率: {acc_cw:.2f}%")

    # 6) AutoAttack 评估（AutoAttack 要求输入图像在 [0,1] 范围，因此需要反归一化）
    try:
        from autoattack import AutoAttack
        print("[AutoAttack] 开始评估 ...")
        # 收集测试集所有数据
        all_images = []
        all_labels = []
        for imgs, lbls in test_loader:
            all_images.append(imgs)
            all_labels.append(lbls)
        all_images = torch.cat(all_images, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)
        # 反归一化到 [0,1]
        mean_tensor = torch.tensor(cifar10_mean).view(1,3,1,1).to(device)
        std_tensor = torch.tensor(cifar10_std).view(1,3,1,1).to(device)
        images_denorm = all_images * std_tensor + mean_tensor

        # 定义包装器模型：接收 [0,1] 图像，再归一化后送入 student_model
        class NormalizedModel(nn.Module):
            def __init__(self, model, mean, std):
                super().__init__()
                self.model = model
                self.mean = torch.tensor(mean).view(1,3,1,1).to(device)
                self.std = torch.tensor(std).view(1,3,1,1).to(device)
            def forward(self, x):
                x_norm = (x - self.mean) / self.std
                return self.model(x_norm)
        wrapped_model = NormalizedModel(student_model, cifar10_mean, cifar10_std).to(device)
        wrapped_model.eval()

        adversary = AutoAttack(wrapped_model, norm='Linf', eps=8/255, version='standard')
        adv_images = adversary.run_standard_evaluation(images_denorm, all_labels, bs=128)
        # 归一化后送入 student_model 评估
        adv_images_norm = (adv_images - mean_tensor) / std_tensor
        with torch.no_grad():
            outputs = student_model(adv_images_norm)
            preds = outputs.argmax(dim=1)
        acc_auto = 100.0 * (preds == all_labels).sum().item() / len(all_labels)
        print(f"[AutoAttack] 测试准确率: {acc_auto:.2f}%")
    except ImportError:
        print("未安装 AutoAttack 库，跳过 AutoAttack 评估。")
    except Exception as e:
        print("AutoAttack 评估时出错:", e)
