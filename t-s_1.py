import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torchattacks

# 预定义的辅助函数
def to_onehot(labels, num_classes):
    # 将标签转换为 one-hot 形式
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
    device = images.device
    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    # epsilon 与 alpha 转换到归一化空间（每个通道除以对应的 std）
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=device).view(1, 3, 1, 1)
    # 初始化对抗样本
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)  # 输出 shape 为 [B, num_classes]
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
            loss_correct = 0.0 * adv_images.sum()
        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask],
                                  onehot[~correct_mask],
                                  squared_pred=True)
        else:
            loss_wrong = 0.0 * adv_images.sum()

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


# CIFAR-10 专用的 ResNet 实现, 用于教师模型：ResNet56
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):

        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 全局平均池化（适应 CIFAR10 的特征图尺寸）
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 自定义 Dataset：返回样本索引, 用于预先计算教师输出
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index

# 知识蒸馏 Dataset：返回图像、标签和教师在 clean 与对抗样本上的 soft label（不融合）
class DistillationDataset(Dataset):
    def __init__(self, base_dataset, teacher_clean, teacher_adv):
        self.base_dataset = base_dataset
        self.teacher_clean = teacher_clean
        self.teacher_adv = teacher_adv

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        teacher_clean_label = self.teacher_clean[index]
        teacher_adv_label = self.teacher_adv[index]
        return img, target, teacher_clean_label, teacher_adv_label

# 以下部分为对抗评估相关函数
def evaluate_clean(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Clean Accuracy: {acc*100:.2f}%")
    return acc

def fgsm_attack_eval(model, images, labels, epsilon, cifar10_mean, cifar10_std):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + epsilon * images.grad.sign()
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(cifar10_mean, cifar10_std)],
                               device=images.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(cifar10_mean, cifar10_std)],
                               device=images.device).view(1, 3, 1, 1)
    adv_images = torch.clamp(adv_images, lower_bound, upper_bound)
    return adv_images

def pgd_attack_eval(model, images, labels, epsilon, alpha, num_iter, cifar10_mean, cifar10_std):
    adv_images = images.clone().detach()
    adv_images.requires_grad = True
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(cifar10_mean, cifar10_std)],
                               device=images.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(cifar10_mean, cifar10_std)],
                               device=images.device).view(1, 3, 1, 1)
    for i in range(num_iter):
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True
    return adv_images

def evaluate_attack(model, testloader, attack_fn, attack_params, device, attack_name):
    model.eval()
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **attack_params)
        outputs = model(adv_images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"{attack_name} Accuracy: {acc*100:.2f}%")
    return acc

def evaluate_autoattack(model, testloader, epsilon, device, batch_size):
    model.eval()
    correct = 0
    total = 0
    # 对于每个 mini-batch 单独生成对抗样本
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        # 为当前 batch 创建新的 AutoAttack 实例，确保不在一个大 batch 上运行
        adversary = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
        adv_images = adversary(images, labels)
        outputs = model(adv_images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"AutoAttack Accuracy: {acc*100:.2f}%")
    return acc


#############################
# 主函数：预处理、计算教师 soft label、对学生进行对抗知识蒸馏训练、以及对抗评估
#############################
def main():
    # ----------------------------
    # 超参数设置
    # ----------------------------
    num_classes = 10
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # PGD 攻击参数（单位为像素值，针对 [0,1] 范围的图像）
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10

    # CIFAR-10 归一化参数
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # 知识蒸馏相关参数
    distill_alpha = 0.7    # KL 损失权重
    temperature = 3.0      # 温度参数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 数据预处理
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # 用于计算教师输出的训练集（需要返回样本索引）
    trainset_indexed = IndexedCIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader_indexed = DataLoader(trainset_indexed, batch_size=batch_size, shuffle=False, num_workers=2)

    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----------------------------
    # 初始化教师模型（ResNet56 for CIFAR10）并加载预训练权重（如果有）
    # ----------------------------
    teacher_model = ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)
    teacher_ckpt = "teacher_resnet56.pth"
    if os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print("加载教师模型预训练参数")
    else:
        print("注意：未找到教师模型预训练参数，教师模型为随机初始化！\n"
              "建议先训练教师模型并保存权重，否则蒸馏效果可能较差。")
    teacher_model.eval()  # 固定教师模型

    # ----------------------------
    # 预先计算教师模型在训练集上对 clean 和对抗样本的 soft output
    # ----------------------------
    num_train = len(trainset_indexed)
    teacher_clean_outputs = torch.zeros(num_train, num_classes)
    teacher_adv_outputs = torch.zeros(num_train, num_classes)

    print("预计算教师模型的 soft label ...")
    for inputs, targets, indices in tqdm(trainloader_indexed, desc="教师模型推理"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.long()

        # 计算 clean 输出（不需要梯度）
        with torch.no_grad():
            outputs_clean = teacher_model(inputs)
            soft_clean = F.softmax(outputs_clean, dim=1)
            teacher_clean_outputs[indices] = soft_clean.cpu()

        # 对抗样本生成部分需要梯度，所以不要用 no_grad
        adv_inputs = mpgd_attack(teacher_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)
        with torch.no_grad():
            outputs_adv = teacher_model(adv_inputs)
            soft_adv = F.softmax(outputs_adv, dim=1)
            teacher_adv_outputs[indices] = soft_adv.cpu()

    # ----------------------------
    # 构建知识蒸馏专用的数据集（不融合教师 soft label，而是分别返回 clean 与 adversarial 的 soft label）
    # ----------------------------
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    distill_dataset = DistillationDataset(trainset, teacher_clean_outputs, teacher_adv_outputs)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ----------------------------
    # 初始化学生模型（ResNet18，改造适应 CIFAR10）
    # ----------------------------
    student_model = torchvision.models.resnet18(pretrained=False)
    # 修改第一层卷积（CIFAR10 图像尺寸较小）
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()  # 去掉 maxpool 层
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    # 使用 KLDivLoss 作为蒸馏损失（注意输入需要是对数概率）
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    # ----------------------------
    # 学生模型对抗训练：对每个 mini-batch 分别计算 clean 样本和对抗样本的输出，
    # 并分别使用教师模型在 clean 和对抗样本上的 soft label 进行知识蒸馏，同时结合真实标签的交叉熵损失
    # ----------------------------
    print("开始对学生模型进行对抗知识蒸馏训练 ...")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        # 修改数据加载器返回四项：img, target, teacher_clean, teacher_adv
        for inputs, targets, teacher_clean, teacher_adv in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher_clean = teacher_clean.to(device)  # 形状 [B, num_classes]
            teacher_adv = teacher_adv.to(device)      # 形状 [B, num_classes]

            # 生成学生模型的对抗样本（由 mpgd_attack 生成）
            adv_inputs = mpgd_attack(student_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)

            # 学生模型对 clean 样本和对抗样本的预测
            outputs_clean = student_model(inputs)
            outputs_adv = student_model(adv_inputs)

            # 1. 对 clean 样本，使用教师模型 clean soft label 进行蒸馏
            log_probs_clean = F.log_softmax(outputs_clean / temperature, dim=1)
            teacher_probs_clean = F.softmax(teacher_clean / temperature, dim=1)
            loss_KL_clean = kl_loss(log_probs_clean, teacher_probs_clean) * (temperature ** 2)
            loss_CE_clean = F.cross_entropy(outputs_clean, targets)
            loss_clean = distill_alpha * loss_KL_clean + (1.0 - distill_alpha) * loss_CE_clean

            # 2. 对 adversarial 样本，使用教师模型 adversarial soft label 进行蒸馏
            log_probs_adv = F.log_softmax(outputs_adv / temperature, dim=1)
            teacher_probs_adv = F.softmax(teacher_adv / temperature, dim=1)
            loss_KL_adv = kl_loss(log_probs_adv, teacher_probs_adv) * (temperature ** 2)
            loss_CE_adv = F.cross_entropy(outputs_adv, targets)
            loss_adv = distill_alpha * loss_KL_adv + (1.0 - distill_alpha) * loss_CE_adv

            # 总损失取两部分的平均
            loss = (loss_clean + loss_adv) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] 学生模型蒸馏训练 Loss: {avg_loss:.4f}")

        # 每个 epoch 后评估学生模型在测试集上的准确率（Clean）
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = student_model(inputs)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] 测试集 Clean Accuracy: {acc:.2f}%")

    # 保存训练好的学生模型
    torch.save(student_model.state_dict(), "student_resnet18_distilled.pth")
    print("学生模型训练完成，模型参数已保存。")

    # ----------------------------
    # 对抗攻击评估：Clean、FGSM、PGD、AutoAttack
    # ----------------------------
    print("开始对学生模型进行对抗攻击评估 ...")
    evaluate_clean(student_model, testloader, device)

    # FGSM 攻击评估
    fgsm_params = {'epsilon': epsilon, 'cifar10_mean': cifar10_mean, 'cifar10_std': cifar10_std}
    evaluate_attack(student_model, testloader, fgsm_attack_eval, fgsm_params, device, "FGSM")

    # PGD 攻击评估
    pgd_params = {'epsilon': epsilon, 'alpha': 2.0/255.0, 'num_iter': 20, 'cifar10_mean': cifar10_mean, 'cifar10_std': cifar10_std}
    evaluate_attack(student_model, testloader, pgd_attack_eval, pgd_params, device, "PGD")

    # AutoAttack 评估（需提前安装 autoattack 库）
    evaluate_autoattack(student_model, testloader, epsilon, device, batch_size)

if __name__ == '__main__':
    main()
