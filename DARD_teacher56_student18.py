import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

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
    """
    DiceLoss 用于计算对抗优化中的损失（这里用于分类任务）。
    input: 模型输出经过 softmax 后的概率分布，形状 [B, num_classes]
    target: onehot 形式的标签，形状 [B, num_classes]
    """
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

# 用于教师模型：ResNet56
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

# 自定义 Dataset：返回样本索引（用于预先计算教师输出）
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index

# 知识蒸馏 Dataset：返回图像、真实标签以及教师在 clean 和对抗样本下的 soft label
class DistillationDataset(Dataset):
    def __init__(self, base_dataset, teacher_clean, teacher_adv):
        """
        :param base_dataset: 原始 CIFAR10 数据集（不包含索引信息）
        :param teacher_clean: 大小为 [N, num_classes] 的 tensor，存储每个样本在 clean 输入下的教师 soft output
        :param teacher_adv: 大小为 [N, num_classes] 的 tensor，存储每个样本在教师对抗样本下的 soft output
        """
        self.base_dataset = base_dataset
        self.teacher_clean = teacher_clean
        self.teacher_adv = teacher_adv

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        teacher_soft_clean = self.teacher_clean[index]
        teacher_soft_adv = self.teacher_adv[index]
        return img, target, teacher_soft_clean, teacher_soft_adv

# 主函数：预处理、计算教师 soft label、对学生进行对抗知识蒸馏训练
def main():
    # 超参数设置
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

    # 数据预处理
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

    # 从 CIFAR10 测试集划分出验证集和测试集
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_size = int(0.5 * len(full_testset))
    test_size = len(full_testset) - val_size
    val_set, test_set = random_split(full_testset, [val_size, test_size])
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化教师模型（ResNet56 for CIFAR10）并加载预训练权重（如果有）
    teacher_model = ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)
    teacher_ckpt = "teacher_resnet56.pth"
    if os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print("加载教师模型预训练参数")
    else:
        print("注意：未找到教师模型预训练参数，教师模型为随机初始化！\n"
              "建议先训练教师模型并保存权重，否则蒸馏效果可能较差。")
    teacher_model.eval()  # 固定教师模型

    # 预先计算教师模型在训练集上对 clean 和对抗样本的 soft output
    num_train = len(trainset_indexed)
    teacher_clean_outputs = torch.zeros(num_train, num_classes)
    teacher_adv_outputs = torch.zeros(num_train, num_classes)

    print("预计算教师模型的 soft label ...")
    for inputs, targets, indices in tqdm(trainloader_indexed, desc="教师模型推理"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.long()

        # 计算 clean 输出，使用 no_grad
        with torch.no_grad():
            outputs_clean = teacher_model(inputs)
            soft_clean = F.softmax(outputs_clean, dim=1)
            teacher_clean_outputs[indices] = soft_clean.cpu()

        # 对抗样本生成部分需要梯度
        adv_inputs = mpgd_attack(teacher_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)
        with torch.no_grad():
            outputs_adv = teacher_model(adv_inputs)
            soft_adv = F.softmax(outputs_adv, dim=1)
            teacher_adv_outputs[indices] = soft_adv.cpu()

    # 构建知识蒸馏专用的数据集（分别返回教师 clean 与对抗 soft label）
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    distill_dataset = DistillationDataset(trainset, teacher_clean_outputs, teacher_adv_outputs)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 初始化学生模型，使用 ResNet18 适应 CIFAR10
    student_model = torchvision.models.resnet18(weights=None)
    # 修改第一层卷积（CIFAR10 图像尺寸较小）
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()  # 去掉 maxpool 层
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    # KLDivLoss 用于知识蒸馏（注意输入需要是对数概率）
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    print("开始对学生模型进行对抗知识蒸馏训练 ...")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets, teacher_soft_clean, teacher_soft_adv in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher_soft_clean = teacher_soft_clean.to(device)  # 形状 [B, num_classes]
            teacher_soft_adv = teacher_soft_adv.to(device)      # 形状 [B, num_classes]

            # 清洁样本分支
            outputs_clean = student_model(inputs)
            log_probs_clean = F.log_softmax(outputs_clean / temperature, dim=1)
            teacher_probs_clean = F.softmax(teacher_soft_clean / temperature, dim=1)
            loss_KL_clean = kl_loss(log_probs_clean, teacher_probs_clean) * (temperature ** 2)
            loss_CE_clean = F.cross_entropy(outputs_clean, targets)
            loss_clean = distill_alpha * loss_KL_clean + (1.0 - distill_alpha) * loss_CE_clean

            # 对抗样本分支（利用学生模型生成对抗样本）
            adv_inputs = mpgd_attack(student_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)
            outputs_adv = student_model(adv_inputs)
            log_probs_adv = F.log_softmax(outputs_adv / temperature, dim=1)
            teacher_probs_adv = F.softmax(teacher_soft_adv / temperature, dim=1)
            loss_KL_adv = kl_loss(log_probs_adv, teacher_probs_adv) * (temperature ** 2)
            loss_CE_adv = F.cross_entropy(outputs_adv, targets)
            loss_adv = distill_alpha * loss_KL_adv + (1.0 - distill_alpha) * loss_CE_adv

            # 最终总损失取 clean 与对抗两个分支的平均
            loss = 0.5 * (loss_clean + loss_adv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] 训练 Loss: {avg_loss:.4f}")

        # 在验证集上评估模型的损失和准确率
        student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = student_model(inputs)
                loss_val = F.cross_entropy(outputs, targets)
                val_loss += loss_val.item() * inputs.size(0)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        avg_val_loss = val_loss / total
        val_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] 验证集 Loss: {avg_val_loss:.4f}, 准确率: {val_acc:.2f}%")

    # 训练结束后保存模型的权重参数
    torch.save(student_model.state_dict(), "DARD_student_resnet18_distilled.pth")
    print("模型权重已保存至 DARD_student_resnet18_distilled.pth")

    # 使用训练后的模型在测试集上评估性能
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = student_model(inputs)
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    test_acc = 100.0 * correct / total
    print(f"最终测试集准确率: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
