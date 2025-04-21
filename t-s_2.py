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

# ----------------------------
# 一些辅助函数
# ----------------------------
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

# ----------------------------
# MPGD 对抗样本生成函数
# ----------------------------
def mpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes):
    """
    使用混合PGD方法 (MPGD) 生成对抗样本。
    其中对每次迭代的损失函数会在正确分类和错误分类之间做一个动态权重。
    """
    device = images.device
    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    # epsilon 与 alpha 转换到归一化空间
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=device).view(1, 3, 1, 1)
    # 初始化对抗样本
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)
        pred_labels = torch.argmax(outputs, dim=1)
        correct_mask = (labels == pred_labels)
        onehot = to_onehot(labels, num_classes)
        adv_pred_softmax = F.softmax(outputs, dim=1)

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

        # 这里的 lam 与图中所示非常类似，只是下标略有差异
        # lam = i / (2.0 * num_iter)  # 原代码
        # 若想与图中公式完全对齐，可改成：
        w_w = (i) / (2.0 * num_iter)    # i 从 0 ~ num_iter-1
        w_c = 1.0 - w_w

        loss = w_c * loss_correct + w_w * loss_wrong

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 投影到 epsilon-ball 内
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images


# ----------------------------
# CIFAR-10 专用 ResNet56 (教师)
# ----------------------------
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
        out = F.avg_pool2d(out, out.size()[3])  # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ----------------------------
# 自定义 Dataset：IndexedCIFAR10
# ----------------------------
class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index

# ----------------------------
# 知识蒸馏专用数据集
# 返回：图像、真实标签、教师在 clean 样本上的 soft label、教师在 adversarial 样本上的 soft label
# ----------------------------
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


# ----------------------------
# 对抗评估相关
# ----------------------------
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
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        adversary = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
        adv_images = adversary(images, labels)
        outputs = model(adv_images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"AutoAttack Accuracy: {acc*100:.2f}%")
    return acc


# ----------------------------
# 主函数
# ----------------------------
def main():
    # ----------------------------
    # 一些超参数
    # ----------------------------
    num_classes = 10
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # MPGD 攻击参数
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10

    # CIFAR-10 归一化
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # 知识蒸馏相关
    distill_alpha = 0.7
    temperature = 3.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 数据加载
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

    trainset_indexed = IndexedCIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader_indexed = DataLoader(trainset_indexed, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----------------------------
    # 初始化教师模型
    # ----------------------------
    teacher_model = ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)
    teacher_ckpt = "teacher_resnet56.pth"
    if os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print("加载教师模型预训练参数")
    else:
        print("警告：未找到教师模型预训练参数，教师模型为随机初始化，蒸馏效果可能较差。")
    teacher_model.eval()

    # ----------------------------
    # 预先计算教师模型对训练集中 clean/adv 样本的 soft label
    # ----------------------------
    num_train = len(trainset_indexed)
    teacher_clean_outputs = torch.zeros(num_train, num_classes)
    teacher_adv_outputs = torch.zeros(num_train, num_classes)

    print("预计算教师模型的 soft label ...")
    for inputs, targets, indices in tqdm(trainloader_indexed, desc="教师模型推理"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.long()

        with torch.no_grad():
            out_clean = teacher_model(inputs)
            soft_clean = F.softmax(out_clean, dim=1)
            teacher_clean_outputs[indices] = soft_clean.cpu()

        # 生成对抗样本 (对教师模型本身)
        adv_inputs = mpgd_attack(teacher_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)
        with torch.no_grad():
            out_adv = teacher_model(adv_inputs)
            soft_adv = F.softmax(out_adv, dim=1)
            teacher_adv_outputs[indices] = soft_adv.cpu()

    # ----------------------------
    # 构建知识蒸馏数据集
    # ----------------------------
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    distill_dataset = DistillationDataset(trainset, teacher_clean_outputs, teacher_adv_outputs)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ----------------------------
    # 初始化学生模型 (ResNet18 for CIFAR10)
    # ----------------------------
    student_model = torchvision.models.resnet18(weights=None)
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    kl_loss = nn.KLDivLoss(reduction='batchmean')

    # ----------------------------
    # 训练：采用图中给出的动态加权方式
    # L_total = w_c * L_correct + w_w * L_wrong
    # 其中 w_w = (t-1)/(2T), w_c = 1 - w_w
    # ----------------------------
    print("开始对学生模型进行对抗知识蒸馏训练 ...")
    T = num_epochs  # 总 epoch 数
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0

        # 计算当前 epoch 的动态权重
        t = epoch + 1  # 当前 epoch（1-based）
        w_w = (t - 1) / (2.0 * T)  # 与图中 (t-1)/(2T) 相对应
        w_c = 1.0 - w_w

        for inputs, targets, teacher_clean, teacher_adv in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            teacher_clean = teacher_clean.to(device)
            teacher_adv = teacher_adv.to(device)

            # 用学生模型自身生成对抗样本
            adv_inputs = mpgd_attack(student_model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std, num_classes)

            # 学生模型对 clean & adv 样本的输出
            outputs_clean = student_model(inputs)
            outputs_adv = student_model(adv_inputs)

            # ---- 计算对 clean 样本的蒸馏损失 + CE ----
            log_probs_clean = F.log_softmax(outputs_clean / temperature, dim=1)
            teacher_probs_clean = F.softmax(teacher_clean / temperature, dim=1)
            loss_KL_clean = kl_loss(log_probs_clean, teacher_probs_clean) * (temperature ** 2)
            loss_CE_clean = F.cross_entropy(outputs_clean, targets)
            L_correct = distill_alpha * loss_KL_clean + (1.0 - distill_alpha) * loss_CE_clean

            # ---- 计算对 adv 样本的蒸馏损失 + CE ----
            log_probs_adv = F.log_softmax(outputs_adv / temperature, dim=1)
            teacher_probs_adv = F.softmax(teacher_adv / temperature, dim=1)
            loss_KL_adv = kl_loss(log_probs_adv, teacher_probs_adv) * (temperature ** 2)
            loss_CE_adv = F.cross_entropy(outputs_adv, targets)
            L_wrong = distill_alpha * loss_KL_adv + (1.0 - distill_alpha) * loss_CE_adv

            # ---- 动态加权融合 ----
            loss = w_c * L_correct + w_w * L_wrong

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | w_w={w_w:.3f}, w_c={w_c:.3f}")

        # 每个 epoch 评估一下 clean 测试准确率
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
        print(f"Epoch [{epoch+1}/{num_epochs}] | Clean Accuracy: {acc:.2f}%")

    # ----------------------------
    # 保存学生模型
    # ----------------------------
    torch.save(student_model.state_dict(), "student_resnet18_distilled.pth")
    print("学生模型训练完成，模型参数已保存。")

    # ----------------------------
    # 对抗评估
    # ----------------------------
    print("开始对学生模型进行对抗攻击评估 ...")
    evaluate_clean(student_model, testloader, device)

    # FGSM
    fgsm_params = {'epsilon': epsilon, 'cifar10_mean': cifar10_mean, 'cifar10_std': cifar10_std}
    evaluate_attack(student_model, testloader, fgsm_attack_eval, fgsm_params, device, "FGSM")

    # PGD
    pgd_params = {'epsilon': epsilon, 'alpha': 2.0/255.0, 'num_iter': 20, 'cifar10_mean': cifar10_mean, 'cifar10_std': cifar10_std}
    evaluate_attack(student_model, testloader, pgd_attack_eval, pgd_params, device, "PGD")

    # AutoAttack
    evaluate_autoattack(student_model, testloader, epsilon, device, batch_size)


if __name__ == '__main__':
    main()
