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




# 定义辅助函数和类
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
    device = images.device
    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)], device=device).view(1, 3, 1, 1)
    epsilon_tensor = torch.tensor([epsilon / s for s in std], device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std], device=device).view(1, 3, 1, 1)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)
        pred_labels = torch.argmax(outputs, dim=1)
        correct_mask = (labels == pred_labels)
        onehot = to_onehot(labels, num_classes)
        adv_pred_softmax = F.softmax(outputs, dim=1)

        if correct_mask.sum() > 0:
            loss_correct = DiceLoss(adv_pred_softmax[correct_mask], onehot[correct_mask], squared_pred=True)
        else:
            loss_correct = 0.0 * adv_images.sum()
        if (~correct_mask).sum() > 0:
            loss_wrong = DiceLoss(adv_pred_softmax[~correct_mask], onehot[~correct_mask], squared_pred=True)
        else:
            loss_wrong = 0.0 * adv_images.sum()

        lam = i / (2.0 * num_iter)
        loss = (1 - lam) * loss_correct + lam * loss_wrong

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images


# 定义 ResNet 模型 教师模型: ResNet56

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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



# IndexedCIFAR10 & DistillationDataset


class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index
class DistillationDataset(Dataset):
    def __init__(self, base_dataset, teacher_clean, teacher_adv):
        self.base_dataset = base_dataset
        self.teacher_clean = teacher_clean
        self.teacher_adv = teacher_adv

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        teacher_target = (self.teacher_clean[index] + self.teacher_adv[index]) / 2.0
        return img, target, teacher_target



#  对抗评估相关函数
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



# CW 攻击损失
class CWLoss(nn.Module):
    def __init__(self, kappa=0):
        super(CWLoss, self).__init__()
        self.kappa = kappa

    def forward(self, logits, target):
        target_onehot = F.one_hot(target, num_classes=logits.shape[1]).float()
        correct_logits = torch.sum(target_onehot * logits, dim=1)
        max_wrong_logits = torch.max((1 - target_onehot) * logits - target_onehot * 10000, dim=1)[0]
        loss = torch.clamp(correct_logits - max_wrong_logits + self.kappa, min=0).mean()
        return loss


# 改进的 TRADES 损失
def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=10,
                beta=6.0):


    #  构造初始对抗样本 x_adv
    #     在 -epsilon ~ +epsilon 范围内对 x_natural 做随机扰动
    x_adv = x_natural.detach() + torch.zeros_like(x_natural).uniform_(-epsilon, epsilon)
    # 不要一次性设置 x_adv.requires_grad = True，否则下一轮迭代 detach 后就失效

    #  进入生成对抗样本的迭代循环
    model.eval()  # TRADES 生成对抗样本时让 BN 固定在 eval
    for _ in range(perturb_steps):
        # 保证 x_adv 的梯度打开
        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)

        # 前向 & KL
        #   x_natural 不需要梯度, x_adv 需要
        outputs_adv = model(x_adv)
        outputs_nat = model(x_natural).detach()  #  detach, 只要对 x_adv 求梯度

        loss_kl = F.kl_div(
            F.log_softmax(outputs_adv, dim=1),
            F.softmax(outputs_nat, dim=1),
            reduction='batchmean'
        )

        # 反向
        model.zero_grad()
        grad = torch.autograd.grad(loss_kl, x_adv)[0]

        # 更新 x_adv
        x_adv = x_adv + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    #  TRADES 最终损失
    model.train()
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)  #  x_adv 可以不再 requires_grad
    loss_ce = F.cross_entropy(logits_nat, y)
    loss_kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat, dim=1),
        reduction='batchmean'
    )
    loss = loss_ce + beta * loss_kl
    return loss


# 主训练流程 main()
def main():
    #超参数设置
    num_classes = 10
    batch_size = 128
    num_epochs = 30
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # PGD 攻击参数
    epsilon = 8.0/255.0
    alpha = 2.0/255.0
    num_iter = 10

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    distill_alpha = 0.7
    temperature = 3.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
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

    # 用于计算教师输出的训练集
    trainset_indexed = IndexedCIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader_indexed = DataLoader(trainset_indexed, batch_size=batch_size, shuffle=False, num_workers=2)

    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化教师模型
    teacher_model = ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes).to(device)
    teacher_ckpt = "teacher_resnet56.pth"
    if os.path.exists(teacher_ckpt):
        teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print("加载教师模型预训练参数")
    else:
        print("未找到教师模型预训练参数，教师模型为随机初始化！")
    teacher_model.eval()

    # 预先计算教师模型在训练集上的 soft label
    num_train = len(trainset_indexed)
    teacher_clean_outputs = torch.zeros(num_train, num_classes)
    teacher_adv_outputs = torch.zeros(num_train, num_classes)

    print("预计算教师模型的 soft label ...")
    for inputs, targets, indices in tqdm(trainloader_indexed, desc="教师模型推理"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.long()

        with torch.no_grad():
            outputs_clean = teacher_model(inputs)
            soft_clean = F.softmax(outputs_clean, dim=1)
            teacher_clean_outputs[indices] = soft_clean.cpu()

        adv_inputs = mpgd_attack(teacher_model, inputs, targets, epsilon, alpha, num_iter,
                                 cifar10_mean, cifar10_std, num_classes)
        with torch.no_grad():
            outputs_adv = teacher_model(adv_inputs)
            soft_adv = F.softmax(outputs_adv, dim=1)
            teacher_adv_outputs[indices] = soft_adv.cpu()


    # 构建知识蒸馏专用的数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    distill_dataset = DistillationDataset(trainset, teacher_clean_outputs, teacher_adv_outputs)
    distill_loader = DataLoader(distill_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    # 初始化学生模型 (ResNet18)
    student_model = torchvision.models.resnet18(weights=None)
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # KLDivLoss 用于蒸馏
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    # 训练学生模型 (结合 TRADES + CW + AutoAttack)
    print("开始对学生模型进行对抗知识蒸馏 + TRADES 训练 ...")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets, teacher_soft in tqdm(distill_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets, teacher_soft = inputs.to(device), targets.to(device), teacher_soft.to(device)
            optimizer.zero_grad()

            # 生成 AutoAttack 对抗样本
            adversary = torchattacks.AutoAttack(student_model, norm='Linf', eps=epsilon)
            adv_inputs_aa = adversary(inputs, targets)

            #  生成 CW-L2 对抗样本
            cw_attack = torchattacks.CW(student_model, c=1, kappa=0, steps=10, lr=0.01)
            adv_inputs_cw = cw_attack(inputs, targets)

            #  TRADES 损失 以 AutoAttack 的对抗样本为输入
            loss_trades = trades_loss(model=student_model,
                                      x_natural=inputs,
                                      y=targets,
                                      optimizer=optimizer,
                                      step_size=2/255,
                                      epsilon=8/255,
                                      perturb_steps=10,
                                      beta=6.0)

            # CW 损失
            logits_cw = student_model(adv_inputs_cw)
            loss_cw = 0.5 * CWLoss()(logits_cw, targets)

            # 知识蒸馏损失 教师提供的 soft label 的 KL 散度

            outputs_clean = student_model(inputs)
            log_probs_stu = F.log_softmax(outputs_clean / temperature, dim=1)
            teacher_probs = F.softmax(teacher_soft / temperature, dim=1)
            loss_kd = kl_loss(log_probs_stu, teacher_probs) * (temperature ** 2)

            #总损失融合
            loss = loss_trades + loss_cw + distill_alpha * loss_kd
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(distill_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 每个 epoch 结束后简单评估
        evaluate_clean(student_model, testloader, device)
        evaluate_attack(student_model, testloader, pgd_attack_eval,
                        {'epsilon': epsilon, 'alpha': alpha, 'num_iter': 20,
                         'cifar10_mean': cifar10_mean, 'cifar10_std': cifar10_std},
                        device, "PGD")
        evaluate_autoattack(student_model, testloader, epsilon, device, batch_size)


    # 保存模型
    torch.save(student_model.state_dict(), "student_resnet18_distilled.pth")
    print("训练完成，模型已保存: student_resnet18_distilled.pth")


if __name__ == '__main__':
    main()
