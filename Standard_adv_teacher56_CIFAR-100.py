import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

##############################
# 辅助函数：to_onehot 和 DiceLoss
##############################
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
    # 计算 Dice 损失
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

#######################################
# mpgd 对抗攻击方法的实现
#######################################
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
        lam = (i - 1 )/ (2.0 * num_iter)
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

#######################################
# ResNet56 的实现（适用于 CIFAR 数据集）
#######################################
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
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet56(num_classes=100):
    # CIFAR ResNet 的层数为 6n+2，此处 n=9 对应 ResNet56
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

#######################################
# 数据加载和对抗训练
#######################################
# CIFAR-100 的均值和标准差
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

# 数据预处理
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载 CIFAR-100 训练集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet56(num_classes=100).to(device)

# 定义优化器和损失函数（这里训练时仍然使用交叉熵损失）
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# 对抗攻击的超参数（以图像空间为单位）
epsilon = 8 / 255.0  # 最大扰动
alpha = 2 / 255.0    # 步长
num_iter = 10        # 攻击迭代次数

num_epochs = 100  # 训练轮数，可根据需要调整

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 生成对抗样本（仅使用对抗样本进行训练）
        adv_images = mpgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std, num_classes=100)

        # 前向传播：使用对抗样本训练
        outputs = model(adv_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] 平均损失: {avg_loss:.4f}")

# 训练结束后保存模型权重
torch.save(model.state_dict(), 'teacher56_adv_CIFAR-100.pth')
print("模型权重已保存为 teacher56_adv_CIFAR-100.pth")
