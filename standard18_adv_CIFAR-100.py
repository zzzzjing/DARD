import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 超参数设置
num_epochs = 50
batch_size = 128
learning_rate = 0.1  # 提高初始学习率

# 对抗攻击相关参数（常见设置）
epsilon = 8 / 255  # 最大扰动
alpha = 2 / 255  # 每步扰动
num_iter = 10  # 攻击迭代次数

# 数据增强和归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# 加载 CIFAR-100 数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 定义 ResNet18 模型，并修改第一层和池化层适应 CIFAR-100
model = resnet18(weights=None)
# 修改第一层卷积：使用 3x3 核，步长 1，填充 1
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 取消最大池化层
model.maxpool = nn.Identity()
# 修改全连接层输出类别数为 100
model.fc = nn.Linear(model.fc.in_features, 100)

# 使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# 辅助函数：one-hot 转换 和 Dice 损失
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



# mpgd 对抗攻击方法的实现
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
        lam = (i - 1) / (2.0 * num_iter)
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



# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # 生成对抗样本
        adv_inputs = mpgd_attack(model, inputs, targets,
                                 epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                                 mean=(0.5071, 0.4867, 0.4408),
                                 std=(0.2675, 0.2565, 0.2761),
                                 num_classes=100)
        optimizer.zero_grad()
        outputs = model(adv_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0



# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Test Accuracy: {100. * correct / total:.2f}%')


# 训练和测试循环
for epoch in range(num_epochs):
    train(epoch)
    test()
    scheduler.step()

# 保存训练好的模型参数
torch.save(model.state_dict(), 'standard18_adv_CIFAR-100.pth')
