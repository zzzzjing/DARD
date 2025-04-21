import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ----------------------------
# ResNet56 模型定义（适用于 CIFAR-10）
# ----------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        # 第一层卷积（CIFAR-10 图像尺寸较小）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 三个 stage，每个 stage 包含 num_blocks 个 BasicBlock
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet56(num_classes=10):
    # 对于 CIFAR-10，常用的 ResNet-56 配置为 [9, 9, 9]
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


# ----------------------------
# 主函数：数据加载、训练、评估和保存模型
# ----------------------------
def main():
    # 超参数设置（与原代码一致）
    num_classes = 10
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # CIFAR-10 归一化参数
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理与加载
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化 ResNet56 模型
    student_model = ResNet56(num_classes=num_classes)
    student_model = student_model.to(device)

    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # 标准训练过程
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(trainset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 在每个 epoch 后评估测试集准确率
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student_model(inputs)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] 测试集准确率: {acc:.2f}%\n")

    # 保存模型参数
    torch.save(student_model.state_dict(), "standard_resnet56.pth")
    print("标准训练完成，模型参数已保存至 standard_resnet56.pth")


if __name__ == '__main__':
    main()
