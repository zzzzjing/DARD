import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def main():
    # ----------------------------
    # 超参数设置（与原代码一致）
    # ----------------------------
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

    # ----------------------------
    # 数据预处理与加载
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----------------------------
    # 初始化 ResNet18（改造适应 CIFAR-10）
    # ----------------------------
    student_model = torchvision.models.resnet18(weights=None)
    # 修改第一层卷积（CIFAR10 图像较小）
    student_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    student_model.maxpool = nn.Identity()  # 去掉 maxpool 层
    student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    student_model = student_model.to(device)

    # ----------------------------
    # 定义损失函数与优化器
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # ----------------------------
    # 标准训练过程
    # ----------------------------
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

    # ----------------------------
    # 保存模型参数
    # ----------------------------
    torch.save(student_model.state_dict(), "standard_resnet18.pth")
    print("标准训练完成，模型参数已保存至 standard_resnet18.pth")


if __name__ == '__main__':
    main()
