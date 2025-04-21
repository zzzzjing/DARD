import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):

    device = images.device
    # 计算归一化空间下图像的合法取值范围
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=device).view(1, 3, 1, 1)
    # 将 epsilon 与 alpha 转换到归一化空间
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=device).view(1, 3, 1, 1)
    # 初始化对抗样本
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        # 根据梯度符号更新对抗样本（归一化空间下更新）
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 对扰动进行投影，确保每个像素扰动不超过 epsilon（归一化空间下）
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images

def main():

    # 超参数设置
    num_classes = 10
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    # CIFAR-10 归一化参数
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # PGD 攻击参数
    epsilon = 8.0 / 255.0
    alpha = 2.0 / 255.0
    num_iter = 20

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


    # 初始化 ResNet18 , 改造适应 CIFAR-10
    model = torchvision.models.resnet18(weights=None)
    # 修改第一层卷积（适应 CIFAR10 小尺寸图像）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 去掉 maxpool 层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)


    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)


    # 对抗训练过程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            # 使用 PGD 生成对抗样本
            adv_inputs = pgd_attack(model, inputs, targets, epsilon, alpha, num_iter, cifar10_mean, cifar10_std)
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(trainset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 每个 epoch 后在测试集（clean 样本）上评估模型性能
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] 测试集准确率: {acc:.2f}%\n")


    # 保存对抗训练后的模型参数
    torch.save(model.state_dict(), "AT_resnet18_20.pth")
    print("对抗训练完成，模型参数已保存至 AT_resnet18_20.pth")

if __name__ == '__main__':
    main()
