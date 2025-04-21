from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm


# ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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
        self.in_planes = 64
        # CIFAR数据集建议使用3x3卷积，stride=1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


# ResNet56
class CIFARBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(CIFARBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(CIFARResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
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


def ResNet56(num_classes=100):
    # 对于 ResNet56，论文中 n=9：共 3 个阶段，每个阶段 9 个基本模块
    return CIFARResNet(CIFARBasicBlock, [9, 9, 9], num_classes)


# 训练和蒸馏部分
# 解析命令行参数
parser = argparse.ArgumentParser(description='CIFAR-100训练：学生模型ResNet18 & 教师模型ResNet56（pth）')
parser.add_argument('--teacher_path', required=True, type=str, help='写入教师模型的权重文件')
parser.add_argument('--epochs', default=200, type=int, help='训练轮数')
parser.add_argument('--lr', default=0.1, type=float, help='初始学习率')
parser.add_argument('--temp', default=30.0, type=float, help='蒸馏温度')
parser.add_argument('--alpha', default=0.7, type=float, help='蒸馏损失权重')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据预处理及加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
num_classes = 100


# 定义 PGD 对抗攻击模块（用于生成对抗样本）
class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.step_size = config['step_size']

    def forward(self, inputs, targets):
        # 随机初始化扰动
        x = inputs.detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        for _ in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.model(x), targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad)
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.model(x), x


# 构建学生模型（ResNet18）和教师模型（ResNet56）
student = ResNet18(num_classes=num_classes).to(device)
teacher = ResNet56(num_classes=num_classes).to(device)
teacher.load_state_dict(torch.load(args.teacher_path))
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# 包装学生模型为对抗攻击模型
pgd_config = {'epsilon': 8.0 / 255, 'num_steps': 10, 'step_size': 2.0 / 255}
pgd_model = AttackPGD(student, pgd_config)

if device == 'cuda':
    cudnn.benchmark = True

# 定义损失函数：KL散度用于知识蒸馏，交叉熵用于标准分类
kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, epoch, initial_lr):
    # 每100和150个epoch下降学习率
    if epoch in [100, 150]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def train(epoch, optimizer):
    student.train()
    pgd_model.train()
    running_loss = 0.0
    progress = tqdm(trainloader, desc="Train Epoch {}".format(epoch))
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # 生成对抗样本并获得学生模型输出
        outputs_adv, _ = pgd_model(inputs, targets)
        # 教师模型在干净样本上的输出
        teacher_outputs = teacher(inputs)
        # 学生模型在干净样本上的输出
        outputs_clean = student(inputs)
        # 综合损失：知识蒸馏损失 + 交叉熵损失
        loss = args.alpha * args.temp * args.temp * kl_loss(
            F.log_softmax(outputs_adv / args.temp, dim=1),
            F.softmax(teacher_outputs / args.temp, dim=1)
        ) + (1.0 - args.alpha) * ce_loss(outputs_clean, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress.set_description("Loss: {:.4f}".format(loss.item()))
    print("Epoch {} training loss: {:.4f}".format(epoch, running_loss / len(trainloader)))


def test(epoch):
    student.eval()
    pgd_model.eval()
    natural_correct = 0
    adv_correct = 0
    total = 0
    progress = tqdm(testloader, desc="Test Epoch {}".format(epoch))
    with torch.no_grad():
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_clean = student(inputs)
            _, pred_clean = outputs_clean.max(1)
            natural_correct += pred_clean.eq(targets).sum().item()

            outputs_adv, _ = pgd_model(inputs, targets)
            _, pred_adv = outputs_adv.max(1)
            adv_correct += pred_adv.eq(targets).sum().item()
            total += targets.size(0)
            progress.set_description("Acc: {:.2f}%".format(100.0 * adv_correct / total))
    natural_acc = 100.0 * natural_correct / total
    robust_acc = 100.0 * adv_correct / total
    print("Epoch {}: 自然样本准确率: {:.2f}%, 对抗样本准确率: {:.2f}%".format(epoch, natural_acc, robust_acc))


def main():
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        train(epoch, optimizer)
        test(epoch)
    # 训练结束后保存学生模型权重
    torch.save(student.state_dict(), "ARD.pth")
    print("学生模型权重已保存至 ARD.pth")


if __name__ == '__main__':
    main()
