import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 设置随机种子，保证结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 CIFAR-10 的归一化均值和标准差
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

#  PGD 对抗攻击实现函数
def pgd_attack(model, images, labels, epsilon, alpha, num_iter, mean, std):
    # 计算归一化后图像的合法取值范围（原始图像取值在 [0,1]）
    lower_bound = torch.tensor([(0 - m) / s for m, s in zip(mean, std)],
                               device=images.device).view(1, 3, 1, 1)
    upper_bound = torch.tensor([(1 - m) / s for m, s in zip(mean, std)],
                               device=images.device).view(1, 3, 1, 1)
    # 将 epsilon 与步长从原始像素空间转换到归一化空间（每个通道除以 std）
    epsilon_tensor = torch.tensor([epsilon / s for s in std],
                                  device=images.device).view(1, 3, 1, 1)
    alpha_tensor = torch.tensor([alpha / s for s in std],
                                device=images.device).view(1, 3, 1, 1)

    # 初始化对抗样本为原始图像
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(num_iter):
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        # 根据梯度符号更新对抗样本
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        # 对扰动进行投影，确保每个像素的扰动不超过 epsilon（归一化空间下）
        perturbation = torch.clamp(adv_images - images, min=-epsilon_tensor, max=epsilon_tensor)
        adv_images = torch.clamp(images + perturbation, lower_bound, upper_bound).detach_()
        adv_images.requires_grad = True

    return adv_images

def random_uniform_perturbation(epsilon, shape, std, device):
    # 如果 std 是 tuple 或 list，则构造与 shape 匹配的标准差张量
    if isinstance(std, (tuple, list)):
        # 假设输入 shape 为 (N, C, H, W)
        std_tensor = torch.tensor(std, device=device).view(1, len(std), 1, 1).expand(shape)
    else:
        std_tensor = torch.full(shape, std, device=device)
    noise = torch.normal(
        mean=torch.zeros(shape, device=device),
        std=std_tensor
    )
    delta = torch.clamp(noise, -epsilon, epsilon)
    return delta


def compute_loss(model, x, y, kappa=0):
    outputs = model(x)
    # 获取正确类别的 logit
    correct_logit = outputs.gather(1, y.view(-1, 1))
    # 构造 mask，将正确类别对应的位置屏蔽掉（设为 -∞）
    mask = torch.ones_like(outputs, dtype=torch.bool)
    mask.scatter_(1, y.view(-1, 1), False)

    other_logits = outputs.masked_fill(~mask, -1e4)
    max_other_logit, _ = torch.max(other_logits, dim=1, keepdim=True)

    # 计算 margin（差值），加上 kappa 以调控攻击强度
    margin = max_other_logit - correct_logit + kappa
    loss = margin.mean()
    return loss

def clip(delta, min_val, max_val):
    return torch.clamp(delta, min_val, max_val)

def adjust_range(x, delta):
    # 保证 x+delta 落在 [0,1] 范围内，然后返回 x+delta 与 x 的差值
    adv = torch.clamp(x + delta, 0, 1)
    return adv - x

def global_sampling(x, y, model, epsilon, num_samples, std):
    # 在全局范围内采样 num_samples 个随机扰动, 并选出使得损失最大的扰动作为初始化扰动。
    candidate_losses = []
    candidate_deltas = []
    for i in range(num_samples):
        delta = random_uniform_perturbation(epsilon, x.shape, std, device=x.device)
        x_candidate = torch.clamp(x + delta, 0, 1)
        loss = compute_loss(model, x_candidate, y).item()
        candidate_losses.append(loss)
        candidate_deltas.append(delta)
    max_idx = np.argmax(candidate_losses)
    best_delta = candidate_deltas[max_idx]
    return best_delta

def pgd_refinement(x, y, model, init_delta, epsilon, alpha, num_iterations):
    # 利用 PGD 对初始化扰动进行细化优化
    delta = x.to(x.device)
    for t in range(num_iterations):
        delta.requires_grad = True
        adv = x + delta
        loss = compute_loss(model, adv, y)
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        # 按照梯度方向更新扰动
        delta = delta + alpha * grad.sign()
        delta = clip(delta, -epsilon, epsilon)
        delta = adjust_range(x, delta)
        delta = delta.detach()
    return delta

def two_stage_attack(x, y, model, epsilon, num_samples, alpha, num_iterations, std):
    # 两阶段攻击：首先利用全局采样找出一个较好的初始扰动, 然后利用 PGD 对该扰动进行细化。
    init_delta = global_sampling(x, y, model, epsilon, num_samples, std)
    refined_delta = pgd_refinement(x, y, model, init_delta, epsilon, alpha, num_iterations)
    adversarial_example = torch.clamp(x + refined_delta, 0, 1)
    return adversarial_example

#  评估函数：支持干净样本和对抗样本评估

def evaluate_model(model, data_loader, attack_method=None, attack_params=None):
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if attack_method is not None:
            # 临时启用梯度计算以生成对抗样本
            with torch.enable_grad():
                # 根据不同攻击方法调用对应的参数（这里只支持 pgd_attack 和 two_stage_attack）
                if attack_method.__name__ == 'pgd_attack':
                    inputs = attack_method(
                        model=model,
                        images=inputs,
                        labels=labels,
                        epsilon=attack_params['epsilon'],
                        alpha=attack_params['alpha'],
                        num_iter=attack_params['num_iter'],
                        mean=attack_params['mean'],
                        std=attack_params['std']
                    )
                elif attack_method.__name__ == 'two_stage_attack':
                    inputs = attack_method(
                        x=inputs,
                        y=labels,
                        model=model,
                        epsilon=attack_params['epsilon'],
                        num_samples=attack_params['num_samples'],
                        alpha=attack_params['alpha'],
                        num_iterations=attack_params['num_iter'],
                        std=attack_params['std']
                    )
                else:
                    inputs = attack_method(model, inputs, labels,
                                           attack_params['epsilon'],
                                           attack_params['alpha'],
                                           attack_params['num_iter'],
                                           attack_params.get('mean', None),
                                           attack_params['std'])
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# 验证函数：计算验证集上的平均损失
def validate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss


#数据加载、训练与评估
def main():
    # 定义包含所有参数的配置，其中部分参数仅用于评估
    full_config = {
        'num_epochs': 50,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
        # 以下参数仅用于评估，不参与训练配置的比对
        'pgd_epsilon': 8/255,
        'pgd_alpha': 2/255,
        'pgd_num_iter': 5,
        'seed': 42,
        'early_stop_patience': 5,    # 早停的耐心
        'lr_scheduler_factor': 0.1,  # 学习率下降因子
        'lr_scheduler_patience': 3   # 学习率调度的耐心
    }

    # 分离出仅用于训练的参数（不包含评估相关的参数）
    training_config = {k: v for k, v in full_config.items() if k not in ('pgd_epsilon', 'pgd_alpha', 'pgd_num_iter')}

    # 设置随机种子
    set_seed(full_config['seed'])

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(),         # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    # 加载 CIFAR-10 数据集
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
    # 从训练集中划分出 10% 作为验证集
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=full_config['batch_size'],
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100,
                                             shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                              shuffle=False, num_workers=2)

    # 定义模型 —— 使用 ResNet18 并调整以适应 CIFAR-10
    model = torchvision.models.resnet18(weights=None)
    # 修改第一层卷积：kernel_size=3, stride=1, padding=1 更适合 32×32 图像
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化层
    # 修改最后全连接层，使输出类别数为10
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # 定义损失函数与优化器
    global criterion  # 使得 criterion 在 pgd_attack 中可调用
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=full_config['lr'],
                          momentum=full_config['momentum'],
                          weight_decay=full_config['weight_decay'])
    # 学习率调度器，根据验证集损失调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=full_config['lr_scheduler_factor'],
                                                     patience=full_config['lr_scheduler_patience'],
                                                     verbose=True)

    # 检查训练配置是否发生改变（仅比较训练参数，不包括评估相关参数）
    config_path = 'resnet_cifar10_config.json'
    weights_path = 'resnet_cifar10.pth'
    need_retrain = True
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        # 如果训练配置相同且模型参数存在，则直接加载，否则重新训练
        if saved_config == training_config and os.path.exists(weights_path):
            need_retrain = False
            print("检测到相同的训练配置，加载保存的模型参数...")
        else:
            print("检测到训练配置已修改或模型参数不存在，重新训练模型...")
    else:
        print("未找到配置文件，开始训练模型...")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 训练过程
    if need_retrain:
        num_epochs = full_config['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, len(train_loader), running_loss / 100))
                    running_loss = 0.0

            # 计算验证集损失
            val_loss = validate(model, val_loader)
            print("Epoch [{}/{}] 验证集损失: {:.4f}".format(epoch + 1, num_epochs, val_loss))

            # 根据验证集损失调整学习率
            scheduler.step(val_loss)

            # 早停判断：若验证集损失降低则保存模型，否则计数增加
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), weights_path)
                print("验证集损失降低，保存模型。")
            else:
                epochs_no_improve += 1
                print("连续 {} 个 epoch 验证集损失未降低。".format(epochs_no_improve))
                if epochs_no_improve >= full_config['early_stop_patience']:
                    print("早停触发，提前结束训练。")
                    break

        # 保存训练配置（仅保存训练参数）
        with open(config_path, 'w') as f:
            json.dump(training_config, f)
    else:
        # 直接加载模型参数
        model.load_state_dict(torch.load(weights_path, map_location=device))

    # 评估模型

    clean_acc = evaluate_model(model, test_loader, attack_method=None)
    print("\n在干净测试集上的准确率: {:.2f}%".format(clean_acc))

    # 分别使用 PGD 和两阶段攻击在不同迭代次数下评估
    attack_iters = [10, 20, 30, 50, 60, 80, 100]
    print("\n对抗攻击评估（不同迭代次数）：")
    for iters in attack_iters:
        print("\n--- 迭代次数: {} ---".format(iters))
        # PGD 攻击参数（从 full_config 中获取评估相关参数）
        pgd_params = {
            'epsilon': full_config['pgd_epsilon'],
            'alpha': full_config['pgd_alpha'],
            'num_iter': iters,
            'mean': cifar10_mean,
            'std': cifar10_std
        }
        pgd_acc = evaluate_model(model, test_loader, attack_method=pgd_attack, attack_params=pgd_params)
        print("PGD 攻击下的准确率: {:.2f}%".format(pgd_acc))

        # 两阶段攻击参数：设定全局采样候选扰动数固定为 10
        two_stage_params = {
            'epsilon': full_config['pgd_epsilon'],
            'alpha': full_config['pgd_alpha'],
            'num_iter': iters,  # 表示 PGD 细化阶段的迭代次数
            'num_samples': 10,
            'std': cifar10_std
        }
        two_stage_acc = evaluate_model(model, test_loader, attack_method=two_stage_attack, attack_params=two_stage_params)
        print("两阶段攻击下的准确率: {:.2f}%".format(two_stage_acc))

if __name__ == '__main__':
    main()
