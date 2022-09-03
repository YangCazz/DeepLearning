import torch
import torchvision
import torch.nn as nn
from LeNet_Net import MyLeNet
import torch.optim as optim
import torchvision.transforms as transforms
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1. 加载数据：数据加载->数据转化->数据集划分
    # 2. 构建模型：训练模型->迭代参数->得到最好模型
    #
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=True,
                                             download=False, transform=transform)
    # 每一批随机拿出36张图片进行训练，numworks表示载入数据的线程数量
    # linux环境下可以自己定义numworks，windows则只能设置为0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)
    # 转化成迭代器
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    # 设置标签
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 展示图像
    # print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))
    # imshow(torchvision.utils.make_grid(val_image))

    # 构建模型后传入参数----模型、损失函数构建、超参数优化
    net = MyLeNet(3, 10)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练模型
    # 迭代5次
    for epoch in range(5):
        print(f"Epoch {epoch + 1} :")
        # 计算损失
        running_loss = 0.0
        # 遍历训练集样本，初始步为1
        for step, data in enumerate(train_loader, start=0):
            # 从样本列表中提取本批次的样本+标签
            # print(step)
            inputs, labels = data
            # 清除历史梯度
            # 如果不清楚则等价于扩大batch size, 可以当作一个trick，相当于在内存不足的时候扩大批大小
            optimizer.zero_grad()
            # forward + backward + optimize
            # 前向传播计算单趟计算结果->计算反向传播参数->优化参数
            outputs = net(inputs)
            # 计算损失
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # 打印结果数据
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                # with是一个上下文管理器，在测试中不计算梯度变化，冻结参数
                with torch.no_grad():
                    # 得到网络模型输出
                    outputs = net(val_image)  # [batch, 10]
                    # 计算预测值-取最大的作为标签
                    # max在每个向量中寻找最大值及其对应的标签下标，输出index代表预测标签
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 计算准确率
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('训练完成')
    path = "./save"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    save_path = path + '/Lenet.pth'
    torch.save(net.state_dict(), save_path)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    # 由于图像转化为tensor过程中维度顺序发生了改变，这里需要将它转化回来
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()