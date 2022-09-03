import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from GoogleNet_Net import MyGoogLeNet


def main():
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 1.定义数据预处理函数
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 2.获取数据集，获取数据集在的根目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 返回上上层目录
    image_path = os.path.join(data_root, "data", "Flower")  # 拼接为花朵数据路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 2.1载入训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 获取数据集大小
    train_num = len(train_dataset)
    # 获取分类类别的名称何对应的下标，class_to_idx是数据装载器的固有属性，以其文件夹名称作为标签label，按字母排序
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 以遍历的方法将字典反过来，{0:'daisy', ...}，这是为了可以直接使用下标在字典中获取类别
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    # 计算可用CPU核心数量最大值，用来加载图片，os.cpu_count()获取系统中的CPU核心的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 载入训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # 2.2载入测试集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    '''print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))'''
    print(f"模型训练，使用训练样本{train_num}张图片，使用验证样本{val_num}张图片")
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # 3.模型训练
    net = MyGoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
    # 官方的模型中使用了bn层以及改了一些参数，不能混用
    # import torchvision
    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)

    # 模型传入GPU
    net.to(device)
    # 损失函数定义
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 30

    # 设置保存权重的路径
    path = "./save"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    save_path = path + '/googleNet.pth'
    # torch.save(net.state_dict(), save_path)

    # 3.1模型训练
    # 记录模型准确率
    best_acc = 0.0
    best_epoch = 0
    # 记录模型步骤
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 模型训练过程采用 net.train()
        net.train()
        running_loss = 0.0
        # 采用进度条可视化来查看模型的训练过程
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # 多个输出口径  [x, aux2, aux1]
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            # 损失叠加计算
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 3.2验证模型过程
        net.eval()
        # 累计计算模型准确度
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model_classification only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)

    print(f"训练结束，在第{best_epoch}轮训练后得到最优的模型，其验证准确率为{best_acc}")


if __name__ == '__main__':
    main()