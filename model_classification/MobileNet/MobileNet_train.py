import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from MobileNet_Net_V2 import MobileNetV2
from MobileNet_Net_V3 import mobilenet_v3_small, mobilenet_v3_large

# 模型产生
def model_generate(device,type ,pre_path, save_path):
    print(f"开始模型{type}的训练")
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 1.定义数据预处理函数, 使用迁移学习需要设置Std和Mean
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
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
    # 标签写入json
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
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

    print(f"模型训练，使用训练样本{train_num}张图片，使用验证样本{val_num}张图片")


    # 3.模型训练
    # 使用迁移学习方法
    if type == "MobileNetV2":
        net = MobileNetV2(num_classes=5)
    elif type == "MobileNetV3_small":
        net = mobilenet_v3_small(num_classes=5)
    elif type == "MobileNetV3_large":
        net = mobilenet_v3_large(num_classes=5)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = pre_path
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # 冻结特征提取部分的参数, 不做梯度更新
    # 只训练最后全连接层的参数
    for param in net.features.parameters():
        param.requires_grad = False

    # 模型传入GPU
    net.to(device)
    # 损失函数定义
    loss_function = nn.CrossEntropyLoss()
    # 提取模型超参数
    params = [p for p in net.parameters() if p.requires_grad]
    # 定义优化器
    optimizer = optim.Adam(params, lr=0.0001)


    epochs = 5


    # 设置保存权重的路径
    path = "./save"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    # save_path = path + '/ResNet34.pth'
    save_path = path + save_path
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
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # 输出当前记录
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
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print()

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)

    print(f"训练结束，在第{best_epoch}轮训练后得到最优的模型，其验证准确率为{best_acc}")
    print()


if __name__ == '__main__':
    model_generate(device="cuda:0", type = "MobileNetV2",
                   pre_path='./pre/mobilenet_v2-pre.pth', save_path='/MobileNet_V2.pth')
    model_generate(device="cuda:1", type="MobileNetV3_small",
                   pre_path='./pre/mobilenet_v3_small-pre.pth', save_path='/MobileNet_V3_small.pth')
    model_generate(device="cuda:0", type="MobileNetV3_large",
                   pre_path='./pre/mobilenet_v3_large-pre.pth', save_path='/MobileNet_V3_large.pth')