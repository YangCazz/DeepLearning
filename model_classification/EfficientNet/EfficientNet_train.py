import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from EfficientNet_Net import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from EfficientNet_Net import efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # print(f"开始模型{type}的训练")
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("当前使用GPU {} .".format(device))
    print(args)
    # 使用Tensorboard显示结果
    print("打开Tensorboard面板, 远程请使用端口配对, 打开http://localhost:16006/")
    tb_writer = SummaryWriter()
    # 创建存储路径
    path = './weight' + '/' + 'Efficient_' + args.model_type
    if os.path.exists(path) is False:
        os.makedirs(path)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # 根据模型类别来确定图像的处理尺寸
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = args.model_type
    # 1.定义数据预处理函数, 设置Std和Mean
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    # 写入参数Batch Size
    batch_size = args.batch_size
    # 计算可用CPU核心数量最大值，用来加载图片，os.cpu_count()获取系统中的CPU核心的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('每个进程同时使用 {} 个dataloader workers来加载数据到CPU'.format(nw))
    # 载入训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    # 2.2载入测试集
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    if args.model_type == 'B0':
        model = efficientnet_b0(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B1':
        model = efficientnet_b1(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B2':
        model = efficientnet_b2(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B3':
        model = efficientnet_b3(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B4':
        model = efficientnet_b4(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B5':
        model = efficientnet_b5(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B6':
        model = efficientnet_b6(num_classes=args.num_classes).to(device)
    elif args.model_type == 'B7':
        model = efficientnet_b7(num_classes=args.num_classes).to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    # Ture则只训练最后的1*1conv+FC
    # Fales则训练全部的结构
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    # 提取出参数中需要更新的权重
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # 训练模型
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # 验证模型
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)
        # 输出准确率
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # 打标签
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        # 保存权重
        save_path = path + "/model_classification-{}.pth"
        torch.save(model.state_dict(), save_path.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../../data/Flower/flower_photos")

    # download model_classification weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str, default='./pre/efficientnetb2.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_classification-type', type=str, default='B2',
                        help='model_classification in (EfficientNet_(B[0-7])')
    # 形成参数序列, 输入模型列表
    opt = parser.parse_args()
    main(opt)