# ShuffleNet模型训练
import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from ShuffleNet_Net import shufflenet_v2_x1_0, shufflenet_v2_x0_5, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # print(f"开始模型{type}的训练")
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("当前使用GPU {} .".format(device))
    # 打印模型超参数列表
    print(args)
    # 使用Tensorboard显示结果
    print("打开Tensorboard面板, 远程请使用端口配对, 打开http://localhost:16006/")
    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    # 创建存储路径
    path = './weight' + '/' + args.model_type
    if os.path.exists(path) is False:
        os.makedirs(path)
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")


    # 读取数据路径
    # 将路径分解为：训练集-图像路径、图像标签；验证集-图像路径、图像标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
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
    # shufflenetv2_x0_5, shufflenetv2_x1, shufflenetv2_x1_5, shufflenetv2_x2_0
    if args.model_type == 'shufflenetv2_x0_5':
        model = shufflenet_v2_x0_5(num_classes=args.num_classes).to(device)
    elif args.model_type == 'shufflenetv2_x1_0':
        model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    elif args.model_type == 'shufflenetv2_x1_5':
        model = shufflenet_v2_x1_5(num_classes=args.num_classes).to(device)
    elif args.model_type == 'shufflenetv2_x2_0':
        model = shufflenet_v2_x2_0(num_classes=args.num_classes).to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 删除分类层的权重
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    # 提取出参数中需要更新的权重
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 定义学习率变化函数
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
        # save_root = args.save_path
        # path = args.save_path + '/weight'
        save_path = path + "/model_classification-{}.pth"
        torch.save(model.state_dict(), save_path.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../../data/Flower/flower_photos")

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default='./pre/shufflenetv2_x2_0-pre.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--save-path', type=str, default='/shufflenetv2_x1_0')
    parser.add_argument('--model_classification-type', type=str, default='shufflenetv2_x2_0',
                        help='model_classification in (shufflenetv2_x0_5, shufflenetv2_x1_0, '
                             'shufflenetv2_x1_5, shufflenetv2_x2_0)')
    # 形成参数序列, 输入模型列表
    opt = parser.parse_args()
    main(opt)