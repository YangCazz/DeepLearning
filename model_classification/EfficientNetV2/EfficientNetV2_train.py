import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from EfficientNetV2_net import efficientnetv2_s, efficientnetv2_l, efficientnetv2_m

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # print(f"开始模型{type}的训练")
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = args.device
    print("当前使用GPU {} .".format(device))
    print(args)
    # 使用Tensorboard显示结果
    print("打开Tensorboard面板, 远程请使用端口配对, 打开http://localhost:16006/")
    tb_writer = SummaryWriter()
    # 创建存储路径
    path = './weight' + '/' + 'EfficientV2_' + args.model_type
    if os.path.exists(path) is False:
        os.makedirs(path)
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # 根据模型类别来确定图像的处理尺寸
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = args.model_type
    # 1.定义数据预处理函数, 设置Std和Mean
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

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
    # model_classification = create_model(num_classes=args.num_classes).to(device)
    if args.model_type == 's':
        model = efficientnetv2_s(num_classes=args.num_classes).to(device)
    elif args.model_type == 'm':
        model = efficientnetv2_m(num_classes=args.num_classes).to(device)
    elif args.model_type == 'l':
        model = efficientnetv2_l(num_classes=args.num_classes).to(device)

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
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    # 提取出参数中需要更新的权重
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    train_loss_record=[]
    train_acc_record=[]
    val_loss_record=[]
    val_acc_record=[]
    for epoch in range(args.epochs):
        # 训练模型
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        scheduler.step()

        # 验证模型
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_acc_record.append(val_acc)
        val_loss_record.append(val_loss)
        # 打标签
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # 写入tensorboard面板
        print(train_acc)
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalars('LOSS', {tags[0]: train_loss_record[epoch],
                                       tags[2]: val_loss_record[epoch]}, epoch)
        tb_writer.add_scalars('ACC', {tags[1]: train_acc_record[epoch],
                                      tags[3]: val_acc_record[epoch]}, epoch)

        # 保存权重
        save_path = path + "/model_classification-{}.pth"
        torch.save(model.state_dict(), save_path.format(epoch))
        # torch.save(model_classification.state_dict(), "./weights/model_classification-{}.pth".format(epoch))

    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    # for epoch in range(args.epochs):
    #     tb_writer.add_scalars('LOSS', {tags[0]: train_loss_record[epoch],
    #                                   tags[2]: val_loss_record[epoch]}, epoch)
    #     tb_writer.add_scalars('ACC', {tags[1]: train_acc_record[epoch],
    #                                  tags[3]: val_acc_record[epoch]}, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../../data/Flower/flower_photos")

    # download model_classification weights
    # 链接: https://pan.baidu.com/s/1uZX36rvrfEss-JGj4yfzbQ  密码: 5gu1
    parser.add_argument('--weights', type=str, default='./pre/pre_efficientnetv2-m.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_type', default='m', help='model_classification in (s, m, l)')
    # 形成参数序列, 输入模型列表
    opt = parser.parse_args()

    main(opt)
