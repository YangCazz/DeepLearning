import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from ConvNeXt_Net import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main(args):
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("当前使用GPU {} .".format(device))
    print(args)

    # 创建存储路径
    path = './weight' + '/' + 'huawei' + '/' + 'ConvNeXt_' + args.model_type
    if os.path.exists(path) is False:
        os.makedirs(path)

    # 使用Tensorboard显示结果
    print("打开Tensorboard面板, 远程请使用端口配对, 打开http://localhost:16006/")
    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)


    # 1.定义数据预处理函数, 设置Std和Mean
    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
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
    # 3.构建模型
    # 如果存在预训练权重则载入
    if args.model_type == 'tiny':
        model = convnext_tiny(num_classes=args.num_classes).to(device)
    elif args.model_type == 'small':
        model = convnext_small(num_classes=args.num_classes).to(device)
    elif args.model_type == 'base':
        model = convnext_base(num_classes=args.num_classes).to(device)
    elif args.model_type == 'large':
        model = convnext_large(num_classes=args.num_classes).to(device)
    elif args.model_type == 'xlarge':
        model = convnext_xlarge(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    # 4.冻结权重
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

    # pg = [p for p in model_classification.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    # 记录最好的成绩
    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # 写入tensorboard面板
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalars('LOSS', {tags[0]: train_loss,
                                       tags[2]: val_loss}, epoch)
        tb_writer.add_scalars('ACC', {tags[1]: train_acc,
                                      tags[3]: val_acc}, epoch)
        # 保存权重
        save_path = path + "/best_model.pth"
        if best_acc < val_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            best_epoch = epoch + 1
        print(f"在训练中第{best_epoch}轮次训练后，达到最好效果：{best_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="../../data/Huawei_all")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./pre/convnext_tiny_1k_224.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_type', default='tiny', help='model in (tiny,small,base,large,xlarge)')
    # 形成参数序列, 输入模型列表
    opt = parser.parse_args()

    main(opt)