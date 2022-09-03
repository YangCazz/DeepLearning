import os
import time
import datetime

import torch

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T

# 1.数据预处理--Train数据
class SegmentationPresetTrain:
    # 传入参数
    # 批数据大小，裁剪图像大小，水平翻转的概率，标准化处理的均值
    # 预处理主要流程：随机放缩 + 随机水平翻转 + 随机裁剪 + Tensor化 + 标准化
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 计算得到一个阈值区间[0.5x, 2x]
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        # 1.随机放缩
        trans = [T.RandomResize(min_size, max_size)]
        # 2.随机翻转-水平+竖直
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        # 3.随机裁剪+Tensor+标准化
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        # 列表中存储了所有的图像预处理方法，将数据预处理转载进入列表
        self.transforms = T.Compose(trans)

    # 调用函数后, 完成模型训练的数据预处理过程
    def __call__(self, img, target):
        return self.transforms(img, target)

# 2.数据预处理-Eval数据
class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 调用基本的数据预处理模块
        # 随机放缩, Tensor化, 归一化
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    # 调用函数后将传入图像列表预处理
    def __call__(self, img, target):
        return self.transforms(img, target)

# 3.分别完成Train和Eval的数据预处理
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

# 构造模型, 不使用预训练权重
def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

# 5.主函数, 训练模型
def main(args):
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("当前使用GPU {} .".format(device))
    print(args)
    # 写入参数Batch Size
    batch_size = args.batch_size
    # 分类模型的类别数目=目标类型数目+背景
    num_classes = args.num_classes + 1

    # 使用计算出来的均值和方差
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    # 时间-年月日-时分秒
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 1.调用自定义数据集读取部分
    # 实例化训练数据集
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))
    # 实例化验证数据集
    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))
    # 计算可用CPU核心数量最大值，用来加载图片，os.cpu_count()获取系统中的CPU核心的数量
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 2.1载入训练集
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)
    # 2.2载入测试集
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # 3.构建模型
    # 3.1如果存在预训练权重则载入
    model = create_model(num_classes=num_classes)
    model.to(device)
    # 提取出参数中需要更新的权重
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    # 3.2构建优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 3.3创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # 学习率更新策略-Poly方法
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # warmup策略相当于热身, 先从非常小的学习率开始学习然后慢慢上升到我们指定的学习率
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # 4.恢复训练
    # 查看我们最近一次训练的权重, 模型, 优化器, 更新策略, 训练轮次,
    # 恢复现场
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    # 5.训练过程
    # 记录使用时间
    # 记录最好的评估水平
    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 5.1分批次处理数据, 计算它们的平均损失和学习率
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # 5.2进行堂前轮的模型评估, 得到评估结果
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # 5.3评估结果写入到txt文件
        # a表示追加模式，文件不存在则会自动创建，从末尾追加，不可读
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")
        # 5.4存储模型, 将权重文件分列
        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    # 6.计算模型训练用时
    total_time = time.time() - start_time
    # 6.1将模型用时字符化
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 6.2输出训练总用时
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="../../data", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # 混合训练精度参数
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # 将参数列表传入变量
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
