import os
import time
import datetime

import torch

from src import fcn_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T

# 1.数据预处理--Train数据
class SegmentationPresetTrain:
    # 传入参数
    # 批数据大小，裁剪图像大小，水平翻转的概率，标准化处理的均值
    # 预处理主要流程：随机放缩 + 随机水平翻转 + 随机裁剪 + Tensor化 + 标准化
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 计算得到一个阈值区间[0.5x, 2x]
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # 1.随机放缩
        trans = [T.RandomResize(min_size, max_size)]
        # 2.随机水平翻转
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
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
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 调用基本的数据预处理模块
        # 随机放缩, Tensor化, 归一化
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    # 调用函数后将传入图像列表预处理
    def __call__(self, img, target):
        return self.transforms(img, target)

# 3.分别完成Train和Eval的数据预处理
def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

# 4.构架FCN+BackBone的模型架构
def create_model(aux, num_classes, pretrain=True):
    # 带有捷径分支
    model = fcn_resnet50(aux=aux, num_classes=num_classes)
    # 使用预训练权重进行训练
    if pretrain:
        weights_dict = torch.load("./pre/fcn_resnet50_coco.pth", map_location='cpu')
        # 重构全连接层
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                # 只有全连接层中有分类器, 选中这些模块后删除对应的参数
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        # 打印出删除的部分和多余的参数部分
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

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

    # 用来保存训练以及验证过程中信息
    # results20220826-184424
    # 时间-年月日-时分秒
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 1.调用自定义数据集读取部分
    # 实例化训练数据集
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")
    # 实例化验证数据集
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")
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
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)
    # 提取出参数中需要更新的权重
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    # 提取捷径分支的参数, 合并到训练参数
    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        # 辅助分类器的学习率是初始的10倍
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    # 3.2构建优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 3.3创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
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
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 5.1分批次处理数据, 计算它们的平均损失和学习率
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # 5.2进行堂前轮的模型评估, 得到评估结果
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # 5.3评估结果写入到txt文件
        # a表示追加模式，文件不存在则会自动创建，从末尾追加，不可读
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")
        # 5.4存储模型, 将权重文件分列
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    # 6.计算模型训练用时
    total_time = time.time() - start_time
    # 6.1将模型用时字符化
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 6.1输出训练总用时
    print("training time {}".format(total_time_str))

# 主参数
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="../../data/VOC", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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