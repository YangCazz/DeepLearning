import os
import time
import datetime
from typing import Union, List

import torch
from torch.utils import data

from src import u2net_full
from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from my_dataset import DUTSDataset
import transforms as T

# 1.进行显著性分割数据预处理-训练集
class SODPresetTrain:
    # 传入参数
    # 批数据大小，裁剪图像大小，水平翻转的概率，标准化处理的均值
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    # 调用函数后, 完成模型训练的数据预处理过程
    def __call__(self, img, target):
        return self.transforms(img, target)

# 2.进行显著性分割数据预处理-测试集
class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    # 调用函数后, 完成模型训练的数据预处理过程
    def __call__(self, img, target):
        return self.transforms(img, target)

# 3.主函数, 训练模型
def main(args):
    # 0.查询GPU当前状态：有则使用第一块GPU，没有则使用CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("当前使用GPU {} .".format(device))
    print(args)
    # 写入参数Batch Size
    batch_size = args.batch_size

    # 用来保存训练以及验证过程中信息
    # 时间-年月日-时分秒
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 1.调用自定义数据集读取部分
    # 实例化训练数据集
    train_dataset = DUTSDataset(args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))
    # 实例化验证数据集
    # 计算可用CPU核心数量最大值，用来加载图片，os.cpu_count()获取系统中的CPU核心的数量
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 2.1载入训练集
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)
    # 2.2载入测试集
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)
    # 3.构建模型
    # 3.1如果存在预训练权重则载入
    model = u2net_full()
    model.to(device)
    # 提取出参数中需要更新的权重
    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    # 3.2构建优化器
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    # 3.3构建学习策略
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
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
    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 5.1分批次处理数据, 计算它们的平均损失和学习率
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # 5.2进行堂前轮的模型评估, 得到评估结果
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
            # 记录当前的结论
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # 5.3评估结果写入到txt文件
            # a表示追加模式，文件不存在则会自动创建，从末尾追加，不可读
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            # 5.4记录当前最好的模型
            if current_mae >= mae_info and current_f1 <= f1_info:
                torch.save(save_file, "save_weights/model_best.pth")

        # 5.5只存储最近的10轮次的模型参数
        if os.path.exists(f"save_weights/model_{epoch-10}.pth"):
            os.remove(f"save_weights/model_{epoch-10}.pth")
        # 5.6存储本轮的模型
        torch.save(save_file, f"save_weights/model_{epoch}.pth")
    # 6.计算模型训练用时
    total_time = time.time() - start_time
    # 6.1将模型用时字符化
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # 6.2输出训练总用时
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net training")

    parser.add_argument("--data-path", default="../../data/DUTS", help="DUTS root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=360, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # 采用模型恢复
    parser.add_argument('--resume', default='./save_weights/model_262.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=262, type=int, metavar='N',
                        help='start epoch')
    # 不采用模型恢复，从头开始训练
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # 混合训练精度参数
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # parser.add_argument("--amp", action='store_ture',
    #                     help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
