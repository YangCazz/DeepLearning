# 2021.06.15-Changed for implementation of TNT model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch.utils.data
import torch.distributed as dist
import numpy as np

from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader

from .rasampler import RASampler

# 判断该进程是否可以做分布式
# 判断该进程是否已经初始化
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# 获取分布式进程的数量
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 获取进程的排序-所处显卡的序号
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# 创建数据加载器
# 构建默认参数列表, 用timm库完成数据集构建
def create_loader(
        dataset,                    # 数据集
        input_size,                 # 输入大小
        batch_size,                 # 批数据大小
        is_training=False,          # 数据用途
        use_prefetcher=True,        # 是否使用数据预存
        no_aug=False,               # 数据增强-
        re_prob=0.,                 #
        re_mode='const',            #
        re_count=1,                 #
        re_split=False,             #
        scale=None,                 #
        ratio=None,                 #
        hflip=0.5,                  #
        vflip=0.,                   #
        color_jitter=0.4,           #
        auto_augment=None,          #
        num_aug_splits=0,           #
        interpolation='bilinear',   # 插值方法
        mean=IMAGENET_DEFAULT_MEAN, # 归一化均值
        std=IMAGENET_DEFAULT_STD,   # 归一化方差
        num_workers=1,              # 装载数据使用的CPU核心个数
        distributed=False,          # 默认分布式
        crop_pct=None,              #
        collate_fn=None,            # 数据映射方法
        pin_memory=False,           # 内存页
        fp16=False,                 #
        tf_preprocessing=False,     #
        use_multi_epochs_loader=False,  #
        repeated_aug=False          #
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    # 更具参数列表, 构造数据预处理栏
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    # 默认不采样
    sampler = None
    # 采用多GPU
    # Repeated Augmentation-允许一个mini-batch中包含来自同一个图像的不同增强版本
    # 此时mini-batch的各个样本并非是完全独立的，这相当于对同一个样本进行重复抽样
    if distributed:
        if is_training:
            if repeated_aug:
                print('using repeated_aug')
                num_tasks = get_world_size()
                global_rank = get_rank()
                sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        if is_training and repeated_aug:
            print('using repeated_aug')
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
    # 对应不同的情况采用不同的映射策略
    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    # 默认数据加载器为DataLoader
    loader_class = torch.utils.data.DataLoader

    # 采用异步方法来加速CPU装载数据到GPU中，避免出现GPU等CPU的过程
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    # 构建数据装载器
    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
    )
    # 采用prefetcher数据预存的方法来提高数据CPU和GPU的使用效率
    # 采用prefetcher方法初始化方法
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader
