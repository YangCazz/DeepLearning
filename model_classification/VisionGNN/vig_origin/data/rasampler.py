# from https://github.com/facebookresearch/deit/blob/main/samplers.py
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.distributed as dist
import math

# Reapted Sampler 策略
# 此时mini-batch的各个样本并非是完全独立的，这相当于对同一个样本进行重复抽样
class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """
    # 传入参数--数据集, 数据复制, ,打乱顺序
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 不采用数据复制策略时, 扩增系数=GPU数量
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # 不指定GPU号时, 自动从系统获取
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 计算需要抽样出来的数据个数=L*3 / 复制系数
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        # 计算取整之后的实际采样数据大小
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        # 两个斜杠即双斜杠（//）表示地板除，即先做除法（/），然后向下取整（floor）
        # 计算实际采样数量
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        # 基于epoch的确定性随机打乱
        g = torch.Generator()
        # 以epoch为随机种子
        g.manual_seed(self.epoch)
        if self.shuffle:
            # randperm, 把1到n这些数随机打乱得到的一个数字序列
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # 进行3轮序列复制indices=[indices, indices, indices]
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # 对应不同的显卡, 按步长(复制系数)进行采样
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # 返回一个迭代器, 将list转化为迭代器
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
