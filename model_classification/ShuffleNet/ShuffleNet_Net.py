# 搭建 ShuffleNet 模型
# ShuffleNet-2018
from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn

# 实现Channel Shuffle功能
# 功能：将channel分组, 分组之后再将各组组员channel随机组合成
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    # 从输入的x->[batch size, channels, height, width]中提取大小
    batch_size, num_channels, height, width = x.size()
    # 根据划分目标组数量计算每组的通道数量
    # // 来向下取整
    channels_per_group = num_channels // groups

    # 分组重塑
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 将数据x中的两个维度交换位置
    # contigous()来将内存中不连续的数据按连续的方式排列, 方便后续处理
    x = torch.transpose(x, 1, 2).contiguous()

    # 还原, 将重塑后的矩阵还原回原来的形式
    # 与原来形式相比只是多了channel的分组和channel顺序的改变
    x = x.view(batch_size, -1, height, width)

    # 完成channel shuffle后返回结果
    return x


# 搭建ShuffleNet的逆残差模块
# 输入特征矩阵的通道数input_c, 输出特征矩阵的通道数output_c: int, DW卷积采用的步距stride
class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        # 判别步长是否合理
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        # 计算输出是否为2的整数倍
        # 当stride为1时，input_channel应该是branch_features的两倍
        assert output_c % 2 == 0
        branch_features = output_c // 2

        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        # s=2时, 左分支需要做channel维度转化
        # 左：3*3 DW s=2, BN, 1*1 conv, BN+ReLU
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        # 右：1*1 conv, BN+ReLU, 3*3 DW s=2, BN,1*1 conv, BN+ReLU
        # s=2时, 没有进行分支深度划分操作, 深度为原有深度
        # s=1时, 进行了分支深度划分操作, 深度为原来的一半
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    # 静态方法来定义DW卷积
    # 无bias, 因为后续接着BN层
    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    # 前向传播
    def forward(self, x: Tensor) -> Tensor:
        # s=1进行channel均分处理
        if self.stride == 1:
            # [N, C, H, W]
            # 使用x.chunk(2, dim=1), 在第2个维度C上进行均分处理
            x1, x2 = x.chunk(2, dim=1)
            # 左分支不做处理, 右分支经过数据处理
            # 处理之后使用cat((), dim=1)在第2个维度上拼接
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # s=1时, 无需均分,左右分支都需要做处理
            # 最后输出的channel应当为原来的2倍
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # 拼接完的输出需要再经过shuffle的2分组处理
        out = channel_shuffle(out, 2)
        # 返回模块计算结果
        return out

# V2的搭建过程
# Conv1 --> MaxPool --> Stage1 --> Stage2
# --> Stage3 --> Stage4 --> Conv5 --> GlobalPool --> FC
# Stage1: s=2, s=1
# Stage2: s=2, s=1
# Stage3: s=2, s=1
# Stage4: s=2, s=1
class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        # 每个Stage里面需要三个逆残差模块
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        # 整个架构共有5个模块, conv1+Stage1+Stage2+Stage3+Stage4+conv5
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        # Conv1 + MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 申明3个Stage是由nn.Sequential实现的
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        # 定义Stage
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        # 搭建Stage的模块
        # 名称, 模块重复次数, 输出深度
        # zip方法来同时遍历各个列表
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            # 首先搭建第一个s=2的逆残差模块
            # 输出channel与输入的不同
            seq = [inverted_residual(input_channels, output_channels, 2)]
            # 搭建剩下的s=1的模块
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            # 使用setattr方法来设置变量
            # 变量的名称就是Stage2-4
            # 变量的值就是为seq一系列层结构
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # 取出Conv5的channel大小
        output_channels = self._stage_out_channels[-1]
        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # FC
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # 全局池化, 使用mean来完成, 最后只剩下[N, C]
        # [N, C, H, W]
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model