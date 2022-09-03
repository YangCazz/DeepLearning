# 搭建EfficientNet模型
# EfficientNet-2019-谷歌
# 在当年的ImageNet top-1上达到当年最高准确率84.3%
# 准确率很高，但是很吃显存
import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# 调整数据为8的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 组合层模块搭建
# Conv + BatchNorm + Activation
# group规定使用DW卷积还是普通卷积
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        # 根据卷积核大小来计算padding, 使输出结果保证为原来的一半大小
        padding = (kernel_size - 1) // 2
        # 默认归一化层为BN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认激活函数为alias Swish
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        # 依次搭建各个层
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())

# 搭建通道注意力模块SE
# AvgPool + fc1 + Swith-LU + fc2 + sigmoid
# 转化结果scale * 初始数据x
class SqueezeExcitation(nn.Module):
    # squeeze_factor设计特征矩阵为原有的1/4
    def __init__(self,
                 input_c: int,   # 模块输入channel
                 expand_c: int,  # 1*1conv的升维比率
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        # 官方喜欢使用1*1卷积来代替全连接层
        # 猜测是官方在底层对卷积做了更多的优化
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()
    # 前向传播
    # AvgPool + fc1 + Swith-LU + fc2 + sigmoid
    def forward(self, x: Tensor) -> Tensor:
        # 自适应全局池化
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        # 输出与原始输入按注意力放缩的结果
        return scale * x



# 逆残差参数输入
# 卷积核大小, 输入张量通道, 输出张量通道, 扩张系数, SE, 步长, drop因子
class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    # index记录当前模块的名称
    # width来记录网络宽度方向上的倍率因子
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        # 根据倍率因子来调整input_c的大小, 然后调整到8的整数倍
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    # 静态方法来调整通道数量
    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)

# 逆残差模块搭建
# 1*1 -BN+SiLU-> 3*3 DW -BN+SiLU-> SE --> 1*1 -BN-> DropOut
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        # 判别步长是否合适
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        # 搭建捷径判别条件
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        # 定义有序的字典
        layers = OrderedDict()
        # 使用SiLU激活
        activation_layer = nn.SiLU  # alias Swish

        # 1*1 -BN+SiLU-> 3*3 DW -BN+SiLU-> SE --> 1*1 -BN-> DropOut
        # 若width扩张因子不为1, 则搭建扩张卷积的模块
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # DW卷积模块搭建
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})
        # SE
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # 1*1 + BN + 线性激活
        # 搭建最后的一个1*1卷积层
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})
        # 搭建好的逆残差放入模型队列
        # 搭建好主分支
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()
    # 前向传播
    def forward(self, x: Tensor) -> Tensor:
        # 主分支计算
        result = self.block(x)
        # 经过dropout
        result = self.dropout(result)
        # 旁支拼接结果
        if self.use_res_connect:
            result += x

        return result

# 整体模型构建
class EfficientNet(nn.Module):
    # width系数, depth系数, 分类类别, FC层前的随机失活比例, 拼接随机失活比率
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # 默认配置表, EfficientNet-B0
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        # 向上取整计算, 计算需要堆叠的次数
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # partial函数用于携带部分参数生成一个新函数
        # 调用partial函数来对函数列表中的某些参数进行更新
        # 计算调整后的channel数量, 最近的8的整数倍
        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # 调用逆残差模块系数来构建逆残差模块
        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        # 统计当前搭建了的MB模块的数量
        b = 0
        # repeats来传入构建的数量, 求和之后就可以的的奥当前网络需要的模块的数量
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        # 将一个可遍历的数据对象（如列表、元组、字典和字符串）组合成一个索引序列
        # 同时列出数据下标和数据
        # 遍历每一个Stage
        for stage, args in enumerate(default_cnf):
            # 复制一份当前的参数
            cnf = copy.copy(args)
            # 遍历每一个Stage当中的参数来搭建模型
            # pop方法将模块的参数从参数列别当中取出用于构建模块
            for i in range(round_repeats(cnf.pop(-1))):
                # 除却第一块, 其它的s=1
                if i > 0:
                    cnf[-3] = 1  # 将s设置为1
                    cnf[1] = cnf[2]  # input_channel equal output_channel
                # 随着模块的构建逐渐累加更新dropout率大小
                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                # 根据ASCii码计算得到index名称
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                # 将处理过的逆残差模块的参数放置到系数列表中
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # 以有序字典的方式来构建模型
        layers = OrderedDict()

        # conv1
        # RGB 3->32
        # 3*3, s=2
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # 根据构建好的参数列表来构建所有的逆残差模块
        # {index: block}
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # 构建最后一层
        # 计算它的输入channel
        last_conv_input_c = inverted_residual_setting[-1].out_c
        # 输出固定为1280
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})
        # 实例化模型的特征提取层
        self.features = nn.Sequential(layers)
        # 自适应平均池化，指定输出（H，W）
        # [N, C, H, W]
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 构建分类器部分
        # dropout + FC
        classifier = []
        # 大于0说明需要dropout层
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))

        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # 前向传播
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# 类别  大小 width depth drop1 drop2
# B0：224*224, 1.0, 1.0, 0.2, 0.2
# B1：240*240, 1.0, 1.1, 0.2, 0.2
# B2：260*260, 1.1, 1.2, 0.2, 0.3
# B3：300*300, 1.2, 1.4, 0.2, 0.3
# B4：380*380, 1.4, 1.8, 0.2, 0.4
# B5：456*456, 1.6, 2.2, 0.2, 0.4
# B6：528*528, 1.8, 2.6, 0.2, 0.5
# B7：600*600, 2.0, 3.1, 0.2, 0.5
def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)