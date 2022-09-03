# 搭建MobileNet模型
# MobileNet-2017-谷歌团队-轻量级神经网络模型参数只有VGG的1/32
# 这个模型是为了能够放到移动设备上而建设，适用于较小设备或者嵌入式设备
# 搭建MobileNet_V2
# MobileNet_V2-2018
from torch import nn
import torch

# 调整数据为8的整数倍
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 限制最小的卷积核个数为8
    if min_ch is None:
        min_ch = divisor
    # 将其调整到最近的8的倍数的数值(计算中适用四舍五入的规则)
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 需要确保向下取整的时候不会削减超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# 组合层模块搭建
# Conv + BatchNorm + ReLU6
class ConvBNReLU(nn.Sequential):
    # group=1为普通卷积, group=in_channels时为DW卷积
    # 默认卷积大小3*3, 步距1
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 根据卷积核大小来计算padding, 使特征图每次缩小一半
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


# 定义倒残差结构-两头小中间大
# 1*1-->3*3 DW-->1*1
class InvertedResidual(nn.Module):
    # expand_ratio扩大因子
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # 计算隐藏层通道的数量(1*1卷积计算得到的结果)
        hidden_channel = in_channel * expand_ratio
        # 判别是否产生捷径
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 使用扩大因子来判别是否为首层, 首层不设计1*1卷积层
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            # group=in_channels时为DW卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        # 将卷积过程组件装入模块列表
        self.conv = nn.Sequential(*layers)

    # 前向传播计算
    def forward(self, x):
        # 根据具体情况来判别是否产生捷径分支
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 构建MobileNet_V2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        # 使用倒残差结构
        block = InvertedResidual
        # 根据alpha来将其转化为实际的8的整数倍, 目的是更好的适用计算机底层
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 传入框架定义参数
        # t-扩展因子
        # c-输出特征矩阵深度
        # n-倒残差层的重复次数
        # s-模块第一层的步长
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # 第一个卷积层, 其参数需要手动定义, 不在参数矩阵中
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 搭建倒残差结构
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # 首层的步长参数由参数列表给出, 其余层固定为1
                stride = s if i == 0 else 1
                # 添加一系列的倒残差模块
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                # 参数传递
                input_channel = output_channel
        # 搭建倒残差结构之后的计算层
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # 将搭建好的结构拼接在一起
        # 特征提取部分都完成了
        self.features = nn.Sequential(*features)

        # 搭建分类器部分
        # AvgPool + DropOut + Linear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # 权重初始化
        # 遍历子模块, 进行对应的参数初始化
        for m in self.modules():
            # 卷积层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # 存在bias则同时对bias初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # BN层初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 线性层初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 模型调试
# x = torch.rand(size=(8, 3, 224, 224))
# net = MobileNetV2(num_classes=5, alpha=1.0)
# print(net)
