# 搭建GoogleNet模型
# GoogleNet-2014-谷歌团队
# 当年的ImageNet中分类任务的第一名
# 也叫InceptionNet, 有V1, V2, V3, V4, V5, ResNet等多个发展版本
# 其核心想法在于并联计算多尺度信息
import torch.nn as nn
import torch
import torch.nn.functional as F

# 定义GoogLeNet
# 类别个数,是否使用辅助分类器, 是否使用事先优化器
class MyGoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(MyGoogLeNet, self).__init__()
        # 是否使用分支辅助输出模式
        self.aux_logits = aux_logits

        # 第一层
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # self.LBN1 = nn.LocalResponseNorm

        # 第二层
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第三层
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第四层
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 第五层
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 如果使用分支辅助输出模式，则构建相应模块
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        # 自适应平均池化下采样操作，无论输入大小是多少都可以做对应计算
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        # 初始化权重函数
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 第一层
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)

        # 第二层
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # 第三层
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)

        # 第四层
        # N x 480 x 14 x 14
        x = self.inception4a(x)

        # 第一个辅助输出层
        # 评估模型中不输出内容
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        # 第二个辅助输出层
        # 评估模型中不输出内容
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)

        # 第五层
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        # 最终输出层
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)

        # 评估模型中不输出内容
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    # 初始化参数矩阵
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 搭建Inception结构模板
# 结构中共有4个分支
class Inception(nn.Module):
    # 传入参数为其间的卷积核数量
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 分支1, 1*1大小的卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 分支2, 1*1 + 3*3 卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )
        # 分支3, 1*1 + 5*5 卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )
        # 分支4, MaxPool + 1*1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    # 数据前向传递
    def forward(self, x):
        # 将输入并行放入四个分支中
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 得到计算结果
        outputs = [branch1, branch2, branch3, branch4]
        # 输出合并参数，在深度C的维度进行合并[N, C, H, W]
        return torch.cat(outputs, 1)

# 搭建 辅助分类器的 结构模板
# 平均池化 + 1*1 卷积 + fc1 + fc2
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # 平均池化层
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        # 卷积层
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        # 全连接层
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    # 前向传播过程
    def forward(self, x):
        # aux1: N x 512 x 14 x 14,  aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4,    aux2: N x 528 x 4 x 4
        x = self.conv(x)

        # 展平
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)

        # fc1
        # 随机丢弃0.5比0.7好, training这个参数随模型的训练和测试发生变化
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)

        # fc2
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x

# 搭建Conv+ReLU的模块, 方便搭建整个模型
class BasicConv2d(nn.Module):
    # in_channels表示输入矩阵的深度, out_channels表示输出矩阵的深度
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
    # 模块中的数据前向传递过程
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# 模型调试
# x = torch.rand(size=(8, 3, 224, 224))
# net = MyGoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
# print(net)
# model_classification = net(x)