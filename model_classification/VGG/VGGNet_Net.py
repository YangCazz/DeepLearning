# 搭建VGGNet模型
# VGGNet-2014-牛津大学研究组Visial Geometry Group(VGG)提出
# 当年的ImageNet中的 定位任务的第一名 和 分类任务的第二名
import torch.nn as nn
import torch

# 获取官方预训练权重链接
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


# 构建VGG模型
# feature表示初始化参数列表，num_classes表示目标类别，init_weights表示是否赋予随机初始化
class MyVGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(MyVGG, self).__init__()
        self.features = features
        # 所有VGG网络的全连接层架构都是一样的
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    # 正向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 初始化权重函数，遍历网络的每一层，逐层初始化参数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 根据模型参数列表来构建模型
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        # M表示MaxPooling，为数字则为卷积
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 构建卷积层，卷积核3*3
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 增加ReLU层，将其添加到模型列表
            layers += [conv2d, nn.ReLU(True)]
            # 更新输出的维度
            in_channels = v
    # 列表以非关键字方法传入
    return nn.Sequential(*layers)


# VGG模型层参数列表
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 参数实例化函数，将配置参数传入后实例化成为一个具体模型
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model_classification number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = MyVGG(make_features(cfg), **kwargs)
    return model

# 模型调试
# x = torch.rand(size=(8, 3, 224, 224))
# net = vgg(model_name="vgg16", num_classes=5, init_weights=True)
# model_classification = vgg("vgg16")
# print(model_classification)
