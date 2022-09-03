# 搭建LeNet模型
# LeNet-1998-Yann LeCun
import torch
from torch import nn
import torch.nn.functional as F

# 构建LeNet模型，RGB图像预测
# inchannel为输入图像深度；outchannel为输出张量特征维度
class MyLeNet(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(MyLeNet, self).__init__()
        Model_structure = []
        Classifier_Output = []
        layer1 = nn.Sequential(
            # 卷积(1->6,5*5)      input(3, 32, 32) output(16, 28, 28)
            # ReLU
            # MaxPooling        output(16, 14, 14)
            nn.Conv2d(in_channels=inchannel, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layer2 = nn.Sequential(
            # 卷积(6->16,5*5)     output(32, 10, 10)
            # ReLU
            # MaxPooling        output(32, 5, 5)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        Model_structure.append(layer1)
        Model_structure.append(layer2)
        self.Feature_Extractor = nn.Sequential(*Model_structure)
        FC = nn.Sequential(
            # 铺展
            # 全连接400->120
            # ReLU
            # 全连接120->84
            # ReLU
            # 全连接84->outchannel
            # 输出
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, outchannel)
        )
        Classifier_Output.append(FC)
        self.classifier = nn.Sequential(*Classifier_Output)

    def forward(self,x):
        """前向传播"""
        # 特征提取
        features = self.Feature_Extractor(x)
        # 将提取到的特征铺展开来
        feature_trans = features.view(x.size(0), -1)
        # view(-1,-1)
        # 铺展开的特征放入分类器中,得出分类结果
        result = self.classifier(feature_trans)
        return result

# 调试
# 随机生成图像张量8张3维的32*32大小的图像[batch,channel,height,width]
# 输出10类判别

# x = torch.rand(size=(8,3,32,32))
# model_classification = MyLeNet(inchannel=3, outchannel=10)
# print(model_classification)
# output = model_classification(x)
