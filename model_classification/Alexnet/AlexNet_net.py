# 搭建AlexNet模型
# AlexNet-2012-Hinton和他的学生Alex
import torch
from torch import nn

class MyAlexNet(nn.Module):
    def __init__(self, class_num=1000, init_weights=False):
        super(MyAlexNet, self).__init__()
        Model_structure = []
        Classifier_Output = []
        # Conv + Pooling Layer
        CP_layer1 = nn.Sequential(
            # Inpt 224*224*3
            # =conv1=>>>  Output 55*55*96
            # =maxpooling=>>>  Output 27*27*96
            # Conv----ReLU----MaxPooling----Normal
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        CP_layer2 = nn.Sequential(
            # Inpt 27*27*96
            # =conv1=>>>  Output 27*27*256
            # =maxpooling=>>> Output 13*13*256
            # Conv----ReLU----MaxPooling----Normal
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        CP_layer3 = nn.Sequential(
            # Inpt 13*13*256
            # =conv1=>>> Output 13*13*384
            # =conv2=>>> Output 13*13*384
            # =conv3=>>> Output 13*13*256
            # =maxpooling=>>> Output 6*6*256
            # Conv----ReLU----MaxPooling----Normal
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        Model_structure.append(CP_layer1)
        Model_structure.append(CP_layer2)
        Model_structure.append(CP_layer3)
        self.Feature_Extractor = nn.Sequential(*Model_structure)

        FC_layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        FC_layer2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )
        FC_layer3 = nn.Sequential(
            nn.Linear(2048, 1000)
        )
        FC_layer_set = nn.Sequential(
            nn.Linear(1000, class_num)
        )
        Classifier_Output.append(FC_layer1)
        Classifier_Output.append(FC_layer2)
        Classifier_Output.append(FC_layer3)
        Classifier_Output.append(FC_layer_set)
        self.classifier = nn.Sequential(*Classifier_Output)

        # 是否需要初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        """前向传播"""
        ## 特征提取
        #print(x.shape)
        features = self.Feature_Extractor(x)
        #print(features.shape)
        ## 将提取到的特征铺展开来
        feature_trans = features.view(features.size(0), -1)
        #feature_trans = features.view(-1,256 * 6 * 6)
        ## 铺展开的特征放入分类器中,得出分类结果
        result = self.classifier(feature_trans)
        return result

    def _initialize_weights(self):
        # 遍历模型结构
        for m in self.modules():
            # 判断层结构类型
            if isinstance(m, nn.Conv2d):
                # 卷积层则用何凯明初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层则采用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# x = torch.rand(size=(8,3,224,224))
# model_classification = MyAlexNet(class_num=5)
# print(model_classification)
