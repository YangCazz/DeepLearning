from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101

# 提取中间层
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    # 声明
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
    # 词用结构遍历的方法来提取子层
    # model.named_children()方法来提取模型的第一层分支结构和中间模块
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # 判断集合的所有元素是否都包含在指定集合中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        # 保存初始参数
        orig_return_layers = return_layers
        # 将结构拆解成字典
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        # return_layer保存着我们需要提取的模块参数，其它的应当删除
        layers = OrderedDict()
        for name, module in model.named_children():
            # 根据结构来重构目录
            layers[name] = module
            # 每提取一个层就从待需求字典中删除
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        # 完成提取后, 还原提取模块
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        # 根据提取模型的结构, 来对应其输出
        for name, module in self.items():
            # 计算得出张量结果
            x = module(x)
            if name in self.return_layers:
                # 提取目标列表
                out_name = self.return_layers[name]
                # 输出层为提取结构的输出层, 存储张量结果进入字典
                out[out_name] = x

        # 返回目标结构 Dict[struct:output]
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    # 声明是带捷径的结构
    __constants__ = ['aux_classifier']
    # 提取结构的基本参数：backbone, 分类器, 捷径分支
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # [N,C,H,W], x.shape[-2:]表示提取输出参数的H,W
        input_shape = x.shape[-2:]
        # features 实际上是一个tensor列表, 存储了结构
        features = self.backbone(x)


        result = OrderedDict()
        # 提取out层的输出张量
        x = features["out"]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        # 输出插值结果
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        # 将结果存储到result字典中, 字段为[result: x]
        result["out"] = x

        # 计算捷径分支的结果
        if self.aux_classifier is not None:
            # 提取旁支的结果
            x = features["aux"]
            # 进行旁支的计算
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            # 和主分支一样计算结果
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            # 将结果存储下来
            result["aux"] = x

        # 输出结果result, 方便后续计算
        # [out: x, aux: x]
        return result

# 构建FCN Head
# 用于图像的重构
# conv2D 3*3 s1 p1 + BN + ReLU
# DropOut
# conv2D 1*1 s1
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

# 架构FCN—ResNet50结构
def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'

    # 这里对应于Layer的 2 3 4
    # 只有3和4是使用膨胀卷积的r=[1,2] [2,4]
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    # 使用官方的ResNet50预训练权重
    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    # 主干的末端输出维度为：60*60*2048
    # 捷径分支输出的维度为：60*60*1024
    out_inplanes = 2048
    aux_inplanes = 1024

    # 设置字典量, 用于提取输出
    # 捷径分支来自Layer3, result结果来自Layer4
    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'

    # 返回ResNet结构的两个结构
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #  默认旁支分类器是关闭的
    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)
    # 构建主干分类器
    classifier = FCNHead(out_inplanes, num_classes)
    # FCN返回result[aux:x ; out:x]
    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model