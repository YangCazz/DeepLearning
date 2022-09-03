from PIL import Image
import torch
from torch.utils.data import Dataset

# 数据集定义
class MyDataSet(Dataset):
    """自定义数据集"""

    # 子图数据路径列表, 图像类别列表, 数据处理方式
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    # 计算数据组中数据的数量
    def __len__(self):
        return len(self.images_path)

    # 获取数据集中分类列表的子图像具体路径
    def __getitem__(self, item):
        # 将这张图片加载出来
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        # 判别图像的类型
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        # 取出图像对应的标签
        label = self.images_class[item]

        # 对图像做具体处理
        if self.transform is not None:
            img = self.transform(img)

        # 返回成组化数据[图像, 标签]
        return img, label

    # 静态实现方法, 可以在不创建类实例的情况下调用方法
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        # torch.stack沿一个新维度对输入张量序列进行连接
        # 将这些成对的图像张量组拼接起来
        # 输入张量信息：
        # a=[i][j]
        # b=[i][j]
        # 输出张量信息：
        # c[0][i][j] = a[i][j]
        # c[1][i][j] = b[i][j]
        images = torch.stack(images, dim=0)
        # torch.as_tensor为data生成tensor
        labels = torch.as_tensor(labels)
        return images, labels