# 数据处理工具类
import os
import torch.utils.data as data
from PIL import Image

# 对VOC数据的处理
# root= ../../data/VOC
class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 1.获取VOC数据的路径
        # ../../data/VOC/VOCdevkit/VOC2012
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        # 2.读取文件路径
        # ../../data/VOC/VOCdevkit/VOC2012/JPEGImages
        image_dir = os.path.join(root, 'JPEGImages')

        # 3.读取掩码地址--实例分割标签
        # ../../data/VOC/VOCdevkit/VOC2012/SegmentationClass
        mask_dir = os.path.join(root, 'SegmentationClass')

        # 4.读取txt--数据
        # ../../data/VOC/VOCdevkit/VOC2012/ImageSets/Segmentation
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 4.1读取txt中指向的文件名
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        # 4.2拼接成完整文件, 保存进入images列表
        # 图像.jpg----掩码.png
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    # 映射函数
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # 1.打开图像
        img = Image.open(self.images[index]).convert('RGB')
        # 2.打开对应的掩码
        target = Image.open(self.masks[index])
        # 3.完成图像预处理
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # 计算数据集长度
    def __len__(self):
        return len(self.images)

    # 拼接一个batch的图像张量, 用于计算
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

# 序列拼接
# 见tester中的测试
# 函数效果：将所有图像拼接成首位相连的张量
def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # [C, H, W]
    # 计算batch shape = [图片数量, 图像通道数量, H, W]
    batch_shape = (len(images),) + max_size
    #
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    #
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)