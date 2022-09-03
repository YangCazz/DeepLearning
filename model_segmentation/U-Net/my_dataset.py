import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
# mask--对应感兴趣的区域(白色)
# manual--标注好的真实标签区域
# image--原图

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        # 载入training目录下的数据-指向训练集或者测试集
        self.flag = "training" if train else "test"
        # ../../data/DRIVE/training
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        # 使用图像预处理
        self.transforms = transforms
        # 读入图像列表.tif文件
        # ../../data/DRIVE/training/images
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # 读入图像标签文件
        # ../../data/DRIVE/training/1st_manual/xx_manual1.gif
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # 校验文件
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        # 读入图像掩码文件
        # ../../data/DRIVE/training/mask/xx_training_manual1.gif
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # 校验文件
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    # 数据映射
    def __getitem__(self, idx):
        # 图像读入, 转化为RGB
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 图像对应的标签打开, 转化为灰度图
        manual = Image.open(self.manual[idx]).convert('L')
        # 标签归一化处理
        manual = np.array(manual) / 255
        # 图像对应掩码打开, 转化为灰度图
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 掩码反选，用作图像框选
        roi_mask = 255 - np.array(roi_mask)
        # np.clip()为图像添加随机高斯噪声
        # 框选出图像的区域, 添加随机高斯噪声
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

# 数据拼接
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

