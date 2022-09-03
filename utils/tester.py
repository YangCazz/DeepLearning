import torch
import random
# 设置随机种子
random.seed(0)
torch.manual_seed(0)
# 数据集
images = torch.rand(size=(8, 3, 224, 224))
# 单张图像
image = torch.rand(size=(3, 224, 224))
# 获取图像的最大三维
max_size = image.shape
# 拼接shape
batch_shape = (len(images),) + max_size
print(f"原shape：{(len(images),)}，{max_size}")
print(f"新shape：{batch_shape}")

# 构建填充张量, 默认填充1
batched_imgs = images[0].new(*batch_shape)
print(f"构建新的张量：规格为{batched_imgs.shape}")
print(batched_imgs)
print("填充数值2")
batched_imgs = batched_imgs.fill_(0)
print(batched_imgs)
# 逐一对图像进行处理
# 将张量的[H, W]用图像填充
# images=[3, 224, 224]
# batched_imgs=[8, 3, 224, 224]
# [,H,W]
for img, pad_img in zip(images, batched_imgs):
    pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    print(img)
    print(pad_img)
print("完成处理")
print(batched_imgs.shape)

