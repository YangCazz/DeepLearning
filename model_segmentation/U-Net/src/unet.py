from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建卷积块
# conv + BN  +ReLU
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 构建下采样块
# 卷积 + Maxpool
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

# 构建上采样块
# upconv + 卷积
class Up(nn.Module):
    # 默认使用双线插值
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 上采样模块中存在两个数据块，路径处理的数据+跳连接过来的数据
        # 1. 上采样数据1
        # 为了避免两块数据的大小不一样(无法被整除)，采用padding方法来处理
        x1 = self.up(x1)
        # [N, C, H, W]
        # 计算[H,W]上两块的差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # 左、右、上、下
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 在channel维度上拼接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 3->64
        self.in_conv = DoubleConv(in_channels, base_c)
        # 64->128
        self.down1 = Down(base_c, base_c * 2)
        # 128->256
        self.down2 = Down(base_c * 2, base_c * 4)
        # 256->512
        self.down3 = Down(base_c * 4, base_c * 8)
        # 双线性插值/转置卷积
        # 512->512,
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        # 512*2=1024->512/2=256
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        # 256*2=512->256/2=128
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        # 128*2=256->128/2=64
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        # 64*2=128->64
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # 64->2
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
