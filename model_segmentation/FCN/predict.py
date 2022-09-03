# FCN-ResNet模型的预测
import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50

# 计时
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    aux = False  # inference time not need aux_classifier
    classes = 20
    weights_path = "./save_weights/model_7.pth"
    img_path = "./test_Pic/test2.jpeg"
    # palette对应标签文件的调色板
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # 使用设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 构建模型
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # 模型预测时, 不使用捷径分支
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # 加载模型权重
    model.load_state_dict(weights_dict)
    model.to(device)

    # 加载图片
    original_img = Image.open(img_path)

    # 完成图像预处理
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # 扩展图像维度[N,C,H,W]
    img = torch.unsqueeze(img, dim=0)
    # 进入验证模式
    model.eval()
    with torch.no_grad():
        # 模型初始化
        # 取出[H,W]
        img_height, img_width = img.shape[-2:]
        # 构建一个mask
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        # 开始计时
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        # 将预测的输出分割导出
        prediction = output['out'].argmax(1).squeeze(0)
        # 从GPU回存CPU
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # Image.fromarray()实现array到image的转换
        mask = Image.fromarray(prediction)
        # 自定义各个类别的颜色
        mask.putpalette(pallette)
        mask.save("./test_Result/test_result2.png")


if __name__ == '__main__':
    main()