import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from EfficientNetV2_net import efficientnetv2_s as create_model
from EfficientNetV2_net import efficientnetv2_s, efficientnetv2_l, efficientnetv2_m
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']

# prdictor(device, img_path, type, model_weight_path)
def prdictor(device, img_path, type, model_weight_path):
    # 1.设备选择
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型类别
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = type
    # 2.数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # 3.加载待测试图片
    img_path = img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # 装载图片
    img = Image.open(img_path)
    # 展示图片
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 增加BatchSize维度
    img = torch.unsqueeze(img, dim=0)
    # 4.读入标签json文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 5.构建测试模型
    model = create_model(num_classes=5).to(device)
    if type == 's':
        model = efficientnetv2_s(num_classes=5).to(device)
    elif type == 'm':
        model = efficientnetv2_m(num_classes=5).to(device)
    elif type == 'l':
        model = efficientnetv2_l(num_classes=5).to(device)
    # 加载模型权重
    model_weight_path = "./weight/" + 'EfficientV2_' + type + '/' +  model_weight_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 使用模型评估，关闭各个层的参数更新
    model.eval()
    with torch.no_grad():
        # 预测类别
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "使用模型: EfficientNetV2_{}   预测类别: {}   概率: {:.4}".format(type, class_indict[str(predict_cla)],
                                                                      predict[predict_cla].numpy())
    plt.title(print_res)
    print(f"使用模型{type}, 得到当前图像{img_path}的预测结果如下：")
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    print()
    plt.show()


if __name__ == '__main__':
    prdictor(device="cuda:0", img_path="./sunflower1.jpeg", type="s", model_weight_path="model_classification-19.pth")
    prdictor(device="cuda:1", img_path="./tulip2.jpeg", type="s", model_weight_path="model_classification-19.pth")

    prdictor(device="cuda:0", img_path="./sunflower1.jpeg", type="m", model_weight_path="model_classification-19.pth")
    prdictor(device="cuda:1", img_path="./tulip2.jpeg", type="m", model_weight_path="model_classification-19.pth")

    prdictor(device="cuda:0", img_path="./sunflower1.jpeg", type="l", model_weight_path="model_classification-19.pth")
    prdictor(device="cuda:1", img_path="./tulip2.jpeg", type="l", model_weight_path="model_classification-19.pth")