import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from MobileNet_Net_V2 import MobileNetV2
from MobileNet_Net_V3 import mobilenet_v3_small, mobilenet_v3_large
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']


def prdictor(device,type, img_path, model_weight_path):
    # 1.设备选择
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2.数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 3.加载待测试图片
    # img_path = "./tulip.jpg"
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
    if type == "MobileNetV2":
        model = MobileNetV2(num_classes=5).to(device)
    elif type == "MobileNetV3_small":
        model = mobilenet_v3_small(num_classes=5).to(device)
    elif type == "MobileNetV3_large":
        model = mobilenet_v3_large(num_classes=5).to(device)
    # model_classification = MobileNetV2(num_classes=5).to(device)
    # 加载模型权重
    # model_weight_path = model_weight_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 使用模型评估，关闭各个层的参数更新
    model.eval()
    with torch.no_grad():
        # 预测类别，将结果输出到cpu
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "使用模型: {}   预测类别: {}   概率: {:.4}".format(type, class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
    plt.title(print_res)
    print(f"使用模型MobileNet_V2, 得到当前图像{img_path}的预测结果如下：")
    for i in range(len(predict)):
        print("预测类别: {:10}   概率: {:.3%}".format(class_indict[str(i)], predict[i].numpy()))
    print()
    plt.show()


if __name__ == '__main__':
    prdictor(device="cuda:0", type="MobileNetV2", img_path="./tulip1.jpg", model_weight_path="./save/MobileNet_V2.pth")
    prdictor(device="cuda:1", type="MobileNetV2", img_path="./tulip.jpg", model_weight_path="./save/MobileNet_V2.pth")

    prdictor(device="cuda:0", type="MobileNetV3_small",
             img_path="./tulip1.jpg", model_weight_path="./save/MobileNet_V3_small.pth")
    prdictor(device="cuda:1", type="MobileNetV3_small",
             img_path="./tulip.jpg", model_weight_path="./save/MobileNet_V3_small.pth")

    prdictor(device="cuda:0", type="MobileNetV3_large",
             img_path="./tulip1.jpg", model_weight_path="./save/MobileNet_V3_large.pth")
    prdictor(device="cuda:1", type="MobileNetV3_large",
             img_path="./tulip.jpg", model_weight_path="./save/MobileNet_V3_large.pth")