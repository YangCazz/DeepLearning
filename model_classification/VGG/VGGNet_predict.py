# AlexNet，使用训练出来的模型做预测
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from VGGNet_Net import vgg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']


def prdictor(device, img_path, model_type, model_weight_path):
    # 1.设备选择
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2.数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 3.加载待测试图片
    #img_path = "./tulip.jpg"
    #img_path = "./tulip1.jpg"
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
    model = vgg(model_name=model_type, num_classes=5).to(device)
    # model_classification = vgg(model_name=model_type, num_classes=5).to(device)
    # 加载模型权重
    # weights_path = "./save/vgg16Net.pth"
    # weights_path = "./save/vgg19Net.pth"
    assert os.path.exists(model_weight_path), "file: '{}' dose not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 使用模型评估，关闭各个层的参数更新
    model.eval()
    with torch.no_grad():
        # 预测类别，将结果输出到cpu
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "使用模型：{}   预测类别: {}   概率: {:.3}".format(model_type, class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("预测类别: {:10}   概率: {:.3%}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    prdictor(device="cuda:1", img_path="./tulip1.jpg",
             model_type="vgg16", model_weight_path="./save/vgg16Net.pth")
    print()
    prdictor(device = "cuda:0", img_path = "./tulip1.jpg",
             model_type = "vgg19", model_weight_path = "./save/vgg19Net.pth")
    # main()
