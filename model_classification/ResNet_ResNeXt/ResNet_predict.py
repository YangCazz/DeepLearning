# ResNet的模型验证
# ResNeXt的模型验证
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from  ResNet_Net import resnet18, resnet34, resnet50,  resnet101, resnet152
from  ResNet_Net import resnext50_32x4d, resnext101_32x8d
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']

# prdictor(device, img_path, aux_logits, model_weight_path)
def prdictor(device, img_path, type,model_weight_path):
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
    if type=='resnet18':
        model = resnet18(num_classes=5).to(device)
    elif type=='resnet34':
        model = resnet34(num_classes=5).to(device)
    elif type=='resnet50':
        model = resnet50(num_classes=5).to(device)
    elif type=='resnet101':
        model = resnet101(num_classes=5).to(device)
    elif type=='resnet152':
        model = resnet152(num_classes=5).to(device)
    elif type=='resnext50_32x4d':
        model = resnext50_32x4d(num_classes=5).to(device)
    elif type=='resnext101_32x8d':
        model = resnext101_32x8d(num_classes=5).to(device)

    # 加载模型权重
    weights_path = model_weight_path
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

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
    print(f"使用模型{type}, 得到当前图像{img_path}的预测结果如下：")
    for i in range(len(predict)):
        print("预测类别: {:10}   概率: {:.3%}".format(class_indict[str(i)], predict[i].numpy()))
    print()
    plt.show()


if __name__ == '__main__':
    prdictor(device="cuda:0", img_path="./tulip1.jpg", type='resnet18', model_weight_path="./save/ResNet18.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg", type='resnet18', model_weight_path="./save/ResNet18.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg", type='resnet34', model_weight_path="./save/ResNet34.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg", type='resnet34', model_weight_path="./save/ResNet34.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg", type='resnet50', model_weight_path="./save/ResNet50.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg", type='resnet50', model_weight_path="./save/ResNet50.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg", type='resnet101', model_weight_path="./save/ResNet101.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg", type='resnet101', model_weight_path="./save/ResNet101.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg", type='resnet152', model_weight_path="./save/ResNet152.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg", type='resnet152', model_weight_path="./save/ResNet152.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg",
             type='resnext50_32x4d', model_weight_path="./save/ResNeXt50_32x4d.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg",
             type='resnext50_32x4d', model_weight_path="./save/ResNeXt50_32x4d.pth")

    prdictor(device="cuda:0", img_path="./tulip1.jpg",
             type='resnext101_32x8d', model_weight_path="./save/ResNeXt101_32x8d.pth")
    prdictor(device="cuda:1", img_path="./tulip.jpg",
             type='resnext101_32x8d', model_weight_path="./save/ResNeXt101_32x8d.pth")