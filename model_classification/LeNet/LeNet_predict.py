# LeNet，使用训练出来的模型做预测
import torch
import torchvision.transforms as transforms
from PIL import Image
from LeNet_Net import MyLeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = MyLeNet(3, 10)
    # 载入权重文件
    net.load_state_dict(torch.load('./save/Lenet.pth'))
    # 载入图像
    im = Image.open('plane.jpeg')
    # 基础预处理
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # 非迭代，进行进行结果预测
    with torch.no_grad():
        outputs = net(im)
        # predict = torch.max(outputs, dim=1)[1].numpy()
        predicts = torch.softmax(outputs, dim=1)
        print(f"预测分类指标:{predicts}")
        predict = torch.max(predicts, 1)[1]
        # 按照类别给出打分
        label = predicts.numpy()[0]
        for i in range(len(classes)):
            print("类别：{} \t 预测概率：{:.4%}".format(classes[i],label[i]))
    print(f"最终预测类别：{classes[int(predict)]}")


if __name__ == '__main__':
    main()