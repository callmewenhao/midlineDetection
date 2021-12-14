import torch
import numpy as np
from skimage import io
import torchvision.transforms as transforms
from model.resnet18 import ResNet18
from utils import show_landmarks
import matplotlib.pyplot as plt

def main():
    save_path = "output\eval_output"
    para_path = "output\parameters_30.pt"
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    for i in range(40):
        """ 暂时使用训练集中的图片测试下 """
        img_path = f"data/train/{i:0>3}.png"
        # print(img_path)
        # continue
        image = io.imread(img_path)
        image_ = image
        image = data_transform(image).unsqueeze(0)  # 增加一维

        model = ResNet18()
        model.eval()
        ckpt = torch.load(para_path)
        model.load_state_dict(ckpt)
        out = model(image)
        out = out.squeeze(0).detach().numpy().T
        # print(out.shape, out.dtype)
        out[:, 0] *= 94
        out[:, 1] *= 60
        # print(out)
        plt.figure("test")
        show_landmarks(image_, out)
        plt.savefig(save_path+f'\pred_output_{i:0>3}.png')
        print("save " + "save_path"+f'\pred_output_{i:0>3}.png')
        plt.close()
        # plt.show()


if __name__ == "__main__":
    main()
