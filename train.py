import yaml
from yaml import CLoader as Loader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model.resnet18 import ResNet18
from MyDataset import LineLandmarksDataset
from utils import plot_loss, get_logger


# TODO target acc: 93%+ 训练了50个epoch 达到了90+%

losses = {'train': [], 'test': []}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, criterion, optim, epoch, total_epoch, logger):
    for i, (images, labels) in enumerate(dataloader):  # dataloader 代表了整个数据集 i 代表一个 batch
        images = images.to(device)  # 数据放到GPU上
        labels = labels.to(device)
        labels[..., 0] /= 94  # x 
        labels[..., 1] /= 60  # y
        labels = labels.transpose(1, -1)

        out = model(images)
        # print(out.shape)  # out 的大小：128X10 ，存放概率
        loss = criterion(out, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses['train'].append(loss.item())
        # print(f"epoch:{epoch}/{total_epoch}, {i+1}th , loss: {loss.item()}, acc: {acc}")
        logger.info(f"epoch:{epoch}/{total_epoch}, {i+1}th , loss: {loss.item()}")


def main():
    f = open('parameters.yml', encoding='utf-8')
    para = yaml.load(f, Loader)
    total_epoch = para['total_epochs']
    batch_size = para['batch_size']
    learning_rate = float(para['lr'])
    model_name = para['model']
    output_path = para['output_path']

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    Line_dataset = LineLandmarksDataset(
        csv_file='data/train/label.txt', 
        root_dir='data/train/',
        transform=data_transform)

    train_dataloader = DataLoader(Line_dataset, batch_size=batch_size, shuffle=True)

    # 查看训练图片
    # plot_image(train_dataloader, index=0)

    # 变量定义
    if model_name == 'resnet18':
        model = ResNet18()
    elif model_name == 'resnet50':
        # model = ResNet50()
        print("resnet50 model is not defined!")
        pass
    else:
        print('Model is not defined!')
    # print(model)  # 展示模型信息
    model = model.to(device)  # 把模型丢到 GPU
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 是一种优化器

    # 训练
    logger = get_logger('train.log')
    logger.info('start training!')

    for epoch in range(1, total_epoch+1):  # 1 to total_epoch+1
        train_one_epoch(model, train_dataloader, criterion, optim, epoch, total_epoch, logger)
    # 打印 loss
    torch.save(model.state_dict(), f"output\parameters_{total_epoch}.pt")
    plot_loss(losses, save_path=output_path+f'\losses_{total_epoch}.png')

    logger.info('finish training!')

if __name__ == "__main__":
    main()