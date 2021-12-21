import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import logging
import yaml
from yaml import CLoader as Loader


def display_landmarks(index=0,label_path="data/train\label.txt"):
    landmarks_frame = pd.read_csv(label_path, header=None)  # txt 无表头
    img_name = landmarks_frame.iloc[index, 0]
    landmarks = landmarks_frame.iloc[index, 1:]
    landmarks = np.asarray(landmarks)
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print(landmarks_frame)
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('All Landmarks: {}'.format(landmarks[:]))

    return landmarks


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    # plt.figure("image")
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()


# def showImagesInDataset(dataset, nums=4):
#     plt.figure()
#     for i in range(len(dataset)):
#         sample = dataset[i]
#         plt.subplot(2, 2, i + 1)
#         show_landmarks(torch.squeeze(sample[0]), torch.squeeze(sample[1]))
#         # sample[0] [c, h, w] sample[1] [1, 5, 2] ?
#         if i == 3:
#             plt.show()
#             print("nums:", i+1, sample[0].shape, sample[1].shape)
#             break


def plot_loss(losses, mode='train', save_path=None):
    """ 
    mode='train' or 'test' or 'both'
    """
    if mode =='train':
        plt.xlabel("iter")
        plt.ylabel("train loss")
        plt.plot(losses["train"], 'r')
        # plt.text(20.0, 2.0, 'lr = '+str(learning_rate),  fontsize=15)    #文本中注释
        plt.title('losses in training!')
        # plt.show()
        # plt.savefig("cifar10_{}_loss.png".format(epochs))
    elif mode =='test':
        plt.xlabel("iter")
        plt.ylabel("test loss")
        plt.plot(losses["test"], 'r')
        plt.title('losses in testing!')
    else:
        plt.xlabel("iter")
        plt.ylabel("train & test loss")
        plt.plot(losses["train"], 'r', label='loss in train')
        plt.plot(losses["test"], 'g', label='loss in test')
        plt.title('losses in training & testing!')
        plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        # "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"  
        # [2021-12-09 16:45:07,913][train_using_single_GPU.py][line:74][INFO] start training!
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def main():
    image_path = "data/train/000.png"
    landmarks = display_landmarks(index=0,label_path="data/train\label.txt")
    plt.figure("image")
    show_landmarks(Image.open(image_path), landmarks)
    plt.show()


if __name__ == "__main__":
    main()







