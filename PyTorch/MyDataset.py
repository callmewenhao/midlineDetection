import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils import show_landmarks, showImagesInDataset

class LineLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        """ 
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, header=None)  # txt文件
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float32').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        # print(image.dtype, landmarks.dtype)
        return image, torch.from_numpy(landmarks)


def main():
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    Line_dataset = LineLandmarksDataset(
        csv_file='data/train/label.txt', 
        root_dir='data/train/',
        transform=data_transform)

    showImagesInDataset(Line_dataset)

    train_dataloader = DataLoader(Line_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(train_dataloader))
    print(images.shape)
    print(labels.shape)
    labels[..., 0] /= 94
    labels[..., 1] /= 60
    print(labels)
    labels = labels.transpose(1, -1)
    print(labels.shape)


if __name__ == '__main__':
    main()
