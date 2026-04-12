"""Small CNN architecture for the deep-learning baseline.

Three convolutional blocks (3 -> 16 -> 32 -> 64 channels, each with a 3x3
kernel, ReLU activation, and 2x2 max pooling), followed by a flatten layer,
a 64-unit fully connected layer with dropout, and a 2-unit output layer.
About 1M trainable parameters for the default IMG_SIZE = 128.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize as sk_resize

from .config import IMG_SIZE, CNN_DROPOUT


class CellImageDataset(Dataset):
    """Loads cell images on demand and converts them to CHW float tensors in [0, 1]."""

    def __init__(self, paths, labels, img_size=IMG_SIZE):
        self.paths = paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = imread(self.paths[idx])
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = sk_resize(
            img, (self.img_size, self.img_size),
            preserve_range=True,
            anti_aliasing=True,
        )
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        x = torch.from_numpy(img)
        y = int(self.labels[idx])
        return x, y


class SmallCNN(nn.Module):
    """Three conv blocks plus a small fully connected head."""

    def __init__(self, num_classes=2, img_size=IMG_SIZE, dropout=CNN_DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * (img_size // 8) * (img_size // 8), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
