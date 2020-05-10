import functools

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


Conv1x1 = functools.partial(nn.Conv2d, kernel_size=1)
Conv3x3 = functools.partial(nn.Conv2d, kernel_size=3, padding=1, padding_mode='reflect')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            self.relu,
            Conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.layers(x) + x
        x = self.relu(x)
        return x


class ResidualNet(nn.Module):
    def __init__(self, num_classes, height: int = 64, width: int = 64,
                 in_channels: int = 3, affine_size: int = 2000):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3,
                                 padding_mode='reflect')
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.stage1 = nn.Sequential(
            nn.Dropout(0.3),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            nn.Dropout(0.3),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.affine = nn.Sequential(
            nn.Linear(64*(height//2)*(width//2), affine_size),
            nn.Linear(affine_size, num_classes),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.max_pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        return x


MODELS = {
    'resnet': ResidualNet,
}
