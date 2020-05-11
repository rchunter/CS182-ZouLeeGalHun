import functools
from typing import List, Optional

import numpy as np
import torch
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from torch import nn


Conv3x3 = functools.partial(nn.Conv2d, kernel_size=3, padding=1, padding_mode='reflect')
Conv5x5 = functools.partial(nn.Conv2d, kernel_size=5, padding=2, padding_mode='reflect')
Conv7x7 = functools.partial(nn.Conv2d, kernel_size=7, padding=3, padding_mode='reflect')


class DenoiseBlock(nn.Module):
    def __init__(self, conv_block: nn.Conv2d, in_channels: int, out_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            conv_block(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            conv_block(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x) + x


class DenoiseNet(nn.Module):
    def __init__(self, height: int = 64, width: int = 64, channels: int = 3,
                 filters: int = 64):
        self.layers = nn.Sequential(
            DenoiseBlock(Conv7x7, channels, filters),
            DenoiseBlock(Conv5x5, filters, filters),
            DenoiseBlock(Conv5x5, filters, filters),
            DenoiseBlock(Conv3x3, filters, channels),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualNet(nn.Module):
    def __init__(self, num_classes, height: int = 64, width: int = 64,
                 in_channels: int = 3, affine_size: int = 4000):
        super().__init__()
        self.base_res_net = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        return self.base_res_net(x)
