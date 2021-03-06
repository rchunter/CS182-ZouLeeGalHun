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
    def __init__(self, conv_block: nn.Conv2d, channels: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            conv_block(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            conv_block(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x) + x


class DenoiseNet(nn.Module):
    def __init__(self, channels: int = 3, filters: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            Conv7x7(channels, filters),
            DenoiseBlock(Conv5x5, filters, dropout=0.4),
            DenoiseBlock(Conv5x5, filters, dropout=0.4),
            Conv3x3(filters, channels),
        )

    def forward(self, x):
        return self.layers(x)
