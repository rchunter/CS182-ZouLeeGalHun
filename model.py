import functools

import numpy as np
import torch
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from torch import nn


class ResidualNet(nn.Module):
    def __init__(self, num_classes, height: int = 64, width: int = 64,
                 in_channels: int = 3, affine_size: int = 4000):
        super().__init__()
        self.base_res_net = torchvision.models.resnet34(pretrained=True)

    def forward(self, x):
        return self.base_res_net(x)


MODELS = {
    'resnet': ResidualNet,
}
