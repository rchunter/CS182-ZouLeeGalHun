import warnings
warnings.filterwarnings('ignore')

import argparse
import pathlib
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


class ConvNet(torch.nn.Module):
    """
    A vanilla convolutional neural net.

    Architecture:
        [conv-batchnorm-relu-maxpool] x M - [affine-relu] x (N-1) - affine
    """

    def __init__(self, num_classes, width=64, height=64, input_channels=3,
                 filter_options=None, affine_sizes=None):
        filter_options = filter_options or [
            {'out_channels': 64, 'kernel_size': 7, 'padding_mode': 'border'},
            {'out_channels': 32, 'kernel_size': 5, 'padding_mode': 'border'},
            {'out_channels': 16, 'kernel_size': 3, 'padding_mode': 'border'},
        ]

        filters = []
        for options in filter_options:
            output_channels = options['out_channels']
            kernel_size, stride = options.get('kernel_size', 3), options.get('stride', 1)
            padding = options.get('padding', kernel_size//2)
            filters.extend([
                nn.Conv2d(input_channels, **options),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            input_channels = output_channels
            # Dimension reduction due to convolutional layer
            height = (height + 2*padding - kernel_size)/stride + 1
            width = (width + 2*padding - kernel_size)/stride + 1
            # Dimension reduction due to pooling
            height, width = height//2, width//2
        self.filter_layers = nn.Sequential(*filters)

        affine_sizes = [input_channels*height*width] + (affine_sizes or [200]) + [num_classes]
        self.affine_layers = nn.Sequential(nn.Linear(*dims)
                                           for dims in zip(affine_sizes[:-1], affine_sizes[1:]))

    def forward(self, x):
        x = self.filter_layers(x)
        x = x.view(x.size(0), -1)
        x = self.affine_layers(x)
        return x


def main():
    options = parse_options()
    params = {
        'lr': [1e-4, 1e-3],
        # 'max_epochs': []
    }
    net = NeuralNetClassifier(
        ConvNet,
        max_epochs=20,
        criterion=torch.nn.CrossEntropyLoss,
        iterator_train__shuffle=True,
    )
    search = GridSearchCV(net, params, refit=False, cv=options.cv_folds, scoring='accuracy')
    search.fit()


if __name__ == '__main__':
    main()
