from collections import namedtuple
import structlog
from torch import nn

log = structlog.get_logger()


ConvOptions = namedtuple('ConvOptions', [
    'out_channels',
    'kernel_size',
    'dropout',
    'stride',
    'padding_mode',
    'residual',
], defaults=[5, 0, 1, 'reflect', False])


class ConvNet(nn.Module):
    """
    A vanilla convolutional neural network.
    """
    def __init__(self, num_classes, width=64, height=64, channels=3,
                 conv_options=None, fc_sizes=None, **_hyperparams):
        super().__init__()
        self.conv_options = conv_options or [
            # TODO: add stride
            ConvOptions(64, 5, 0.9),
            ConvOptions(32, 3, 0.9),
            ConvOptions(32, 3),
            ConvOptions(32, 3),
        ]

        self.conv_layers = []
        for conv_option in self.conv_options:
            if conv_option.kernel_size%2 != 1:
                raise ValueError('Convolutional kernel must have odd size')
            padding = conv_option.kernel_size//2

            # Add convolutional layer
            conv_layer = [
                nn.Conv2d(
                    channels,
                    out_channels=conv_option.out_channels,
                    kernel_size=conv_option.kernel_size,
                    stride=conv_option.stride,
                    padding=padding,
                    padding_mode=conv_option.padding_mode,
                ),
                nn.BatchNorm2d(conv_option.out_channels),
            ]
            if conv_option.dropout > 0:
                conv_layer.append(nn.Dropout(conv_option.dropout, inplace=True))
            conv_layer.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
            self.conv_layers.append(conv_layer)
            log.debug('Added convolutional layer', **conv_option._asdict())

            # Update dimensions
            channels = conv_option.out_channels
            height = (height + 2*padding - conv_option.kernel_size)/conv_option.stride + 1
            width = (width + 2*padding - conv_option.kernel_size)/conv_option.stride + 1
            height, width = int(height/2), int(width/2)

        fc_sizes = [channels*height*width] + (fc_sizes or [200]) + [num_classes]
        fc_layers = [nn.Linear(*dims) for dims in zip(fc_sizes[:-1], fc_sizes[1:])]
        self.fc_layers = nn.Sequential(*fc_layers)
        log.debug('Added affine layers', sizes=fc_sizes)
        log.debug('Initialized CNN')

    def forward(self, x):
        for conv_option, conv_layer in zip(self.conv_options, self.conv_layers):
            for layer in conv_layer:
                # TODO: Downsample for nonzero stride
                x_preimage = x
                x = layer(x)
                if conv_option.residual:
                    x += x_preimage
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
