#!/usr/bin/env python3

import argparse
import copy
import dataclasses
import datetime
import functools
import itertools
import logging.config
import pathlib
import random
import typing
import io

from PIL import Image
import numpy as np
import structlog
import torch
import torchvision
from torchvision import transforms
import yaml

import model as visionmodel


log = structlog.get_logger()
cwd = pathlib.Path('.')

def JPEGCompression(image):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=75, optimize=True)
    output.seek(0)
    return Image.open(output)

spatial_transforms = transforms.Compose([
    transforms.RandomChoice([
        transforms.Compose([
            # Ensure that the rotated image has its corners filled.
            transforms.Pad(60, padding_mode='reflect'),
            transforms.RandomAffine(90, translate=(0.1, 0.1)),
            transforms.CenterCrop(224),
        ]),
        transforms.RandomResizedCrop(224, scale=(0.5, 1)),
    ]),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
])
jpeg_transform = transforms.Lambda(JPEGCompression)
color_transform = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.05)
noise_transform = transforms.Lambda(lambda image, std=0.1: torch.clamp(image + std*torch.randn_like(image), 0, 1))
normalize_stds = [0.229, 0.224, 0.225]
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], normalize_stds)


def parse_options():
    parser = argparse.ArgumentParser(description='Training script for image classifier')
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--log', type=str, default=None, help='path to log')
    parser.add_argument('--params', type=str, default=None, help='path to model params')
    parser.add_argument('--dataset', type=str, default=str(cwd/'data/tiny-imagenet-200'),
                        help='path to dataset')
    parser.add_argument('--print-every', type=int, default=500,
                        help='print every number of minibatches')
    parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay every epoch')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='optimizer weight decay')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--max-epochs', type=int, default=10, help='max training passes')
    parser.add_argument('--jpeg', type=bool, default=False, help='use jpeg compression preprocessing')
    parser.add_argument('--crop-ensemble', type=bool, default=False, help='use crop ensemble preprocessing')
    options = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    options.train = f'{options.model}-{now}'
    options.params = options.params or str(cwd/f'params/{options.train}.pt')
    return options


def initialize_logging(options):
    """
    Initialize logging for the training script.

    We use the third-party `structlog` package to automatically write log
    records as machine-readable files. This ensures we have a record of every
    experiment.
    """
    timestamper = structlog.processors.TimeStamper(fmt='%Y-%m-%d %H:%M:%S')
    filename = options.log if options.log else str(cwd/f'logs/train-{options.train}.log')

    pre_chain = [structlog.stdlib.add_log_level, timestamper]
    handlers = {
        'screen': {'level': 'DEBUG', 'class': 'logging.StreamHandler', 'formatter': 'color'},
        'file': {'level': 'DEBUG', 'class': 'logging.FileHandler',
                 'formatter': 'plain', 'filename': filename, 'mode': 'a+'},
    }

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'plain': {'()': structlog.stdlib.ProcessorFormatter,
                      'processor': structlog.processors.JSONRenderer(),
                      'foreign_pre_chain': pre_chain},
            'color': {'()': structlog.stdlib.ProcessorFormatter,
                      'processor': structlog.dev.ConsoleRenderer(colors=True),
                      'foreign_pre_chain': pre_chain},
        },
        'handlers': handlers,
        'loggers': {'': {'handlers': handlers.keys(), 'level': 'DEBUG', 'propagate': True}},
    })
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def load_datasets(options):
    if options.model == 'denoise':
        train_transforms = [
            transforms.Resize(256, interpolation=Image.LANCZOS),
            transforms.RandomChoice([transforms.CenterCrop(224), spatial_transforms]),
            transforms.ToTensor() 
        ]
        test_transforms = train_transforms
    else:
        train_transforms = [transforms.Resize(224)]
        test_transforms = [transforms.Resize(256)]
        if options.jpeg:
            train_transforms.append(jpeg_transform)
            test_transforms.append(jpeg_transform)
        train_transforms.append(color_transform)
        if options.crop_ensemble:
            train_transforms.append(transforms.Compose([
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
            ]))
            test_transforms.append(transforms.Compose([
                transforms.FiveCrop(224),
                transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
            ]))
        else:
            test_transforms.append(transforms.CenterCrop(224))
        train_transforms.extend([
            transforms.ToTensor(),
            noise_transform,
            transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.5, 2)),
            normalize_transform,
        ])
        test_transforms += [
            transforms.ToTensor(),
            normalize_transform,
        ]

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    train_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'train'), train_transforms)
    val_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'val'), test_transforms)
    test_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'test'), test_transforms)
    labels = np.array([item.name for item in (cwd/options.dataset/'train').glob('*')])
    return train_set, val_set, test_set, labels


class Trainer:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rounding: int = 4

    def __init__(self, model, criterion, optimizer, lr_scheduler=None, print_every: int = 200, options=None):
        self.options = options
        self.model, self.criterion = model, criterion
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.total, self.losses = 0, []
        self.print_every = print_every

    def forward(self, inputs, targets):

        if self.options is not None and self.options.crop_ensemble:
            bs, ncrops, c, h, w = inputs.size()
            outputs = self.model(input.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.losses.append(loss.item())
        return outputs, loss

    @property
    def statistics(self):
        return {'loss': round(np.mean(self.losses), self.rounding)}

    def reset_statistics(self):
        self.total = 0
        self.losses.clear()

    def set_model_mode(self, train: bool = True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def transform_batch(self, inputs, targets):
        return inputs.to(self.device), targets.to(self.device)

    def train_epoch(self, dataloader, train: bool = True):
        self.set_model_mode(train=train)
        with torch.set_grad_enabled(train):
            for batch_num, (inputs, targets) in enumerate(dataloader):
                inputs, targets = self.transform_batch(inputs, targets)
                self.optimizer.zero_grad()
                outputs, loss = self.forward(inputs, targets)
                if train:
                    loss.backward()
                    self.optimizer.step()
                if (batch_num + 1)%self.print_every == 0:
                    log.debug('Update', batch_num=batch_num+1, **self.statistics)
                    if train:
                        self.reset_statistics()

    def train(self, max_epochs: int, train_loader, val_loader, metric: str = 'loss'):
        metric_history, best_weights = [], copy.deepcopy(self.model.state_dict())
        try:
            for epoch in range(max_epochs):
                log.info('Training phase', epoch=epoch)
                self.train_epoch(train_loader)

                log.info('Validation phase', epoch=epoch)
                self.train_epoch(val_loader, train=False)

                stats = self.statistics
                self.reset_statistics()
                if stats[metric] > np.max(metric_history + [-np.inf]):
                    best_weights = copy.deepcopy(self.model.state_dict())
                    log.info('Updated best weights', **stats, metric=metric)
                metric_history.append(stats[metric])

                if self.lr_scheduler:
                    self.lr_scheduler.step()
        except Exception as exc:
            log.error('Encountered error during training', exc_info=exc)
        finally:
            return best_weights


class ClassificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correct, self.top_correct = 0, 0

    def forward(self, inputs, targets, k: int = 5):
        outputs, loss = super().forward(inputs, targets)
        _, top_predictions = torch.topk(outputs, k, dim=1)
        self.top_correct += top_predictions.eq(targets.unsqueeze(0).T).any(1).sum().item()
        self.correct += top_predictions[:, 0].eq(targets).sum().item()
        self.total += targets.size(0)
        return outputs, loss

    @property
    def statistics(self):
        return {
            **super().statistics,
            'accuracy': round(self.correct/self.total, self.rounding),
            'top_accuracy': round(self.top_correct/self.total, self.rounding),
        }

    def reset_statistics(self):
        super().reset_statistics()
        self.correct, self.top_correct = 0, 0

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs, metric='accuracy')


class DenoiseNetTrainer(Trainer):
    input_transforms = transforms.Compose([
        transforms.ToPILImage(),
        color_transform,
        transforms.ToTensor(),
        noise_transform,
    ])

    @property
    def statistics(self, height=224, width=224):
        stats = super().statistics
        return {**stats, 'psnr': round(10*np.log10(3*height*width/stats['loss']), self.rounding)}

    def transform_batch(self, inputs, _targets):
        targets = inputs.to(self.device)
        transformed_inputs = torch.empty_like(inputs)
        for i in range(inputs.size(0)):
            transformed_inputs[i] = self.input_transforms(inputs[i])
        transformed_inputs = transformed_inputs.to(self.device)
        return transformed_inputs, targets


def train(train_set, val_set, test_set, labels, options, trainer_cls=ClassificationTrainer):
    loader = functools.partial(torch.utils.data.DataLoader, shuffle=True,
                               pin_memory=True, num_workers=8,
                               batch_size=options.batch_size)
    train_loader, val_loader = loader(train_set), loader(val_set)

    if options.model == 'denoise':
        model, trainer_cls = visionmodel.DenoiseNet(), DenoiseNetTrainer
        params = model.parameters()
        criterion = torch.nn.MSELoss()
    elif options.model == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        trainer_cls = ClassificationTrainer
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(model.fc.in_features, 4000),
            torch.nn.Linear(4000, len(labels)),
        )
        params = model.fc.parameters()
        criterion = torch.nn.CrossEntropyLoss()
    elif options.model == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        trainer_cls = ClassificationTrainer
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(model.last_channel, len(labels))
        )
        params = model.parameters()
        criterion = torch.nn.CrossEntropyLoss()
    
    model.to(Trainer.device)
    optimizer = torch.optim.Adam(params, lr=options.lr,
                                 weight_decay=options.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                          gamma=options.lr_decay)

    trainer = trainer_cls(model, criterion, optimizer, print_every=options.print_every,
                          lr_scheduler=lr_scheduler, options=options)
    best_weights = trainer.train(options.max_epochs, train_loader, val_loader)
    torch.save({'net': best_weights}, options.params)
    log.info('Saved weights')


def main():
    options = parse_options()
    initialize_logging(options)
    try:
        log.debug('Started training', options=options.__dict__)
        train_set, val_set, test_set, labels = load_datasets(options)
        log.debug('Loaded datasets', labels=len(labels), num_train=len(train_set),
                  num_val=len(val_set), num_test=len(test_set))
        log.debug('CUDA status', is_available=torch.cuda.is_available())
        train(train_set, val_set, test_set, labels, options)
    except Exception as exc:
        log.critical('Received error', exc_info=exc)
    finally:
        log.debug('Stopped training')


if __name__ == '__main__':
    main()
