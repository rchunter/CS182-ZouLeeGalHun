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

import numpy as np
import structlog
import torch
import torchvision
from torchvision import transforms
import yaml

from model import MODELS


log = structlog.get_logger()
cwd = pathlib.Path('.')


def parse_options():
    parser = argparse.ArgumentParser(description='Training script for image classifier')
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--log', type=str, default=None, help='path to log')
    parser.add_argument('--params', type=str, default=None, help='path to model params')
    parser.add_argument('--dataset', type=str, default=str(cwd/'data/tiny-imagenet-200'),
                        help='path to dataset')
    parser.add_argument('--print-every', type=int, default=200,
                        help='print every number of minibatches')
    parser.add_argument('--lr', type=float, default=2e-4, help='base learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay every epoch')
    parser.add_argument('--weight-decay', type=float, default=5e-3, help='optimizer weight decay')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--max-epochs', type=int, default=10, help='max training passes')
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
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'train'), data_transforms)
    val_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'val'), data_transforms)
    test_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'test'), data_transforms)
    labels = np.array([item.name for item in (cwd/options.dataset/'train').glob('*')])
    return train_set, val_set, test_set, labels


@dataclasses.dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.nn.Module
    criterion: torch.nn.Module = dataclasses.field(default_factory=torch.nn.CrossEntropyLoss)
    total: int = 0
    correct: int = 0
    top_correct: int = 0
    losses: typing.List[float] = dataclasses.field(default_factory=list)
    rounding: int = 4
    print_every: int = 200

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, k: int = 5):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.losses.append(loss.item())
        _, top_predictions = torch.topk(outputs, k, dim=1)
        self.top_correct += top_predictions.eq(targets.unsqueeze(0).T).any(1).sum().item()
        self.correct += top_predictions[:, 0].eq(targets).sum().item()
        self.total += targets.size(0)
        return loss

    def reset_statistics(self):
        self.total, self.correct, self.top_correct = 0, 0, 0
        self.losses.clear()

    @property
    def statistics(self):
        return {
            'accuracy': round(self.correct/self.total, self.rounding),
            'top_accuracy': round(self.top_correct/self.total, self.rounding),
            'loss': round(np.mean(self.losses), self.rounding),
        }

    def train_epoch(self, dataloader, train: bool = True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            for batch_num, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.forward(inputs, targets)
                if train:
                    loss.backward()
                    self.optimizer.step()

                if (batch_num + 1)%self.print_every == 0:
                    log.debug('Training update', batch_num=batch_num+1, **self.statistics)

    def train(self, max_epochs: int, train_loader, val_loader):
        val_accuracies, best_weights = [], copy.deepcopy(self.model.state_dict())
        try:
            for epoch in range(max_epochs):
                log.info('Training phase', epoch=epoch)
                self.train_epoch(train_loader)
                self.reset_statistics()

                log.info('Validation phase', epoch=epoch)
                self.train_epoch(val_loader, train=False)
                stats = self.statistics
                if stats['accuracy'] > np.max(val_accuracies + [-np.inf]):
                    best_weights = copy.deepcopy(self.model.state_dict())
                val_accuracies.append(stats['accuracy'])
                self.reset_statistics()
                self.lr_scheduler.step()
        finally:
            return best_weights


def train(train_set, val_set, test_set, labels, options):
    loader = functools.partial(torch.utils.data.DataLoader, shuffle=True,
                               pin_memory=True, num_workers=4,
                               batch_size=options.batch_size)
    train_loader, val_loader = loader(train_set), loader(val_set)

    model_module = MODELS.get(options.model)
    model = model_module(len(labels))
    model.to(Trainer.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr,
                                 weight_decay=options.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=options.lr_decay,
    )

    trainer = Trainer(model, optimizer, print_every=options.print_every)
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
