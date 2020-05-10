#!/usr/bin/env python3

import argparse
import datetime
import functools
import itertools
import logging.config
import pathlib
import random

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
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
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


def forward_pass(model, criterion, batch, k: int = 5):
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    _, top_k_predictions = torch.topk(outputs, k=k, dim=1)
    top_k_correct = top_k_predictions.eq(targets.unsqueeze(0).T).any(1)
    correct = top_k_predictions[:, 0].eq(targets)
    return loss, targets.size(0), correct.sum().item(), top_k_correct.sum().item()


def train_with_tuning(train_set, val_set, test_set, labels, options):
    model_module = MODELS.get(options.model)
    model = model_module(len(labels))
    try:
        log.debug('Initialized model')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr,
                                     weight_decay=options.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=options.lr_decay,
        )
        log.debug('Initialized loss and optimizer')

        loader = functools.partial(torch.utils.data.DataLoader, shuffle=True,
                                   pin_memory=True, num_workers=2)
        train_loader = loader(train_set, batch_size=options.batch_size)
        test_loader_factory = lambda: iter(loader(test_set, batch_size=5*options.batch_size))
        test_loader = test_loader_factory()
        forward = functools.partial(forward_pass, model, criterion)

        for epoch in range(options.max_epochs):
            train_total, correct_total, top_k_correct_total = 0, 0, 0
            train_losses = []
            for batch_num, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss, batch_size, correct, top_k_correct = forward(batch)
                train_losses.append(loss.item())
                train_total += batch_size
                correct_total += correct
                top_k_correct_total += top_k_correct
                loss.backward()
                optimizer.step()

                if (batch_num + 1)%options.print_every == 0:
                    try:
                        test_batch = next(test_loader)
                    except StopIteration:
                        test_loader = test_loader_factory()
                        test_batch = next(test_loader)
                    with torch.no_grad():
                        model.eval()
                        test_loss, test_total, correct, top_k_correct = forward(test_batch)
                        model.train()

                    log.debug('Training update',
                              batch_num=batch_num+1, epoch=epoch,
                              train_acc=round(correct_total/train_total, 4),
                              train_top_k_acc=round(top_k_correct_total/train_total, 4),
                              train_loss=round(np.mean(train_losses), 4),
                              test_acc=round(correct/test_total, 4),
                              test_top_k_acc=round(top_k_correct/test_total, 4),
                              test_loss=round(test_loss.item(), 4))
                    train_losses.clear()
                    train_total, correct_total, top_k_correct_total = 0, 0, 0

            torch.save({'net': model.state_dict()}, options.params)
            log.info('Epoch complete, saved model state')
            lr_scheduler.step()
    finally:
        torch.save({'net': model.state_dict()}, 'params/param-dump.pt')
        log.warn('Dumped parameters')


def main():
    options = parse_options()
    initialize_logging(options)
    try:
        log.debug('Started training', options=options.__dict__)
        train_set, val_set, test_set, labels = load_datasets(options)
        log.debug('Loaded datasets', labels=len(labels), num_train=len(train_set),
                  num_val=len(val_set), num_test=len(test_set))
        log.debug('CUDA status', is_available=torch.cuda.is_available())
        train_with_tuning(train_set, val_set, test_set, labels, options)
    except Exception as exc:
        log.critical('Received error', exc_info=exc)
    finally:
        log.debug('Stopped training')


if __name__ == '__main__':
    main()
