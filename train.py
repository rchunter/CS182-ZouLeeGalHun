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
import torchvision.transforms as transforms
import yaml

import model as visionmodel

log = structlog.get_logger()
cwd = pathlib.Path('.')


def parse_options():
    parser = argparse.ArgumentParser(description='Training script for image classifier')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--log', type=str, default=None, help='path to log')
    parser.add_argument('--params', type=str, default=None, help='path to model params')
    parser.add_argument('--dataset', type=str, default=str(cwd/'data/tiny-imagenet-200'),
                        help='path to dataset')
    parser.add_argument('--hyperparams', type=str, default=str(cwd/'config/default.yaml'),
                        help='path to hyperparameter config')
    parser.add_argument('--print-every', type=int, default=100,
                        help='print every number of minibatches')
    parser.add_argument('--search', type=int, default=None,
                        help='number of random hyperparameter combinations to try '
                             '(defaults to grid search)')
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
    train_set = torchvision.datasets.ImageFolder(cwd/options.dataset/'train', data_transforms)
    val_set = torchvision.datasets.ImageFolder(cwd/options.dataset/'val', data_transforms)
    test_set = torchvision.datasets.ImageFolder(cwd/options.dataset/'test', data_transforms)
    labels = np.array([item.name for item in (cwd/options.dataset/'train').glob('*')])
    return train_set, val_set, test_set, labels


def grid_search(hyperparam_config):
    names, choices = zip(*hyperparam_config.items())
    for choice in itertools.product(*choices):
        yield dict(zip(names, choice))


def random_search(hyperparam_config, count):
    searched = []
    for _ in range(count):
        hyperparams = None
        while hyperparams is None or hyperparams in searched:
            hyperparams = {name: random.choice(choices)
                           for name, choices in hyperparam_config.items()}
        yield hyperparams


def train_with_tuning(train_set, val_set, test_set, labels, options):
    model_module = visionmodel.MODELS.get(options.model)
    if model_module is None:
        return log.error('Unknown model', model=options.model)

    with open(options.hyperparams) as config_file:
        hyperparam_config = yaml.safe_load(config_file)
        log.info('Hyperparameter search space', **hyperparam_config)

    if options.search is None:
        search = grid_search
    else:
        search = functools.partial(random_search, count=options.search)

    for hyperparams in search(hyperparam_config):
        log.info('Trying new hyperparameters', **hyperparams)
        model = model_module(len(labels), **hyperparams)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams.get('lr', 1e-4),
            weight_decay=hyperparams.get('optimizer_weight_decay', 0),
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            batch_size=hyperparams.get('batch_size', 32),
        )

        for epoch in range(options.max_epochs):
            train_total, train_correct, losses = 0, 0, []
            for minibatch, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                if minibatch%options.print_every == 0:
                    log.debug(
                        'Training update',
                        accuracy=train_correct/train_total,
                        minibatch=minibatch, epoch=epoch,
                        mean_train_loss=np.mean(losses),
                    )
                    losses.clear()
            torch.save({'net': model.state_dict()}, options.params)
            log.info('Epoch complete, saved model state')


def main():
    options = parse_options()
    initialize_logging(options)
    try:
        log.debug('Started training', options=options.__dict__)
        train_set, val_set, test_set, labels = load_datasets(options)
        log.debug('Loaded datasets', labels=len(labels), num_train=len(train_set),
                  num_val=len(val_set), num_test=len(test_set))
        train_with_tuning(train_set, val_set, test_set, labels, options)
    except Exception as exc:
        log.critical('Received error', exc_info=exc)
    finally:
        log.debug('Stopped training')


if __name__ == '__main__':
    main()
