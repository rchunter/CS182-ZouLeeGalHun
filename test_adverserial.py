#!/usr/bin/env python3
from __future__ import print_function

import argparse
import copy
import datetime
import functools
import itertools
import logging.config
import pathlib
import random
import typing
import sys

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import yaml
import numpy as np

def JPEGCompression(image):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=75, optimize=True)
    output.seek(0)
    return Image.open(output)

try:
    from model import DenoiseNet
except ImportError:
    print('Unable to import module with model definitions.', file=sys.stderr)
    exit(1)

cwd = pathlib.Path('.')


normalize_stds = [0.229, 0.224, 0.225]
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], normalize_stds)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def fsgm_test(model, device, test_loader, epsilon, options):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # if options.crop_ensemble:
        #     bs, ncrops, c, h, w = data.size()
        #     result = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
        #     output = result.view(bs, ncrops, -1).mean(1) # avg over crops

        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def load_datasets(options):
    data_transforms = [
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if options.jpeg:
        data_transforms += [transforms.Lambda(JPEGCompression)]
    if options.crop_ensemble:
        data_transforms += [transforms.Compose([
            transforms.FiveCrop(),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])]


def test(test_set, labels, options):
    pass
    

def parse_options():
    parser = argparse.ArgumentParser(description='Testing script for image classifier on adversial inputs')
    parser.add_argument('--denoise', type=bool, default=False, help='use denoise network')
    parser.add_argument('--jpeg', type=bool, default=False, help='use jpeg compression')
    parser.add_argument('--crop-ensemble', type=bool, default=False, help='use crop ensemble')
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--params', type=str, default=None, help='path to model params')
    parser.add_argument('--dataset', type=str, default=str(cwd/'data/tiny-imagenet-200'),
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    options = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    options.test = f'{options.model}-{now}'
    options.params = options.params or str(cwd/f'params/{options.train}.pt')
    return options

def main():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    options = parse_options()
