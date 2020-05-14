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


def fsgm_test(model, device, test_loader, epsilon, preprocess, midprocess, options):
    correct = 0
    adv_examples = []

    # Generate adverserial samples from test set
    # Adverserial samples are FSGM on exposed model
    for idx, (data, target) in enumerate(test_loader):
        print(f'\rOf {idx} images, found {len(adv_examples)} adverserial examples', end='')
        if len(adv_examples) >= 200:
            break

        data, target = data.to(device), target.to(device)

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
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 200:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), target.item(), adv_ex) )
    print()
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tPure Test Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Test on robust network
    if options.denoise:
        checkpoint = torch.load('params/denoise-final.pt', map_location=device)
        denoise_model = DenoiseNet()
        denoise_model.load_state_dict(checkpoint['net'])
        denoise_model.eval()

    correct = 0
    for _, _, adv_ex, target in adv_examples:
        # First preprocess image for denoising
        adv_ex = preprocess(adv_ex)[None, :]
        if options.denoise:
            adv_ex = denoise_model(adv_ex)[0]
        adv_ex = midprocess(adv_ex)[None, :]
        output = model(adv_ex)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target:
            correct += 1

    final_acc = correct/200.
    print("Epsilon: {}\tRobust Test Accuracy = {} / {} = {}".format(epsilon, correct, 200, final_acc))

def test(options, epsilons=[.25, .3]):
    # Create data loaders
    loader = functools.partial(torch.utils.data.DataLoader, shuffle=True,
                            pin_memory=True, num_workers=8,
                            batch_size=1)
    test_set = torchvision.datasets.ImageFolder(str(cwd/options.dataset/'val'),
                                                transform=transforms.ToTensor())
    test_loader = loader(test_set)

    # Create transforms
    preprocessing_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    intermediate_transforms = [transforms.ToPILImage]
    if options.jpeg:
        intermediate_transforms += [transforms.Lambda(JPEGCompression)]
    intermediate_transforms += [
        transforms.Lambda(lambda image: torch.clamp(image, 0, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if options.crop_ensemble:
        intermediate_transforms += [transforms.Compose([
            transforms.FiveCrop(),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])]
    intermediate_transforms = transforms.Compose(intermediate_transforms)

    device = torch.device('cuda:0' if False and torch.cuda.is_available() else 'cpu')
    
    # Model
    if options.model == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
    elif options.model == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)

    model.to(device)
    model.eval()

    for eps in epsilons:
        fsgm_test(model, device, test_loader, eps, preprocessing_transforms, intermediate_transforms, options)

def parse_options():
    parser = argparse.ArgumentParser(description='Testing script for image classifier on adversial inputs')
    parser.add_argument('--denoise', type=bool, default=False, help='use denoise network')
    parser.add_argument('--jpeg', type=bool, default=False, help='use jpeg compression')
    parser.add_argument('--crop-ensemble', type=bool, default=False, help='use crop ensemble')
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--params', type=str, default=None, help='path to model params')
    parser.add_argument('--dataset', type=str, default=str(cwd/'data/tiny-imagenet-200'),
                        help='path to dataset')
    options = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    options.test = f'{options.model}-{now}'
    return options

def main():

    options = parse_options()
    test(options)

if __name__ == '__main__':
    main()
