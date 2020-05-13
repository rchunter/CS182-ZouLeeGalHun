#!/usr/bin/env python3

import csv
import pathlib
import sys

from PIL import Image
import torch
from torchvision import transforms

try:
    from model import DenoiseNet
except ImportError:
    print('Unable to import module with model definitions.', file=sys.stderr)
    exit(1)


def main(classes_path='data/tiny-imagenet-200/wnids.txt', output_filename='eval_classified.csv'):
    data_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        with open(classes_path) as classes_file:
            classes = classes_file.read().strip().split('\n')
            classes.sort()
            print('Found {} classes'.format(len(classes)))
    except FileNotFoundError:
        print('Unable to find file with classes.', file=sys.stderr)
        exit(1)

    if len(sys.argv) < 2:
        print('Usage: python3 {} [eval.csv]'.format(sys.argv[0]))
        exit(1)

    with open(output_filename, 'w') as output_file:
        with pathlib.Path(sys.argv[1]).open() as input_file:
            for line in input_file:
                image_id, image_path, image_height, image_width, image_channels = line.strip().split(',')
                print(image_id, image_path, image_height, image_width, image_channels)
                with open(image_path, 'rb') as image_file:
                    image = Image.open(image_file).convert('RGB')
                image = data_transforms(image)[None, :]
                # outputs = model(image)
                # _, predicted = outputs.max(1)
                # output_file.write('{},{}\n'.format(image_id, classes[predicted]))


if __name__ == '__main__':
    main()
