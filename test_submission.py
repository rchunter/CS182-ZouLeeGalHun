#!/usr/bin/env python3

import copy
import csv
import pathlib
import sys

from PIL import Image
import torch
import torchvision
from torchvision import transforms

try:
    from model import DenoiseNet
except ImportError:
    print('Unable to import module with model definitions.', file=sys.stderr)
    exit(1)


preprocessing_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=Image.LANCZOS),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
intermediate_transforms = transforms.Compose([
    transforms.Lambda(lambda image: torch.clamp(image, 0, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def classify(image, denoise_model, model):
    image = preprocessing_transforms(image)[None, :]
    denoised_image = image[0]
    # denoised_image = denoise_model(image)[0]
    normalized_image = intermediate_transforms(denoised_image)[None, :]
    outputs = model(normalized_image)
    _, predicted = outputs.max(1)
    return predicted


def test_all(denoise_model, model, classes):
    correct, total = 0, 0
    for path in pathlib.Path('data/tiny-imagenet-200/val').glob('**/*.JPEG'):
        with open(path, 'rb') as image_file:
            image = Image.open(image_file).convert('RGB')
        _, _, _, true_label, _ = str(path).split('/')
        predicted_label = classes[classify(image, denoise_model, model)]
        correct += predicted_label == true_label
        total += 1
        print('Running accuracy: {:.3f}%, total: {}'.format(100*correct/total, total))


def main(classes_path='data/tiny-imagenet-200/wnids.txt', output_filename='eval_classified.csv'):
    device = torch.device('cuda:0' if False and torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('params/denoise-final.pt', map_location=device)
    denoise_model = DenoiseNet()
    denoise_model.load_state_dict(checkpoint['net'])
    denoise_model.eval()
    print('Loaded denoising network')

    try:
        with open(classes_path) as classes_file:
            classes = classes_file.read().strip().split('\n')
            classes.sort()
            print('Found {} classes'.format(len(classes)))
    except FileNotFoundError:
        print('Unable to find file with classes.', file=sys.stderr)
        exit(1)

    checkpoint = torch.load('params/mobilenet-final.pt', map_location=device)
    state_dict = checkpoint['net']
    corrected_state_dict = copy.deepcopy(state_dict)
    for name in state_dict:
        if name.startswith('classifier.1'):
            _, _, end = name.split('.')
            corrected_state_dict['classifier.' + end] = state_dict[name]
            del corrected_state_dict[name]
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(corrected_state_dict)
    model.eval()
    print('Loaded classifier')

    if len(sys.argv) < 2:
        print('Usage: python3 {} [eval.csv]'.format(sys.argv[0]))
        exit(1)

    with open(output_filename, 'w') as output_file:
        with pathlib.Path(sys.argv[1]).open() as input_file:
            for line in input_file:
                image_id, image_path, image_height, image_width, image_channels = line.strip().split(',')
                with open(image_path, 'rb') as image_file:
                    image = Image.open(image_file).convert('RGB')
                predicted = classify(image, denoise_model, model)
                output_file.write('{},{}\n'.format(image_id, classes[predicted]))
                print('Classified {} -> {}'.format(image_id, classes[predicted]))


if __name__ == '__main__':
    main()
