import pathlib
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms


def show_perturbations(image, transform, rows: int = 5, columns: int = 5):
    fig, axes = plt.subplots(rows, columns, figsize=(8, 8))
    axes[0, 0].imshow(image)
    for i in range(1, rows*columns):
        axes[i//columns, i%columns].imshow(transform(image))
    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()


def add_noise(image, std=20):
    pixels = np.array(image)
    pixels = pixels.astype(np.float32) + np.random.normal(0, std, pixels.shape)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, 'RGB')
    return image


def main():
    simple_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cwd = pathlib.Path('.')
    train_path = str(cwd/'data/tiny-imagenet-200/train')
    train_set = torchvision.datasets.ImageFolder(train_path, simple_transforms)

    image = train_set[random.randrange(len(train_set))][0]

    complex_transforms = transforms.Compose([
        # Spatial transformations
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

        # Color/detail transformations
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.Lambda(add_noise),

        transforms.RandomApply([transforms.ToTensor(),
                                transforms.RandomErasing(p=1, scale=(0.02, 0.2), ratio=(0.5, 2)),
                                transforms.ToPILImage()]),
    ])

    show_perturbations(image, complex_transforms)
    plt.show()


if __name__ == '__main__':
    main()
