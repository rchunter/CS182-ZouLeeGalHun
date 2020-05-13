import pathlib
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

from model import DenoiseNet


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
color_transform = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.05)
noise_transform = transforms.Lambda(lambda image, std=0.1: torch.clamp(image + std*torch.randn_like(image), 0, 1))


def show_perturbations(image, transform, rows: int = 4, columns: int = 4):
    fig, axes = plt.subplots(rows, columns, figsize=(8, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Ground Truth')
    for i in range(1, rows*columns):
        axes[i//columns, i%columns].imshow(transform(image))
    for ax in axes.flatten():
        ax.axis('off')
    plt.suptitle('Classifier Perturbations', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


def main():
    cwd = pathlib.Path('.')
    train_path = str(cwd/'data/tiny-imagenet-200/train')
    train_set = torchvision.datasets.ImageFolder(train_path)

    image = train_set[random.randrange(len(train_set))][0]

    data_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        spatial_transforms,
        color_transform,
        transforms.ToTensor(),
        noise_transform,
        transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.5, 2)),
        transforms.ToPILImage(),
    ])

    show_perturbations(image, data_transforms)
    plt.savefig('report/figures/perturb.png', dpi=200, transparent=True)
    plt.show()

    # Denoise Network

    image = train_set[random.randrange(len(train_set))][0]

    checkpoint = torch.load('params/denoise-final.pt', map_location=torch.device('cpu'))
    denoise_model = DenoiseNet()
    denoise_model.load_state_dict(checkpoint['net'])
    denoise_model.eval()

    denoise_model_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    to_image = transforms.ToPILImage()

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    denoise_input = denoise_model_transforms(image)[None, :]

    axes[0, 0].imshow(to_image(denoise_input[0]))
    axes[0, 0].set_title('Input (No Perturbation)')

    denoise_output = denoise_model(denoise_input)
    axes[0, 1].imshow(to_image(torch.clamp(denoise_output[0], 0, 1)))
    axes[0, 1].set_title('Output (No Perturbation)')

    denoise_input = transforms.Compose([
        to_image,
        color_transform,
        transforms.ToTensor(),
        noise_transform,
    ])(denoise_input[0])[None, :]
    axes[1, 0].imshow(to_image(denoise_input[0]))
    axes[1, 0].set_title('Input (Color Perturbation)')

    denoise_output = denoise_model(denoise_input)
    axes[1, 1].imshow(to_image(torch.clamp(denoise_output[0], 0, 1)))
    axes[1, 1].set_title('Output (Color Perturbation)')

    for ax in axes.flatten():
        ax.axis('off')

    plt.suptitle('Denoise Network', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('report/figures/denoise.png', dpi=200, transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
