import pathlib
import json
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
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


def plot_denoise():
    losses = []
    batches_per_epoch = (100000 + 15)//16
    with open('logs/denoise-final.log') as log_file:
        for line in log_file:
            data = json.loads(line)
            losses.append(data['loss'])
    plt.style.use('seaborn')
    plt.figure(figsize=(8, 4))
    plt.plot(100*(1 + np.arange(len(losses))), 10*np.log10(9*224*224/np.array(losses)))
    plt.title('Denoise Network PSNR Over Time (One Epoch)')
    plt.xlim(80, batches_per_epoch)
    plt.xlabel('Minibatch Number')
    plt.ylabel('PSNR [dB]')
    plt.tight_layout()
    plt.savefig('report/figures/denoise-psnr.png', dpi=200)
    plt.show()


def plot_train():
    with open('logs/mobilenet-final.log') as log_file:
        mode = 'train'
        train_ctr, val_ctr = 0, 0
        train_accuracy, val_accuracy = [], []
        train_top_accuracy, val_top_accuracy = [], []
        for line in log_file:
            line = json.loads(line)
            if line['event'] == 'Validation phase':
                mode = 'valid'
            elif line['event'] == 'Training phase':
                mode = 'train'
            elif line['event'] == 'Update':
                if mode == 'valid':
                    val_ctr += 500
                    train_accuracy.append(line['accuracy'])
                    train_top_accuracy.append(line['top_accuracy'])
                else:
                    train_ctr += 500
                    val_accuracy.append(line['accuracy'])
                    val_top_accuracy.append(line['top_accuracy'])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].set_title('Training Accuracy Over Time')
    axes[0].plot(500 + 500*np.arange(len(train_accuracy)), train_accuracy, label='Top-1 Accuracy')
    axes[0].plot(500 + 500*np.arange(len(train_top_accuracy)), train_top_accuracy, label='Top-5 Accuracy')
    axes[0].set_xlabel('Batch number')
    axes[0].set_ylabel('Percentage')
    axes[0].legend()
    axes[1].set_title('Validation Accuracy Over Time')
    axes[1].plot(500 + 500*np.arange(len(val_accuracy)), val_accuracy, label='Top-1 Accuracy')
    axes[1].plot(500 + 500*np.arange(len(val_top_accuracy)), val_top_accuracy, label='Top-5 Accuracy')
    axes[1].legend()
    axes[1].set_xlabel('Batch number')
    axes[1].set_ylabel('Percentage')
    plt.tight_layout()
    plt.savefig('report/figures/training.png', dpi=200)
    plt.show()


def main():
    cwd = pathlib.Path('.')
    train_path = str(cwd/'data/tiny-imagenet-200/train')
    train_set = torchvision.datasets.ImageFolder(train_path)

    samples = [np.array(train_set[random.randrange(len(train_set))][0].convert('L')) for _ in range(100)]
    dct = np.empty((8, 8), dtype=np.float64)
    n = 0
    for sample in samples:
        for row in range(0, 64, 8):
            for column in range(0, 64, 8):
                n += 1
                block = sample[row : row+8, column : column+8].astype(np.float64)
                block = scipy.fftpack.dct(block, axis=0)
                block = scipy.fftpack.dct(block, axis=1)
                dct += np.abs(block)

    # print(dct)
    plt.figure(figsize=(4, 4))
    plt.imshow(np.log10(dct), cmap='plasma')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Discrete Cosine Transform Coefficients')
    plt.xlabel('Horizontal Index')
    plt.ylabel('Vertical Index')
    plt.tight_layout()
    plt.savefig('report/figures/dct.png', dpi=200, transparent=True)
    plt.show()

    image = train_set[random.randrange(len(train_set))][0]

    data_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        spatial_transforms,
        color_transform,
        transforms.ToTensor(),
        noise_transform,
        # transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.5, 2)),
        transforms.ToPILImage(),
    ])

    show_perturbations(image, data_transforms)
    # plt.savefig('report/figures/perturb.png', dpi=200, transparent=True)
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

    # denoise_output = denoise_model(denoise_input)
    # axes[0, 1].imshow(to_image(torch.clamp(denoise_output[0], 0, 1)))
    # axes[0, 1].set_title('Output (No Perturbation)')

    denoise_input = transforms.Compose([
        to_image,
        color_transform,
        transforms.ToTensor(),
        noise_transform,
    ])(denoise_input[0])[None, :]
    axes[0, 1].imshow(to_image(denoise_input[0]))
    axes[0, 1].set_title('Input (Color Perturbation)')

    denoise_output = denoise_model(denoise_input)
    denoise_output = torch.clamp(denoise_output[0], 0, 1)
    axes[1, 0].imshow(to_image(denoise_output))
    axes[1, 0].set_title('Output')

    axes[1, 1].imshow(to_image(5*(denoise_output - denoise_input[0])))
    axes[1, 1].set_title('10x Error')

    for ax in axes.flatten():
        ax.axis('off')

    plt.suptitle('Denoise Network', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig('report/figures/denoise.png', dpi=200, transparent=True)
    plt.show()


if __name__ == '__main__':
    # main()
    # plot_denoise()
    plot_train()
