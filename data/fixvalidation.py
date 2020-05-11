#!/usr/bin/env python3

import pathlib

def main():
    val_dir = pathlib.Path('tiny-imagenet-200/val')
    assert val_dir.exists()
    with open(val_dir/'val_annotations.txt') as annotations_file:
        lines = list(annotations_file)
        labels = dict(line.strip().split()[:2] for line in lines)
    print(f'Samples in validation set: {len(labels)}')
    print(f'Labels: {len(set(labels.values()))}')
    for label in labels.values():
        (val_dir/label).mkdir(exist_ok=True)

    for image in val_dir.glob('images/val_*.JPEG'):
        dest = val_dir/labels[image.name]/image.name
        print(f'Renaming: {image} -> {dest}')
        image.rename(dest)
    else:
        print('No images to move')

    if not len(list(val_dir.glob('images/*'))):
        (val_dir/'images').rmdir()

if __name__ == '__main__':
    main()
