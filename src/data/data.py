'''
Data loading and helper methods for MNIST.
'''

import numpy as np
import random
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from typing import Tuple
from torch.utils.data import Dataset, random_split


def get_MNIST(val_split: float = 0.1, rotation_degrees: int = 0, crop_padding: int = 0, duplicate_with_augment: bool = False, verbose: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads the MNIST dataset and optionally applies specified transformations.

    Args:
        val_split (float): Fraction of training data to use for validation. Must be in [0, 1).
        rotation_degrees (int): Degrees of random rotation. If 0, no rotation is applied.
        crop_padding (int): Pixels of padding for random cropping. If 0, no cropping is applied.
        duplicate_with_augment (bool): Whether to duplicate images when appling augmentation.
        verbose (bool): If True, prints data details.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: (train_dataset, test_dataset, val_dataset)
    """
    assert 0 <= val_split < 1, "val_split must be in the range [0, 1)."
    mnist_train = torchvision.datasets.MNIST('src/data', train=True, download=True)

    # Standardize data based on train split values
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255

    base_transforms_list = [
    	transforms.ToTensor(),
		transforms.Normalize(mean=[mean], std=[std]),
	]
    aug_transforms_list = []

    # Data Augmentation
    if duplicate_with_augment:
        if rotation_degrees > 0:
            aug_transforms_list.append(DuplicateWithRandomRotation(rotation_degrees))
        if crop_padding > 0:
            aug_transforms_list.append(DuplicateWithRandomCrop(28, crop_padding))
    else:
        if rotation_degrees > 0:
            aug_transforms_list.append(transforms.RandomRotation(rotation_degrees, fill=(0,)))
        if crop_padding > 0:
            aug_transforms_list.append(transforms.RandomCrop(28, crop_padding))

    train_transforms = transforms.Compose(base_transforms_list+aug_transforms_list)
    test_transforms = transforms.Compose(base_transforms_list)
    
    mnist_train = torchvision.datasets.MNIST(
		'src/data', train=True, download=True, transform=train_transforms)
    mnist_test = torchvision.datasets.MNIST(
		'src/data', train=False, download=True, transform=test_transforms)

    val_count = int(val_split * len(mnist_train))
    train_count = len(mnist_train) - val_count
    mnist_train, mnist_val = data.random_split(mnist_train, [train_count, val_count])

    # Remove augmentation from validation set
    mnist_val = copy.deepcopy(mnist_val)
    mnist_val.dataset.transform = test_transforms

    if verbose:
        print(f"Train: Dataset MNIST")
        print(f"    Number of datapoints: {len(mnist_train)}\n")
        print(f"Val: Dataset MNIST")
        print(f"    Number of datapoints: {len(mnist_val)}")
        print(f'Test: {mnist_test}\n')

    return mnist_train, mnist_test, mnist_val


class DuplicateWithRandomCrop:
    """
    A transformation that returns both the original image and a randomly cropped version.
    """
    def __init__(self, crop_size, padding=0):
        """
        Initialize the transformation with the crop size.

        Args:
            crop_size (tuple of int): The size (height, width) of the random crop to apply.
            padding (int): The size of the padding to apply.
        """
        self.random_crop = transforms.RandomCrop(crop_size, padding)

    def __call__(self, img):
        """
        Apply the transformation to an image.

        Args:
            img (PIL.Image or Tensor)

        Returns:
            Tuple[PIL.Image, PIL.Image]: The original image and the randomly cropped version.
        """
        original = img
        cropped = self.random_crop(img)
        return original, cropped


class DuplicateWithRandomRotation:
    """
    A transformation that returns both the original image and a randomly rotated version.
    """
    def __init__(self, degrees):
        """
        Initialize the transformation with the rotation range.

        Args:
            degrees (float or tuple of float): The range of degrees to select from. If a single float is provided, the rotation range will be (-degrees, +degrees). If a tuple is provided, it specifies the (min, max) rotation range.
        """
        self.random_rotation = transforms.RandomRotation(degrees)

    def __call__(self, img):
        """
        Apply the transformation to an image.

        Args:
            img (PIL.Image or Tensor)

        Returns:
            Tuple[PIL.Image, PIL.Image]: The original image and the randomly rotated version.
        """
        original = img
        rotated = self.random_rotation(img)
        return original, rotated
