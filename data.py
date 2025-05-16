'''
Data helper methods for MNIST.
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

def get_MNIST(val_split: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
	"""
    Loads the MNIST dataset.

    Args:
        val_split (float): Fraction of training data to use for validation. Must be in [0, 1).

    Returns:
        Tuple[Dataset, Dataset, Dataset]: (train_dataset, test_dataset, val_dataset)
    """
	assert 0 <= val_split < 1, "val_split must be in the range [0, 1)."
	mnist_train = torchvision.datasets.MNIST('data', train=True, download=True)

    # Standardize data based on train split values
	mean = mnist_train.data.float().mean() / 255
	std = mnist_train.data.float().std() / 255
	transform = transforms.Compose([
    	transforms.ToTensor(),
		transforms.Normalize(mean=[mean], std=[std]),
	])

	mnist_train = torchvision.datasets.MNIST(
		'data', train=True, download=True, transform=transform)
	mnist_test = torchvision.datasets.MNIST(
		'data', train=False, download=True, transform=transform)

	val_count = int(val_split * len(mnist_train))
	train_count = len(mnist_train) - val_count
	mnist_train, mnist_val = data.random_split(mnist_train, [train_count, val_count])

	return mnist_train, mnist_test, mnist_val


def import_MNIST(augment_data=True, duplicate_with_transform=False):
    """
    Loads the MNIST dataset and applies specified transformations, including optional data augmentation.

    Parameters:
    - augment_data : bool, optional (default=True). Whether to apply data augmentation to the training data.
    - duplicate_with_transform : bool, optional (default=False). Whether to duplicate data when apply augmentation.

    Returns:
    - tuple:
        - mnist_train : torch.utils.data.Dataset. The training dataset.
        - mnist_test : torch.utils.data.Dataset. The test dataset.
        - mnist_val : torch.utils.data.Dataset. The validation dataset, derived from the training data, without data augmentation.
    """
    mnist_train = torchvision.datasets.MNIST('data', train=True, download=True)

    # Standardize data based on train split values
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255

    transform_list = [transforms.ToTensor(
    ), transforms.Normalize(mean=[mean], std=[std]),]

    # Add data augmentation if specified
    if duplicate_with_transform:
        augmentation_transform_list = [DuplicateWithRandomRotation(
            5), DuplicateWithRandomCrop(28, padding=2),]
    else:
        augmentation_transform_list = [transforms.RandomRotation(
            5, fill=(0,)), transforms.RandomCrop(28, padding=2),]

    train_transforms = transforms.Compose(
        transform_list+augmentation_transform_list if augment_data else transform_list)
    test_transforms = transforms.Compose(transform_list)

    # Load MNIST and apply transformations
    mnist_train = torchvision.datasets.MNIST(
        'data', train=True, download=True, transform=train_transforms)
    mnist_test = torchvision.datasets.MNIST(
        'data', train=False, download=True, transform=test_transforms)

    # Create validation set by randomly sampling 10% of test set
    train_count, val_count = len(
        mnist_train) - int(0.1 * len(mnist_train)), int(0.1 * len(mnist_train))
    mnist_train, mnist_val = data.random_split(
        mnist_train, [train_count, val_count])

    # Remove augmentation from validation set
    mnist_val = copy.deepcopy(mnist_val)
    mnist_val.dataset.transform = test_transforms

    # Inspect dataset
    print(f"Train: Dataset MNIST")
    print(f"    Number of datapoints: {len(mnist_train)}\n")
    print(f'Test: {mnist_test}\n')
    print(f"Val: Dataset MNIST")
    print(f"    Number of datapoints: {len(mnist_val)}")

    return mnist_train, mnist_test, mnist_val

class DuplicateWithRandomCrop:
    """
    A transformation that returns both the original image and a randomly cropped version.
    """
    def __init__(self, crop_size, padding=0):
        """
        Initialize the transformation with the crop size.

        Parameters:
        - crop_size : tuple of int. The size (height, width) of the random crop to apply.
        - padding : int, optional (default=0). The size of the padding to apply.
        """
        self.random_crop = transforms.RandomCrop(crop_size, padding)

    def __call__(self, img):
        """
        Apply the transformation to an image.

        Parameters:
        - img : PIL.Image or Tensor

        Returns:
        - tuple : A tuple containing the original image and the randomly cropped version.
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

        Parameters:
        - degrees : float or tuple of float
            The range of degrees to select from. If a single float is provided, the rotation
            range will be (-degrees, +degrees). If a tuple is provided, it specifies the
            (min, max) rotation range.
        """
        self.random_rotation = transforms.RandomRotation(degrees)

    def __call__(self, img):
        """
        Apply the transformation to an image.

        Parameters:
        - img : PIL.Image or Tensor. The input image to transform.

        Returns:
        - tuple : A tuple containing the original image and the randomly rotated version.
        """
        original = img
        rotated = self.random_rotation(img)
        return original, rotated
