"""
Tests for data loading methods in src/data/data.py

Run with: pytest tests/test_data.py -v
"""

import pytest
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
from pathlib import Path
from PIL import Image

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data import (
    get_MNIST,
    DuplicateWithRandomCrop,
    DuplicateWithRandomRotation,
)


# Tests for get_MNIST function
class TestGetMNIST:
    """Test MNIST data loading function."""

    def test_get_mnist_returns_three_datasets(self):
        """Test get_MNIST returns three datasets (train, test, val)."""
        train, test, val = get_MNIST(val_split=0.1, verbose=False)
        assert train is not None
        assert test is not None
        assert val is not None

    def test_get_mnist_no_validation_split(self):
        """Test get_MNIST with no validation split."""
        train, test, val = get_MNIST(val_split=0.0, verbose=False)

        assert len(train) == 60000
        assert len(test) == 10000
        assert len(val) == 0

    def test_get_mnist_custom_validation_split(self):
        """Test get_MNIST with custom validation split."""
        val_split = 0.2
        train, test, val = get_MNIST(val_split=val_split, verbose=False)

        expected_val_size = int(val_split * 60000)
        expected_train_size = 60000 - expected_val_size

        assert len(train) == expected_train_size
        assert len(val) == expected_val_size

    def test_get_mnist_invalid_val_split_raises_error(self):
        """Test get_MNIST raises error for invalid val_split."""
        with pytest.raises(AssertionError):
            get_MNIST(val_split=1.5)

        with pytest.raises(AssertionError):
            get_MNIST(val_split=-0.1)

        with pytest.raises(AssertionError):
            get_MNIST(val_split=1.0)

    def test_get_mnist_data_shape(self):
        """Test get_MNIST returns data with correct shape."""
        train, test, val = get_MNIST(val_split=0.1, verbose=False)

        # Get a sample from training set
        img, label = train[0]

        # MNIST images should be 1x28x28 after transforms
        assert img.shape == (1, 28, 28)
        assert isinstance(label, int)
        assert 0 <= label <= 9

    def test_get_mnist_with_duplicate_augmentation(self):
        """Test get_MNIST with duplicate augmentation."""
        train, test, val = get_MNIST(
            val_split=0.1,
            rotation_degrees=15,
            duplicate_with_augment=True,
            verbose=False
        )

        assert any(isinstance(t, DuplicateWithRandomRotation) for t in train.dataset.transform.transforms)

    def test_get_mnist_val_no_augmentation(self):
        """Test that validation set has no augmentation even when train does."""
        train, test, val = get_MNIST(
            val_split=0.2,
            rotation_degrees=15,
            crop_padding=4,
            verbose=False
        )

        assert test.transform is not None
        assert val.dataset.transform is not None

        # Verify the validation set uses the correct transforms (no augmentation)
        assert not any(isinstance(t, DuplicateWithRandomCrop) for t in val.dataset.transform.transforms)
        assert not any(isinstance(t, DuplicateWithRandomRotation) for t in val.dataset.transform.transforms)

        # The transform should be the same as test transforms
        assert val.dataset.transform.transforms == test.transform.transforms


# Tests for DuplicateWithRandomCrop
class TestDuplicateWithRandomCrop:
    """Test DuplicateWithRandomCrop transformation."""

    def test_duplicate_crop_returns_tuple(self):
        """Test DuplicateWithRandomCrop returns tuple of two images."""
        transform = DuplicateWithRandomCrop(crop_size=28, padding=4)

        # Create a sample PIL image
        img = Image.new('L', (28, 28), color=128)

        result = transform(img)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_duplicate_crop_preserves_original(self):
        """Test DuplicateWithRandomCrop preserves original image."""
        transform = DuplicateWithRandomCrop(crop_size=28, padding=4)

        # Create a sample PIL image with specific pattern
        img = Image.new('L', (28, 28), color=100)

        original, cropped = transform(img)

        # Original should be the same as input
        assert original == img

    def test_duplicate_crop_output_shape(self):
        """Test DuplicateWithRandomCrop maintains correct image size."""
        transform = DuplicateWithRandomCrop(crop_size=28, padding=4)

        img = Image.new('L', (28, 28), color=128)
        original, cropped = transform(img)

        # Both should be PIL Images with same size
        assert original.size == (28, 28)
        assert cropped.size == (28, 28)


# Tests for DuplicateWithRandomRotation
class TestDuplicateWithRandomRotation:
    """Test DuplicateWithRandomRotation transformation."""

    def test_duplicate_rotation_returns_tuple(self):
        """Test DuplicateWithRandomRotation returns tuple of two images."""
        transform = DuplicateWithRandomRotation(degrees=15)

        # Create a sample PIL image
        img = Image.new('L', (28, 28), color=128)

        result = transform(img)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_duplicate_rotation_preserves_original(self):
        """Test DuplicateWithRandomRotation preserves original image."""
        transform = DuplicateWithRandomRotation(degrees=15)

        # Create a sample PIL image
        img = Image.new('L', (28, 28), color=100)

        original, rotated = transform(img)

        # Original should be the same as input
        assert original == img

    def test_duplicate_rotation_output_shape(self):
        """Test DuplicateWithRandomRotation maintains correct image size."""
        transform = DuplicateWithRandomRotation(degrees=15)

        img = Image.new('L', (28, 28), color=128)
        original, rotated = transform(img)

        # Both should be PIL Images with same size
        assert original.size == (28, 28)
        assert rotated.size == (28, 28)
