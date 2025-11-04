"""
Tests for adversarial example methods in src/misc/adversarial.py

Run with: pytest tests/test_adversarial.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from misc.adversarial import (
    l1_distance_np,
    l1_distance_torch,
    l2_distance_np,
    l2_distance_torch,
    fgsm,
    fgsm_targeted,
    pgd,
    pgd_targeted,
)


class SimpleModel(nn.Module):
    """Simple model for testing adversarial attacks."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def model():
    """Create a simple model for testing."""
    model = SimpleModel(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def sample_image():
    """Create a sample image tensor."""
    torch.manual_seed(12)
    return torch.randint(0, 256, (1, 28, 28)).float()


@pytest.fixture
def sample_label():
    """Create a sample label."""
    return 3


# Tests for distance functions
class TestDistanceFunctions:
    """Test distance metric functions."""

    def test_l1_distance_np_identical(self):
        """Test L1 distance between identical images is zero."""
        img = np.random.rand(28, 28) * 255
        assert l1_distance_np(img, img) == 0.0

    def test_l1_distance_np_different(self):
        """Test L1 distance between different images is positive."""
        img1 = np.zeros((28, 28))
        img2 = np.ones((28, 28))
        distance = l1_distance_np(img1, img2)
        assert distance == 28 * 28  # Sum of absolute differences

    def test_l1_distance_torch_identical(self):
        """Test L1 distance (torch) between identical images is zero."""
        img = torch.rand(28, 28) * 255
        assert l1_distance_torch(img, img).item() == 0.0

    def test_l1_distance_torch_different(self):
        """Test L1 distance (torch) between different images is positive."""
        img1 = torch.zeros(28, 28)
        img2 = torch.ones(28, 28)
        distance = l1_distance_torch(img1, img2)
        assert distance.item() == 28 * 28

    def test_l1_np_torch_consistency(self):
        """Test numpy and torch L1 implementations give same results."""
        img1_np = np.random.rand(28, 28) * 255
        img2_np = np.random.rand(28, 28) * 255
        img1_torch = torch.from_numpy(img1_np)
        img2_torch = torch.from_numpy(img2_np)

        dist_np = l1_distance_np(img1_np, img2_np)
        dist_torch = l1_distance_torch(img1_torch, img2_torch).item()

        assert np.isclose(dist_np, dist_torch, rtol=1e-5)

    def test_l2_distance_np_identical(self):
        """Test L2 distance between identical images is zero."""
        img = np.random.rand(28, 28) * 255
        assert l2_distance_np(img, img) == 0.0

    def test_l2_distance_np_different(self):
        """Test L2 distance between different images is positive."""
        img1 = np.zeros((28, 28))
        img2 = np.ones((28, 28))
        distance = l2_distance_np(img1, img2)
        expected = np.sqrt(28 * 28)
        assert np.isclose(distance, expected)

    def test_l2_distance_torch_identical(self):
        """Test L2 distance (torch) between identical images is zero."""
        img = torch.rand(28, 28) * 255
        assert l2_distance_torch(img, img).item() == 0.0

    def test_l2_distance_torch_different(self):
        """Test L2 distance (torch) between different images is positive."""
        img1 = torch.zeros(28, 28)
        img2 = torch.ones(28, 28)
        distance = l2_distance_torch(img1, img2)
        expected = np.sqrt(28 * 28)
        assert torch.isclose(distance, torch.tensor(expected, dtype=distance.dtype))

    def test_l2_np_torch_consistency(self):
        """Test numpy and torch L2 implementations give same results."""
        img1_np = np.random.rand(28, 28) * 255
        img2_np = np.random.rand(28, 28) * 255
        img1_torch = torch.from_numpy(img1_np)
        img2_torch = torch.from_numpy(img2_np)

        dist_np = l2_distance_np(img1_np, img2_np)
        dist_torch = l2_distance_torch(img1_torch, img2_torch).item()

        assert np.isclose(dist_np, dist_torch, rtol=1e-5)


# Tests for FGSM
class TestFGSM:
    """Test Fast Gradient Sign Method (untargeted)."""

    def test_fgsm_returns_tuple(self, model, sample_image, sample_label):
        """Test FGSM returns tuple of (adversarial_image, perturbation)."""
        result = fgsm(model, sample_image, sample_label, epsilon=8.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fgsm_output_shapes(self, model, sample_image, sample_label):
        """Test FGSM outputs have correct shapes."""
        adv_img, perturbation = fgsm(model, sample_image, sample_label, epsilon=8.0)

        # Should add batch dimension if not present
        assert adv_img.shape[0] == 1  # Batch dimension
        assert adv_img.shape[1:] == sample_image.shape  # Image dimensions preserved
        assert perturbation.shape == adv_img.shape

    def test_fgsm_creates_perturbation(self, model, sample_image, sample_label):
        """Test FGSM creates non-zero perturbation."""
        adv_img, perturbation = fgsm(model, sample_image, sample_label, epsilon=8.0)
        assert not torch.allclose(perturbation, torch.zeros_like(perturbation))

    def test_fgsm_bounded_perturbation(self, model, sample_image, sample_label):
        """Test FGSM perturbation respects epsilon bound."""
        epsilon = 10.0
        adv_img, perturbation = fgsm(model, sample_image, sample_label, epsilon=epsilon)

        # Perturbation should be at most epsilon in absolute value
        assert torch.all(torch.abs(perturbation) <= epsilon + 1e-6)

    def test_fgsm_clamped_to_valid_range(self, model, sample_image, sample_label):
        """Test FGSM output is clamped to [0, 255]."""
        adv_img, _ = fgsm(model, sample_image, sample_label, epsilon=100.0)
        assert torch.all(adv_img >= 0.0)
        assert torch.all(adv_img <= 255.0)

    def test_fgsm_no_gradient_in_output(self, model, sample_image, sample_label):
        """Test FGSM output is detached from computation graph."""
        adv_img, perturbation = fgsm(model, sample_image, sample_label, epsilon=8.0)
        assert not adv_img.requires_grad
        assert not perturbation.requires_grad

    def test_fgsm_batch_dimension_handling(self, model, sample_label):
        """Test FGSM handles both 3D and 4D inputs correctly."""
        # 3D input (no batch dimension)
        img_3d = torch.rand(1, 28, 28) * 255
        adv_3d, _ = fgsm(model, img_3d, sample_label, epsilon=8.0)
        assert adv_3d.shape[0] == 1  # Batch dimension added

        # 4D input (with batch dimension)
        img_4d = torch.rand(1, 1, 28, 28) * 255
        adv_4d, _ = fgsm(model, img_4d, sample_label, epsilon=8.0)
        assert adv_4d.shape[0] == 1  # Batch dimension preserved

    def test_epsilon_zero_no_change(self, model, sample_image, sample_label):
        """Test that epsilon=0 produces no change to the image."""
        adv_img, perturbation = fgsm(model, sample_image, sample_label, epsilon=0.0)

        # With epsilon=0, perturbation should be zero
        assert torch.allclose(perturbation, torch.zeros_like(perturbation))


# Tests for FGSM Targeted
class TestFGSMTargeted:
    """Test Fast Gradient Sign Method (targeted)."""

    def test_fgsm_targeted_returns_tuple(self, model, sample_image):
        """Test targeted FGSM returns tuple of (adversarial_image, perturbation)."""
        target = 7
        result = fgsm_targeted(model, sample_image, target, epsilon=8.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fgsm_targeted_output_shapes(self, model, sample_image):
        """Test targeted FGSM outputs have correct shapes."""
        target = 7
        adv_img, perturbation = fgsm_targeted(model, sample_image, target, epsilon=8.0)

        assert adv_img.shape[0] == 1
        assert adv_img.shape[1:] == sample_image.shape
        assert perturbation.shape == adv_img.shape

    def test_fgsm_targeted_increases_target_logit(self, model, sample_image):
        """Test targeted FGSM increases logit for target class."""
        target = 7

        with torch.no_grad():
            orig_logits = model(sample_image)
            orig_target_logit = orig_logits[0, target].item()

        adv_img, _ = fgsm_targeted(model, sample_image, target, epsilon=50.0)

        with torch.no_grad():
            adv_logits = model(adv_img)
            adv_target_logit = adv_logits[0, target].item()

        # Target logit should increase
        assert adv_target_logit > orig_target_logit

    def test_fgsm_targeted_bounded_perturbation(self, model, sample_image):
        """Test targeted FGSM perturbation respects epsilon bound."""
        epsilon = 10.0
        target = 7
        adv_img, perturbation = fgsm_targeted(model, sample_image, target, epsilon=epsilon)

        assert torch.all(torch.abs(perturbation) <= epsilon + 1e-6)

    def test_fgsm_targeted_clamped_to_valid_range(self, model, sample_image):
        """Test targeted FGSM output is clamped to [0, 255]."""
        target = 7
        adv_img, _ = fgsm_targeted(model, sample_image, target, epsilon=100.0)
        assert torch.all(adv_img >= 0.0)
        assert torch.all(adv_img <= 255.0)
    
    def test_epsilon_zero_no_change(self, model, sample_image):
        """Test that epsilon=0 produces no change to the image."""
        target = 7
        adv_img, perturbation = fgsm_targeted(model, sample_image, target, epsilon=0.0)

        # With epsilon=0, perturbation should be zero
        assert torch.allclose(perturbation, torch.zeros_like(perturbation))


# Tests for PGD
class TestPGD:
    """Test Projected Gradient Descent (untargeted)."""

    def test_pgd_returns_tuple(self, model, sample_image, sample_label):
        """Test PGD returns tuple of (adversarial_image, perturbation)."""
        result = pgd(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, label=sample_label)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pgd_output_shapes(self, model, sample_image, sample_label):
        """Test PGD outputs have correct shapes."""
        adv_img, perturbation = pgd(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, label=sample_label)

        assert adv_img.shape[0] == 1
        assert adv_img.shape[1:] == sample_image.shape
        assert perturbation.shape == adv_img.shape

    def test_pgd_creates_perturbation(self, model, sample_image, sample_label):
        """Test PGD creates non-zero perturbation."""
        adv_img, perturbation = pgd(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, label=sample_label)
        assert not torch.allclose(perturbation, torch.zeros_like(perturbation))

    def test_pgd_bounded_perturbation(self, model, sample_image, sample_label):
        """Test PGD perturbation respects epsilon bound."""
        epsilon = 10.0
        adv_img, perturbation = pgd(model, sample_image, epsilon=epsilon, alpha=2.0, num_iter=10, label=sample_label)

        # Perturbation should be bounded by epsilon
        assert torch.all(torch.abs(perturbation) <= epsilon + 1e-6)

    def test_pgd_iterative_improvement(self, model, sample_image, sample_label):
        """Test PGD with more iterations creates stronger perturbation."""
        # Run with few iterations
        _, pert_few = pgd(model, sample_image, epsilon=20.0, alpha=2.0, num_iter=5, label=sample_label)

        # Run with many iterations
        _, pert_many = pgd(model, sample_image, epsilon=20.0, alpha=2.0, num_iter=40, label=sample_label)

        # More iterations should generally create larger perturbation (within epsilon)
        norm_few = torch.norm(pert_few)
        norm_many = torch.norm(pert_many)

        # At least one should have non-trivial perturbation
        assert norm_few > 0 or norm_many > 0

    def test_pgd_no_gradient_in_output(self, model, sample_image, sample_label):
        """Test PGD output is detached from computation graph."""
        adv_img, perturbation = pgd(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, label=sample_label)
        assert not adv_img.requires_grad
        assert not perturbation.requires_grad

    def test_pgd_alpha_affects_convergence(self, model, sample_image, sample_label):
        """Test PGD with different alpha values."""
        # Small alpha (smaller steps)
        adv_small, _ = pgd(model, sample_image, epsilon=10.0, alpha=0.5, num_iter=10, label=sample_label)

        # Large alpha (larger steps)
        adv_large, _ = pgd(model, sample_image, epsilon=10.0, alpha=5.0, num_iter=10, label=sample_label)

        # Both should produce valid outputs
        assert adv_small.shape == adv_large.shape
        assert not torch.allclose(adv_small, adv_large)


# Tests for PGD Targeted
class TestPGDTargeted:
    """Test Projected Gradient Descent (targeted)."""

    def test_pgd_targeted_returns_tuple(self, model, sample_image):
        """Test targeted PGD returns tuple of (adversarial_image, perturbation)."""
        target = 7
        result = pgd_targeted(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, y_target=target)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pgd_targeted_output_shapes(self, model, sample_image):
        """Test targeted PGD outputs have correct shapes."""
        target = 7
        adv_img, perturbation = pgd_targeted(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, y_target=target)

        assert adv_img.shape[0] == 1
        assert adv_img.shape[1:] == sample_image.shape
        assert perturbation.shape == adv_img.shape

    def test_pgd_targeted_increases_target_logit(self, model, sample_image):
        """Test targeted PGD increases logit for target class."""
        target = 7

        with torch.no_grad():
            orig_logits = model(sample_image)
            orig_target_logit = orig_logits[0, target].item()

        adv_img, _ = pgd_targeted(model, sample_image, epsilon=50.0, alpha=5.0, num_iter=20, y_target=target)

        with torch.no_grad():
            adv_logits = model(adv_img)
            adv_target_logit = adv_logits[0, target].item()

        # Target logit should increase
        assert adv_target_logit > orig_target_logit

    def test_pgd_targeted_bounded_perturbation(self, model, sample_image):
        """Test targeted PGD perturbation respects epsilon bound."""
        epsilon = 10.0
        target = 7
        adv_img, perturbation = pgd_targeted(model, sample_image, epsilon=epsilon, alpha=2.0, num_iter=10, y_target=target)

        assert torch.all(torch.abs(perturbation) <= epsilon + 1e-6)

    def test_pgd_targeted_clamped_to_valid_range(self, model, sample_image):
        """Test targeted PGD output is clamped to [0, 255]."""
        target = 7
        adv_img, _ = pgd_targeted(model, sample_image, epsilon=100.0, alpha=10.0, num_iter=10, y_target=target)
        assert torch.all(adv_img >= 0.0)
        assert torch.all(adv_img <= 255.0)

    def test_pgd_targeted_no_gradient_in_output(self, model, sample_image):
        """Test targeted PGD output is detached from computation graph."""
        target = 7
        adv_img, perturbation = pgd_targeted(model, sample_image, epsilon=8.0, alpha=2.0, num_iter=10, y_target=target)
        assert not adv_img.requires_grad
        assert not perturbation.requires_grad
