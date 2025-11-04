"""
Tests for interpretability methods in src/misc/interp.py

Run with: pytest tests/test_interp.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from misc.interp import saliency_map, grad_cam
from models.torch_lenet import TorchLeNet


@pytest.fixture
def model():
    """Create a LeNet model for testing."""
    model = TorchLeNet(act_fn='relu', init='kaiming')
    model.eval()
    return model


@pytest.fixture
def sample_image():
    """Create a sample normalized image tensor (C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(1, 28, 28)


@pytest.fixture
def sample_image_batch():
    """Create a batch of sample normalized images (N, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def sample_label():
    """Create a sample label."""
    return 3


# Tests for saliency_map
class TestSaliencyMap:
    """Test gradient-based saliency map generation."""

    def test_saliency_map_returns_tensor(self, model, sample_image, sample_label):
        """Test saliency_map returns a tensor."""
        result = saliency_map(model, sample_image, sample_label)
        assert isinstance(result, torch.Tensor)

    def test_saliency_map_output_shape(self, model, sample_image, sample_label):
        """Test saliency_map output has correct shape."""
        result = saliency_map(model, sample_image, sample_label)
        # Should return (H, W) - channel dimension squeezed out
        assert result.ndim == 2
        assert result.shape == sample_image.shape[-2:]  # H, W dimensions

    def test_saliency_map_normalized_range(self, model, sample_image, sample_label):
        """Test saliency_map output is normalized to [0, 1]."""
        result = saliency_map(model, sample_image, sample_label)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_saliency_map_no_gradients(self, model, sample_image, sample_label):
        """Test saliency_map output is detached from computation graph."""
        result = saliency_map(model, sample_image, sample_label)
        assert not result.requires_grad

    def test_saliency_map_on_cpu(self, model, sample_image, sample_label):
        """Test saliency_map returns CPU tensor."""
        result = saliency_map(model, sample_image, sample_label)
        assert result.device.type == 'cpu'

    def test_saliency_map_with_batch_input(self, model, sample_image_batch, sample_label):
        """Test saliency_map handles 4D input (with batch dimension)."""
        # Take first image from batch
        single_img = sample_image_batch[0]
        result = saliency_map(model, single_img, sample_label)
        # Should return (H, W) - channel dimension squeezed out
        assert result.ndim == 2
        assert result.shape == single_img.shape[-2:]

    def test_saliency_map_grad_times_input(self, model, sample_image, sample_label):
        """Test saliency_map with grad_times_input=True."""
        result_grad = saliency_map(model, sample_image, sample_label, grad_times_input=False)
        result_grad_input = saliency_map(model, sample_image, sample_label, grad_times_input=True)

        # Both should have same shape
        assert result_grad.shape == result_grad_input.shape

        # Results should generally be different
        assert not torch.allclose(result_grad, result_grad_input)

    def test_saliency_map_model_in_eval_mode(self, model, sample_image, sample_label):
        """Test that saliency_map puts model in eval mode."""
        model.train()  # Set to train mode
        assert model.training

        _ = saliency_map(model, sample_image, sample_label)
        # Model should be in eval mode after saliency_map call
        assert not model.training

    def test_saliency_map_deterministic(self, model, sample_image, sample_label):
        """Test saliency_map produces deterministic results."""
        result1 = saliency_map(model, sample_image, sample_label)
        result2 = saliency_map(model, sample_image, sample_label)
        assert torch.allclose(result1, result2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_saliency_map_with_gpu_model(self, model, sample_image, sample_label):
        """Test saliency_map with model on GPU."""
        model_gpu = model.to('cuda')
        result = saliency_map(model_gpu, sample_image, sample_label)

        # Result should be on CPU
        assert result.device.type == 'cpu'
        assert result.ndim == 2
        assert result.shape == sample_image.shape[-2:]


# Tests for grad_cam
class TestGradCAM:
    """Test Grad-CAM heatmap generation."""

    def test_grad_cam_returns_tensor(self, model, sample_image):
        """Test grad_cam returns a tensor."""
        result = grad_cam(model, sample_image, target_layer='conv1')
        assert isinstance(result, torch.Tensor)

    def test_grad_cam_output_shape_single_image(self, model, sample_image):
        """Test grad_cam output shape for single image."""
        result = grad_cam(model, sample_image, target_layer='conv1')
        # Should return (N, H, W) where N=1 for single image
        assert result.ndim == 3
        assert result.shape[0] == 1
        assert result.shape[1] == sample_image.shape[-2]  # Height
        assert result.shape[2] == sample_image.shape[-1]  # Width

    def test_grad_cam_output_shape_batch(self, model, sample_image_batch):
        """Test grad_cam output shape for batch of images."""
        result = grad_cam(model, sample_image_batch, target_layer='conv1')
        # Should return (N, H, W)
        assert result.ndim == 3
        assert result.shape[0] == sample_image_batch.shape[0]
        assert result.shape[1] == sample_image_batch.shape[-2]
        assert result.shape[2] == sample_image_batch.shape[-1]

    def test_grad_cam_normalized_range(self, model, sample_image):
        """Test grad_cam output is normalized to [0, 1] when normalize=True."""
        result = grad_cam(model, sample_image, target_layer='conv1', normalize=True)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_grad_cam_no_gradients(self, model, sample_image):
        """Test grad_cam output is detached from computation graph."""
        result = grad_cam(model, sample_image, target_layer='conv1')
        assert not result.requires_grad

    def test_grad_cam_target_layer_as_string(self, model, sample_image):
        """Test grad_cam with target_layer as string."""
        result = grad_cam(model, sample_image, target_layer='conv1')
        assert result.shape[0] == 1
        assert result.shape[1:] == sample_image.shape[-2:]

    def test_grad_cam_target_layer_as_module(self, model, sample_image):
        """Test grad_cam with target_layer as module reference."""
        result = grad_cam(model, sample_image, target_layer=model.conv1)
        assert result.shape[0] == 1
        assert result.shape[1:] == sample_image.shape[-2:]

    def test_grad_cam_auto_target_class(self, model, sample_image):
        """Test grad_cam with automatic target class selection (argmax)."""
        result = grad_cam(model, sample_image, target_layer='conv1', target_class=None)
        assert result.shape[0] == 1
        assert result.shape[1:] == sample_image.shape[-2:]

    def test_grad_cam_different_layers(self, model, sample_image):
        """Test grad_cam on different convolutional layers."""
        result_conv1 = grad_cam(model, sample_image, target_layer='conv1')
        result_conv2 = grad_cam(model, sample_image, target_layer='conv2')

        # Both should have same output shape (upsampled to input size)
        assert result_conv1.shape == result_conv2.shape

        # But values should differ
        assert not torch.allclose(result_conv1, result_conv2)

    def test_grad_cam_deterministic(self, model, sample_image):
        """Test grad_cam produces deterministic results."""
        result1 = grad_cam(model, sample_image, target_layer='conv1')
        result2 = grad_cam(model, sample_image, target_layer='conv1')
        assert torch.allclose(result1, result2)

    def test_grad_cam_model_in_eval_mode(self, model, sample_image):
        """Test that grad_cam puts model in eval mode."""
        model.train()  # Set to train mode
        assert model.training

        _ = grad_cam(model, sample_image, target_layer='conv1')
        # Model should be in eval mode after grad_cam call
        assert not model.training

    def test_grad_cam_batch_processing(self, model, sample_image_batch):
        """Test grad_cam processes batches correctly."""
        batch_size = sample_image_batch.shape[0]
        result = grad_cam(model, sample_image_batch, target_layer='conv1')

        # Should produce one CAM per image
        assert result.shape[0] == batch_size

        # Each CAM should be valid
        for i in range(batch_size):
            cam = result[i]
            assert torch.all(cam >= 0.0) and torch.all(cam <= 1.0)

    def test_grad_cam_batch_vs_individual(self, model, sample_image_batch):
        """Test grad_cam batch processing matches individual processing."""
        # Process batch
        result_batch = grad_cam(model, sample_image_batch, target_layer='conv1', target_class=5)

        # Process individually
        results_individual = []
        for i in range(sample_image_batch.shape[0]):
            cam = grad_cam(model, sample_image_batch[i:i+1], target_layer='conv1', target_class=5)
            results_individual.append(cam)

        result_individual_stacked = torch.cat(results_individual, dim=0)

        # Results should be very close (small numerical differences acceptable)
        assert torch.allclose(result_batch, result_individual_stacked, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grad_cam_with_gpu_model(self, model, sample_image):
        """Test grad_cam with model on GPU."""
        model_gpu = model.to('cuda')
        sample_gpu = sample_image.to('cuda')
        result = grad_cam(model_gpu, sample_gpu, target_layer='conv1')

        # Result should maintain device consistency
        assert result.shape[0] == 1
        assert result.shape[1:] == sample_image.shape[-2:]
