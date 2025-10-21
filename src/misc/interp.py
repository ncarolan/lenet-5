'''
Interpretability techniques and associated helper methods for LeNet-5.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

def saliency_map(model, x, y, grad_times_input=False, epsilon=1e-8) -> torch.Tensor:
    """
    Returns a gradient-based saliency map (∂ score_y / ∂ x) for a given model, input, and label.

    Args:
        model (nn.Module): Classifier
        x (tensor): CxHxW normalized image
        y (int): Class index (target label)
        grad_times_input (bool): If true, returns a gradient x input saliency map. (input * ∂ score_y / ∂ x)
        epsilon (float): Small value to prevent division by zero during normalization. Default: 1e-8

    Returns:
        saliency (torch.Tensor): Saliency map.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.clone().detach().to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)  # Add batch dim if needed
    y = torch.tensor([int(y)], device=device)

    x.requires_grad_(True)  # Enable input gradients
    model.zero_grad(set_to_none=True)  # Set gradients to 0

    with torch.enable_grad():
        logits = model(x)
        score_y = logits.gather(1, y.view(-1,1)).squeeze()
        score_y.backward()

    saliency = x.grad
    if grad_times_input:
        saliency = saliency * x.detach()
    saliency = saliency.detach().abs().squeeze()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + epsilon)  # Normalize to [0,1] for visualization
    return saliency.cpu()

def grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: Union[str, torch.nn.Module],
    target_class: Optional[int] = None,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Computes a Grad-CAM heatmap for a given model and input.

    Args:
        model (nn.Module): Classifier
        input_tensor (torch.Tensor): Tensor of shape (N, C, H, W). N can be 1+ (returns one CAM per item).
        target_layer (str or nn.Module): Layer or name of the layer to probe (e.g. 'layer4.2.relu').
        target_class (int): Class index to explain. If None, uses model's argmax per item.
        normalize (bool): If True, min-max normalize CAMs to [0, 1] per item.
        epsilon (float): Small value to prevent division by zero during normalization. Default: 1e-8

    Returns:
        cam (torch.Tensor): Heatmap tensor of shape (N, H, W).
    """
    model.eval()

    # Resolve target layer if passed as string
    if isinstance(target_layer, str):
        module = model
        for attr in target_layer.split('.'):
            if attr.isdigit():
                module = module[int(attr)]  # handle Sequential/ModuleList index
            else:
                module = getattr(module, attr)
        target_module = module
    else:
        target_module = target_layer

    activations = []
    gradients = []

    def fwd_hook(_m, _i, o):
        activations.append(o.detach())

    def bwd_hook(_m, gi, go):
        # go is a tuple; grab grad wrt output (same shape as activation)
        gradients.append(go[0].detach())

    # Register hooks
    fwd_handle = target_module.register_forward_hook(fwd_hook)
    bwd_handle = target_module.register_full_backward_hook(bwd_hook)

    # Forward
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim if needed
    logits = model(input_tensor)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)  # (N, num_classes)

    # Determine target indices per item
    if target_class is None:
        target_idxs = logits.argmax(dim=1)
    else:
        target_idxs = torch.full((logits.shape[0],), int(target_class), device=logits.device, dtype=torch.long)

    gather = logits.gather(1, target_idxs.view(-1, 1)).sum()
    # Backward (compute grads wrt target layer)
    model.zero_grad(set_to_none=True)
    gather.backward(retain_graph=False)

    # Clean up hooks
    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Failed to capture hooks; check target_layer is used by the forward pass.")

    A = activations[-1]           # (N, C, h, w)
    dY_dA = gradients[-1]         # (N, C, h, w)

    # Compute channel weights via global average pooling of gradients
    weights = dY_dA.mean(dim=(2, 3), keepdim=True)      # (N, C, 1, 1)

    # Weighted combination and ReLU
    cam = (weights * A).sum(dim=1)                      # (N, h, w)
    cam = F.relu(cam)

    # Upsample to input spatial size
    H, W = input_tensor.shape[-2:]
    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

    if normalize:
        # Normalize per item to [0,1]
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        denom = (cam_max - cam_min).clamp(min=epsilon)
        cam = (cam - cam_min) / denom

    return cam.detach()