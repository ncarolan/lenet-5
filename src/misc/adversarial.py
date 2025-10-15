'''
Adversarial example generation and associated helper methods for LeNet-5.
'''

import numpy as np
import torch
import torch.nn as nn

def l1_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes the L1 (Manhattan) distance between two images."""
    return np.sum(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))

def l1_distance(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Computes the L1 (Manhattan) distance between two images. """
    return torch.sum(torch.abs(img1.float() - img2.float()))

def l2_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the L2 (Euclidean) distance between two images."""
    return np.sqrt(np.sum((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

def l2_distance(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Computes the L2 (Euclidean) distance between two images."""
    return torch.sqrt(torch.sum((img1.float() - img2.float()) ** 2))

def fgsm(model: torch.nn.Module, image: torch.Tensor, label: int, epsilon: float):
    """
    Returns an adversarial example found with the fast gradient sign method.

    This technique was introduced in `Explaining and Harnessing Adversarial Examples` (https://arxiv.org/abs/1412.6572).

    Args:
        model (nn.Module)
        image (torch.Tensor)
        label (int): True label of image
        epsilon (float): Magnitude of attack

    Returns:
        adversarial_image, perturbation (Tuple[torch.Tensor, torch.Tensor]): Modified image and the adversarial perturbation.
    """
    if image.ndim == 3:      
        image = image.unsqueeze(0)  # Add batch dim if needed
    image = image.clone().detach().requires_grad_(True)

    model.zero_grad()
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, torch.tensor([label]))
    loss.backward()

    perturbation = epsilon * torch.sign(image.grad)
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 256)
    
    return adversarial_image.detach(), perturbation.detach()

def fgsm_targeted(model: torch.nn.Module, image: torch.Tensor, target: int, epsilon: float):
    """
    Returns a targeted adversarial example found with the fast gradient sign method.

    Args:
        model (nn.Module)
        image (torch.Tensor)
        target (int): Target class index to maximize
        epsilon (float): Magnitude of attack

    Returns:
        adversarial_image, perturbation (torch.Tensor, torch.Tensor)
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add batch dim if needed
    image = image.clone().detach().requires_grad_(True)

    model.zero_grad()
    outputs = model(image)
    loss = outputs[:, target].sum()   # Maximize target logit
    loss.backward()

    perturbation = epsilon * torch.sign(image.grad.data)
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 256)

    return adversarial_image.detach(), perturbation.detach()


def pgd(model: torch.nn.Module, image: torch.Tensor, epsilon: float, alpha: float, num_iter: int, label: int):
    """ 
    Returns a untargeted adversarial example found with projected gradient descent.

    Args:
        model (nn.Module): Classifier
        image (torch.Tensor): Starter image
        epsilon (float): Maximum allowed perturbation size
        alpha (float): "Learning rate" / step size
        num_iter (int): Number of iterations of gradient descent (ascent)
        label (int): True label of image

    Returns:
        adversarial_image, perturbation (Tuple[torch.Tensor, torch.Tensor]): Modified image and the adversarial perturbation.
    """
    if image.ndim == 3:      
        image = image.unsqueeze(0)  # Add batch dim if needed
    image = image.clone().detach().requires_grad_(True)
    
    delta = torch.zeros_like(image, requires_grad=True)
    for t in range(num_iter):
        yp = model(image + delta)
        loss = nn.CrossEntropyLoss(reduction='sum')(yp, torch.tensor([label]))
        model.zero_grad()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    adversarial_image = image + delta
    return adversarial_image.detach(), delta.detach()

def pgd_targeted(model: torch.nn.Module, image: torch.Tensor, epsilon: float, alpha: float, num_iter: int, y_target: int):
    """ 
    Returns a targeted adversarial example found with projected gradient descent.

    Referencing https://adversarial-ml-tutorial.org/adversarial_examples/

    Args:
        model (nn.Module)
        image (torch.Tensor)
        epsilon (float): Maximum allowed perturbation size
        alpha (float): "Learning rate" / step size
        num_iter (int): Number of iterations of gradient descent (ascent)
        y_target (int): Targeted class label prediction after adversarial manipulation

    Returns:
        adversarial_image, perturbation (Tuple[torch.Tensor, torch.Tensor]): Modified image and the adversarial perturbation.
    """
    if image.ndim == 3:      
        image = image.unsqueeze(0)  # Add batch dim if needed
    image = image.clone().detach().requires_grad_(True)
    
    delta = torch.zeros_like(image, requires_grad=True)
    for t in range(num_iter):
        yp = model(image + delta)
        loss = 2*yp[:,y_target].sum() - yp.sum()
        model.zero_grad()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    adversarial_image = image + delta
    return adversarial_image.detach(), delta.detach()