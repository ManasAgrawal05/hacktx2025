import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def make_overlay(base_img, noise_tensor, epsilon):
    """
    Apply noise overlay to a base image.

    Args:
        base_img: PIL Image in RGB format
        noise_tensor: PyTorch tensor of shape (C, H, W) containing noise pattern
        epsilon: Scaling factor for the noise (typically 0.01 - 0.1)

    Returns:
        PIL Image with noise applied
    """
    # Convert PIL Image to tensor: (H, W, C) -> (C, H, W), float [0, 1]
    img_np = np.array(base_img.convert("RGB"))
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    # Get image dimensions
    _, h, w = img_tensor.shape

    # Resize noise to match image dimensions
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    noise_batched = noise_tensor.unsqueeze(0)
    noise_resized = F.interpolate(
        noise_batched,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    # Remove batch dimension: (1, C, H, W) -> (C, H, W)
    noise_resized = noise_resized.squeeze(0)

    # Apply noise scaled by epsilon
    perturbed = img_tensor + (noise_resized)

    # Clamp to valid range [0, 1]
    perturbed = torch.clamp(perturbed, 0, 1)

    # Convert back to PIL Image: (C, H, W) -> (H, W, C), uint8 [0, 255]
    perturbed_np = (perturbed.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return Image.fromarray(perturbed_np)
