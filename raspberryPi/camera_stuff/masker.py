"""Overlay the universal mask on a passed in jpg image"""

import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import camera_stuff.camera_stuff

class masker:
    def __init__(self, mask_path):
        self.mask_tensor = mask_path # Path to the tensor mask
        pass

    """Applies universal mask to the jpg and returns new jpg"""
    def make_overlay(base_img, noise_tensor, epsilon, device="cpu"):
        """Overlay noise on image: show original (left) vs modified (right) side by side."""
        base = base_img.convert("RGB")
        w, h = base.size

        # Convert base image to tensor in [0,1]
        base_tensor = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255.0
        base_tensor = base_tensor.unsqueeze(0).to(device)

        # Resize noise to match base
        noise_tensor = noise_tensor.unsqueeze(0).to(device)
        noise_resized = F.interpolate(noise_tensor, size=(h, w), mode="bilinear", align_corners=False)
        noise_resized = noise_resized.squeeze(0)

        # Apply perturbation: clamp to [0,1]
        perturbed = torch.clamp(base_tensor + noise_resized.unsqueeze(0) * epsilon, 0, 1)
        perturbed = perturbed.squeeze(0)

        orig_np = (base_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pert_np = (perturbed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Combine side-by-side with divider
        divider_width = 5
        combined = np.ones((h, w * 2 + divider_width, 3), dtype=np.uint8) * 255
        combined[:, :w] = orig_np
        combined[:, w:w + divider_width] = 0  # black divider
        combined[:, w + divider_width:] = pert_np

        return Image.fromarray(combined, mode="RGB")