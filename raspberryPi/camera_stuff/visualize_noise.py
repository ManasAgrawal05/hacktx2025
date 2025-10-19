#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_noise.py

Usage:
    python render_noise.py --input noise.pt --output out.png
    python render_noise.py --input noise.pt --output out_gray.png --neutral-gray --epsilon 0.03
    python render_noise.py --input noise.pt --overlay image.png --output overlay.png --epsilon 0.03

Description:
    Loads a .pt file containing a tensor (e.g., noise_bounded) and renders it as a PNG image.
    Options:
      - Default: min→black, max→white
      - --neutral-gray: visualize around 0.5 ± ε
      - --overlay: overlay the noise on top of a reference image,
                   showing original (left) vs modified (right) with divider.
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


def tensor_to_image(tensor, neutral_gray=False, epsilon=None):
    """Convert tensor to an Image object (8-bit RGB or grayscale)."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().float()
    else:
        raise TypeError("Input must be a torch.Tensor")

    if neutral_gray:
        if epsilon is None:
            raise ValueError("--epsilon must be specified when using --neutral-gray")
        img = 0.5 + tensor * epsilon
        img = img.clamp(0.0, 1.0)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        if min_val == max_val:
            print("[warn] Tensor constant; output mid-gray.")
            img = torch.zeros_like(tensor) + 0.5
        else:
            img = (tensor - min_val) / (max_val - min_val)

    img_np = img.numpy()

    if img_np.ndim == 3:
        if img_np.shape[0] == 1:
            img_np = np.squeeze(img_np, axis=0)
            mode = "L"
        elif img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            mode = "RGB"
        else:
            raise ValueError(f"Unsupported channel count: {img_np.shape[0]}")
    elif img_np.ndim == 2:
        mode = "L"
    else:
        raise ValueError(f"Unexpected tensor shape: {img_np.shape}")

    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8, mode=mode)

def make_overlay(base_img, noise_tensor, epsilon, device="cpu"):
    """
    Overlay noise on the base image by adding the scaled noise,
    return only the noisy image (not side-by-side).
    """
    base = base_img.convert("RGB")
    w, h = base.size

    # Convert base image to tensor in [0,1]
    base_tensor = torch.from_numpy(np.array(base)).permute(2, 0, 1).float() / 255.0
    base_tensor = base_tensor.unsqueeze(0).to(device)  # shape: (1, 3, H, W)

    # Resize noise to match base image size
    noise_tensor = noise_tensor.unsqueeze(0).to(device)  # shape: (1, C, h_noise, w_noise)
    noise_resized = torch.nn.functional.interpolate(
        noise_tensor, size=(h, w), mode="bilinear", align_corners=False
    )
    noise_resized = noise_resized.squeeze(0)  # shape: (C, H, W)

    # Apply noise with epsilon scaling, clamp to [0,1]
    perturbed = torch.clamp(base_tensor + noise_resized.unsqueeze(0) * epsilon, 0, 1)
    perturbed = perturbed.squeeze(0)  # shape: (3, H, W)

    # Convert back to PIL Image
    perturbed_np = (perturbed.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
    noisy_img = Image.fromarray(perturbed_np, mode="RGB")

    return noisy_img


def main():
    parser = argparse.ArgumentParser(description="Render a .pt tensor as an image.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output PNG file")
    parser.add_argument("--key", "-k", type=str, default=None,
                        help="Optional key inside checkpoint (e.g., 'noise_bounded')")
    parser.add_argument("--neutral-gray", action="store_true",
                        help="Visualize noise as gray background ± epsilon")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Epsilon scaling factor (required if --neutral-gray or --overlay)")
    parser.add_argument("--overlay", type=str, default=None,
                        help="Optional path to base image for overlay visualization")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for resizing (default: cpu)")
    args = parser.parse_args()

    obj = torch.load(args.input, map_location="cpu")
    if isinstance(obj, dict) and args.key:
        if args.key not in obj:
            raise KeyError(f"Key '{args.key}' not found in checkpoint keys: {list(obj.keys())}")
        tensor = obj[args.key]
    elif isinstance(obj, dict) and "noise_bounded" in obj:
        print("[info] Auto-detected key 'noise_bounded'")
        tensor = obj["noise_bounded"]
    elif isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        raise ValueError("Input file does not contain a tensor or recognizable structure.")

    # If overlay mode
    if args.overlay:
        if args.epsilon is None:
            raise ValueError("--epsilon must be specified when using --overlay")
        base_img = Image.open(args.overlay)
        combined_img = make_overlay(base_img, tensor, epsilon=args.epsilon, device=args.device)
        combined_img.save(args.output)
        print(f"[info] Saved overlay comparison image to {args.output}")
    else:
        img = tensor_to_image(tensor, neutral_gray=args.neutral_gray, epsilon=args.epsilon)
        img.save(args.output)
        print(f"[info] Saved image to {args.output}")


if __name__ == "__main__":
    main()
