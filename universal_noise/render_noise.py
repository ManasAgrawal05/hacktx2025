#!/usr/bin/env python3
"""
Simple noise application script.

Usage:
    python apply_noise.py --image photo.jpg --noise noise.pt --epsilon 0.03
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


def apply_noise(image_path, noise_path, epsilon, noise_key="noise_bounded"):
    """Load image and noise, apply perturbation, save both versions."""

    # Load noise tensor
    obj = torch.load(noise_path, map_location="cpu")
    if isinstance(obj, dict):
        noise = obj[noise_key]
    else:
        noise = obj

    # Load and convert image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Resize noise to match image
    noise = noise.unsqueeze(0)
    noise_resized = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
    noise_resized = noise_resized.squeeze(0)

    # Apply perturbation
    perturbed = torch.clamp(img_tensor + noise_resized, 0, 1)

    # Save original
    img.save("orig.png")

    # Save perturbed
    perturbed_np = (perturbed.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(perturbed_np).save("perturbed.png")

    print("Saved: orig.png, perturbed.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--noise", required=True, help="Noise .pt file path")
    parser.add_argument("--epsilon", type=float, required=True, help="Perturbation strength")
    parser.add_argument("--noise-key", default="noise_bounded", help="Key in .pt file (if dict)")
    args = parser.parse_args()

    apply_noise(args.image, args.noise, args.epsilon, args.noise_key)
