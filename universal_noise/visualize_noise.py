#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_noise.py

Validates adversarial noise by computing FaceNet embedding distance
between original and noised versions of an image.

Usage:
    python validate_noise.py --noise best.pt --image face.jpg --epsilon 0.03
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download


def try_load_state_dict(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                model.load_state_dict(obj[key], strict=False)
                return True
        try:
            model.load_state_dict(obj, strict=False)
            return True
        except Exception:
            pass
    model.load_state_dict(obj, strict=False)
    return True


def load_facenet_from_hf(device, repo_id=None, filename=None, fallback_to_facenet_pytorch=True):
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    candidates = []
    if repo_id and filename:
        candidates.append((repo_id, filename))
    candidates.extend([
        ("py-feat/facenet", "facenet_20180402_114759_vggface2.pth"),
        ("lllyasviel/Annotators", "facenet.pth"),
    ])

    for rid, fname in candidates:
        try:
            print(f"[info] Trying HF repo '{rid}' file '{fname}'...")
            ckpt_path = hf_hub_download(repo_id=rid, filename=fname)
            if try_load_state_dict(model, ckpt_path):
                print(f"[info] Loaded FaceNet weights from HF: {rid}/{fname}")
                return model
        except Exception as e:
            print(f"[warn] Could not fetch {rid}/{fname}: {e}")

    if fallback_to_facenet_pytorch:
        print("[info] Falling back to facenet-pytorch pretrained='vggface2'.")
        return model

    raise RuntimeError("Failed to load FaceNet weights from the Hugging Face Hub.")


def preprocess_for_facenet(img_tensor):
    """Expects img_tensor in [0,1] range with shape (C, H, W) or (B, C, H, W)"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    x = F.interpolate(img_tensor, size=(160, 160), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def load_noise_checkpoint(noise_path, key="noise_bounded"):
    """Load noise tensor from checkpoint"""
    obj = torch.load(noise_path, map_location="cpu")

    if isinstance(obj, dict):
        if key in obj:
            noise = obj[key]
        elif "noise_bounded" in obj:
            noise = obj["noise_bounded"]
        elif "noise_raw" in obj:
            print(f"[warn] Using 'noise_raw' - you may need to apply bounded_noise() yourself")
            noise = obj["noise_raw"]
        else:
            raise KeyError(f"Could not find noise tensor in checkpoint keys: {list(obj.keys())}")
    elif isinstance(obj, torch.Tensor):
        noise = obj
    else:
        raise ValueError("Checkpoint does not contain a tensor or recognizable structure")

    return noise


def main():
    parser = argparse.ArgumentParser(
        description="Compute FaceNet embedding distance between original and noised image")
    parser.add_argument("--noise", type=str, required=True, help="Path to noise checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--epsilon", type=float, required=True, help="Noise epsilon scaling")
    parser.add_argument("--noise-key", type=str, default="noise_bounded",
                        help="Key in checkpoint for noise tensor")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-filename", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    # Load model
    print("[info] Loading FaceNet model...")
    model = load_facenet_from_hf(device, repo_id=args.hf_repo_id, filename=args.hf_filename)

    # Load noise
    print(f"[info] Loading noise from {args.noise}...")
    noise_tensor = load_noise_checkpoint(args.noise, key=args.noise_key).to(device)
    print(f"[info] Noise shape: {noise_tensor.shape}")

    # Load image
    print(f"[info] Loading image from {args.image}...")
    img = Image.open(args.image).convert("RGB")
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).to(device)  # [0,1] range

    # Resize noise to match image dimensions
    h, w = img_tensor.shape[1:]
    noise_resized = F.interpolate(
        noise_tensor.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # Apply noise
    img_noised = torch.clamp(img_tensor + noise_resized * args.epsilon, 0.0, 1.0)

    # Compute embeddings
    print("[info] Computing embeddings...")
    with torch.no_grad():
        x_clean = preprocess_for_facenet(img_tensor)
        x_noised = preprocess_for_facenet(img_noised)

        emb_clean = model(x_clean)
        emb_noised = model(x_noised)

        dist = torch.norm(emb_clean - emb_noised, p=2).item()

    print(f"\nEmbedding distance: {dist:.4f}")


if __name__ == "__main__":
    main()
