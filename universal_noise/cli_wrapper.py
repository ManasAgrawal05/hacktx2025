#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facenet_distance.py
Command-line tool to compute FaceNet embedding distance (Euclidean) between two images.
Usage:
    python facenet_distance.py img1.jpg img2.jpg [--device cuda]
"""
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download

# --- Utility: load pretrained FaceNet model ---


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

# --- Image preprocessing: MATCH TRAINING EXACTLY ---


def preprocess_for_facenet_TRAINING_STYLE(img):
    """
    Mimics the exact preprocessing from train_facenet_noise.py:
    1. Resize to 224x224 using PIL (from dataset_simple.py)
    2. Convert to tensor [0,1]
    3. Resize to 160x160 using torch bilinear with align_corners=False
    4. Normalize to [-1,1]
    """
    # Step 1: PIL resize to 224x224 (as done in dataset_simple.py)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0)  # Shape: [1, 3, 224, 224], range [0,1]

    # Step 2: Torch interpolate to 160x160 (as done in train_facenet_noise.py)
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)

    # Step 3: Normalize to [-1,1]
    x = x * 2.0 - 1.0

    return x


def preprocess_for_facenet_ORIGINAL(img):
    """
    Original preprocessing from this script (single resize, PIL only).
    """
    tfm = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    return tfm(img).unsqueeze(0)

# --- Main logic ---


def compute_distance(img1_path, img2_path, device, use_training_preprocess=False):
    model = load_facenet_from_hf(device)
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    if use_training_preprocess:
        print("[info] Using TRAINING preprocessing: 224x224 (PIL) → 160x160 (torch)")
        x1 = preprocess_for_facenet_TRAINING_STYLE(img1).to(device)
        x2 = preprocess_for_facenet_TRAINING_STYLE(img2).to(device)
    else:
        print("[info] Using ORIGINAL preprocessing: 160x160 (PIL)")
        x1 = preprocess_for_facenet_ORIGINAL(img1).to(device)
        x2 = preprocess_for_facenet_ORIGINAL(img2).to(device)

    with torch.no_grad():
        e1 = model(x1)
        e2 = model(x2)
        dist = torch.norm(e1 - e2, p=2).item()

    return dist

# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Compute FaceNet embedding Euclidean distance between two images.")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Computation device")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-filename", type=str, default=None)
    parser.add_argument(
        "--use-training-preprocess",
        action="store_true",
        help="Use the training preprocessing (224→160) instead of direct 160x160 resize")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    dist = compute_distance(args.img1, args.img2, device, args.use_training_preprocess)
    print(f"Euclidean distance: {dist:.4f}")


if __name__ == "__main__":
    main()
