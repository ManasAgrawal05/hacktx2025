#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
facenet_similarity.py

Command-line tool to compute FaceNet embedding similarity between two images.

Usage:
    python facenet_similarity.py img1.jpg img2.jpg [--device cuda]
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


# --- Image preprocessing ---
def preprocess_for_facenet(img):
    tfm = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    return tfm(img).unsqueeze(0)


# --- Main logic ---
def compute_similarity(img1_path, img2_path, device):
    model = load_facenet_from_hf(device)
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    x1 = preprocess_for_facenet(img1).to(device)
    x2 = preprocess_for_facenet(img2).to(device)

    with torch.no_grad():
        e1 = model(x1)
        e2 = model(x2)
        sim = F.cosine_similarity(e1, e2).item()
    return sim


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Compute FaceNet embedding similarity between two images.")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[
            None,
            "cpu",
            "cuda"],
        help="Computation device")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-filename", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    sim = compute_similarity(args.img1, args.img2, device)
    print(f"Cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
