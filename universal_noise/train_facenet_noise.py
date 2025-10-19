#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_facenet_noise.py

Adversarial noise optimization for FaceNet embeddings, using a frozen model.

Workflow per batch:
  1) Encode raw images with frozen FaceNet -> E_clean
  2) Add learnable noise (bounded by epsilon) to images
  3) Encode noised images -> E_noised
  4) Maximize L2 distance ||E_clean - E_noised||_2 (equivalently minimize -distance)

Notes:
- Uses your dataset_simple.build_dataloaders() with default params.
- Noise is a learnable 3x250x250 tensor, squashed with tanh and scaled by epsilon,
  then resized to the current image shape before adding to the batch.
- Model is frozen (no grad on weights), but we allow gradients to flow through
  the forward pass w.r.t. inputs so the noise can learn.
"""

import os
import csv
import json
import math
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd

# Your dataloader module (provided in the prompt)
from dataset_simple import build_dataloaders

# FaceNet architecture
from facenet_pytorch import InceptionResnetV1

# Hugging Face Hub (for checkpoint download)
from huggingface_hub import hf_hub_download


def try_load_state_dict(model: nn.Module, ckpt_path: str) -> bool:
    """
    Attempt to load a state_dict from a ckpt into the given model.
    Returns True if successful, False otherwise.
    Handles a few common wrapping patterns like {'state_dict': ...}.
    """
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            # Try common key names
            for key in ["state_dict", "model_state_dict", "model", "weights"]:
                if key in obj and isinstance(obj[key], dict):
                    sd = obj[key]
                    model.load_state_dict(sd, strict=False)
                    return True
            # Might already be a state dict
            try:
                model.load_state_dict(obj, strict=False)
                return True
            except Exception:
                pass
        # Fall back to strict load if it's actually a state dict
        model.load_state_dict(obj, strict=False)
        return True
    except Exception as e:
        print(f"[warn] Failed to load state_dict from '{ckpt_path}': {e}")
        return False


def load_facenet_from_hf(
    device: torch.device,
    repo_id: str = None,
    filename: str = None,
    fallback_to_facenet_pytorch: bool = True,
) -> InceptionResnetV1:
    """
    Try to instantiate FaceNet (InceptionResnetV1) and load weights from HF Hub.
    If HF load fails and fallback is enabled, use facenet-pytorch pretrained weights.

    Returns a model set to eval() with requires_grad=False for all params.
    """
    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval()
    model.to(device)

    # Freeze model weights
    for p in model.parameters():
        p.requires_grad = False

    tried_any = False
    # If explicit repo/filename are given, try those first
    candidates = []
    if repo_id and filename:
        candidates.append((repo_id, filename))

    # Known mirrors (may change over time; strict=False load helps)
    candidates.extend([
        # A commonly mirrored VGGFace2 FaceNet-style checkpoint (name may vary)
        ("py-feat/facenet", "facenet_20180402_114759_vggface2.pth"),
        # Another mirror used in tooling repos (architecture may or may not match)
        ("lllyasviel/Annotators", "facenet.pth"),
    ])

    for rid, fname in candidates:
        try:
            print(f"[info] Trying HF repo '{rid}' file '{fname}'...")
            ckpt_path = hf_hub_download(repo_id=rid, filename=fname)
            tried_any = True
            if try_load_state_dict(model, ckpt_path):
                print(f"[info] Loaded FaceNet weights from HF: {rid}/{fname}")
                return model
        except Exception as e:
            print(f"[warn] Could not fetch {rid}/{fname} from HF: {e}")

    if fallback_to_facenet_pytorch:
        # Use packaged weights from facenet-pytorch as a last resort
        print("[info] Falling back to facenet-pytorch pretrained='vggface2'.")
        model_fallback = InceptionResnetV1(pretrained="vggface2", classify=False)
        model_fallback.eval().to(device)
        for p in model_fallback.parameters():
            p.requires_grad = False
        return model_fallback

    if not tried_any:
        raise RuntimeError("No HF candidates were attempted. Provide a valid repo_id and filename.")
    raise RuntimeError("Failed to load FaceNet weights from the Hugging Face Hub.")


def preprocess_for_facenet(x: torch.Tensor) -> torch.Tensor:
    """
    FaceNet (InceptionResnetV1) typically expects 160x160 RGB, normalized to [-1, 1].
    x: float tensor in [0,1], shape [B,3,H,W]
    """
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    x = x * 2.0 - 1.0
    return x


def bounded_noise(noise_raw: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Map unconstrained noise_raw -> bounded noise in [-epsilon, +epsilon],
    centered around zero via tanh.
    """
    return epsilon * torch.tanh(noise_raw)


def train(
    root_dir: str,
    out_root: str,
    num_epochs: int,
    epsilon: float,
    lr: float,
    seed: int,
    device: str,
    repo_id: str = None,
    filename: str = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    # --- Data ---
    # Keep default parameters for dataloaders (as requested)
    train_loader, _ = build_dataloaders(root_dir)

    # --- Model ---
    model = load_facenet_from_hf(device=device, repo_id=repo_id, filename=filename)

    # --- Learnable noise: 3 x 250 x 250 ---
    noise_raw = nn.Parameter(torch.zeros(3, 250, 250))
    # Small random init to help escape symmetry
    nn.init.normal_(noise_raw, mean=0.0, std=1e-3)
    noise_raw = noise_raw.to(device)

    optimizer = torch.optim.Adam([noise_raw], lr=lr)

    # --- Output dirs & bookkeeping ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"facenet_noise_run_{ts}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = {
        "root_dir": root_dir,
        "num_epochs": num_epochs,
        "epsilon": epsilon,
        "lr": lr,
        "seed": seed,
        "device": str(device),
        "repo_id": repo_id,
        "filename": filename,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    history_rows = []  # (epoch, step_idx, loss, distance)

    try:
        for epoch in range(1, num_epochs + 1):
            model.eval()  # keep model in eval mode (frozen)

            running_loss = 0.0
            running_dist = 0.0
            steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=120)
            for step, (imgs, _persons) in enumerate(pbar, start=1):
                imgs = imgs.to(device).float()  # in [0,1], shape [B,3,H,W]

                # 1) Clean embeddings (no gradient needed)
                with torch.no_grad():
                    e_clean = model(preprocess_for_facenet(imgs))  # [B, 512]

                # 2) Build bounded noise and resize to image size
                noise = bounded_noise(noise_raw, epsilon)  # [3,250,250]
                noise_resized = F.interpolate(
                    noise.unsqueeze(0), size=imgs.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)  # [3,H,W]
                # Broadcast to batch
                noise_batch = noise_resized.unsqueeze(0).expand(imgs.size(0), -1, -1, -1)

                # 3) Add noise (keep valid image range)
                imgs_noised = torch.clamp(imgs + noise_batch, 0.0, 1.0)

                # 4) Encode noised images (allow grad through the model wrt input)
                e_noised = model(preprocess_for_facenet(imgs_noised))  # [B, 512]

                # 5) Maximize L2 distance -> minimize negative distance
                dist = torch.norm(e_clean - e_noised, dim=1)  # [B]
                mean_dist = dist.mean()
                loss = -mean_dist

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                steps += 1
                running_loss += loss.item()
                running_dist += mean_dist.item()
                avg_loss = running_loss / steps
                avg_dist = running_dist / steps

                pbar.set_postfix({
                    "avg_loss": f"{avg_loss:.4f}",
                    "avg_dist": f"{avg_dist:.4f}",
                    "eps": f"{epsilon:.4f}",
                })

                history_rows.append((epoch, step, float(loss.item()), float(mean_dist.item())))

            # --- Save checkpoint each epoch ---
            # Save BOTH the raw param and a bounded snapshot for convenience
            noise_bounded_snapshot = bounded_noise(noise_raw.detach().clone(), epsilon)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "epsilon": epsilon,
                "noise_raw": noise_raw.detach().cpu(),
                "noise_bounded": noise_bounded_snapshot.cpu(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"[info] Saved checkpoint: {ckpt_path}")

        # --- Finished all epochs: write CSV ---
        csv_path = os.path.join(run_dir, "loss_history.csv")
        _write_history_csv(csv_path, history_rows)
        print(f"[info] Wrote loss history: {csv_path}")

    except KeyboardInterrupt:
        print("\n[info] KeyboardInterrupt received. Saving partial results...")
        # Save current checkpoint
        epoch = len({r[0] for r in history_rows}) + 1  # rough indicator
        noise_bounded_snapshot = bounded_noise(noise_raw.detach().clone(), epsilon)
        ckpt_path = os.path.join(ckpt_dir, f"interrupt_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "epsilon": epsilon,
            "noise_raw": noise_raw.detach().cpu(),
            "noise_bounded": noise_bounded_snapshot.cpu(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }, ckpt_path)
        print(f"[info] Saved interrupt checkpoint: {ckpt_path}")

        # Write CSV so far
        csv_path = os.path.join(run_dir, "loss_history_partial.csv")
        _write_history_csv(csv_path, history_rows)
        print(f"[info] Wrote partial loss history: {csv_path}")


def _write_history_csv(csv_path: str, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "distance"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial noise training against FaceNet embeddings (frozen).")
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Root image folder (people/Name/*.jpg). Used by dataset_simple.build_dataloaders().")
    parser.add_argument("--out-root", type=str, default="runs",
                        help="Root directory under which a new run folder is created.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--epsilon", type=float, default=0.03,
                        help="L∞ bound for noise per pixel (range [0,1]). Common choices: 0.01, 0.03 (~8/255), 0.05.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="Learning rate for the noise parameter (Adam).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"],
                        help="Force device. Defaults to CUDA if available else CPU.")
    # Hugging Face options
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="Hugging Face repo_id for FaceNet weights (optional; script will try known mirrors).")
    parser.add_argument("--hf-filename", type=str, default=None,
                        help="Filename within the HF repo to download (optional).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        root_dir=args.root_dir,
        out_root=args.out_root,
        num_epochs=args.epochs,
        epsilon=args.epsilon,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        repo_id=args.hf_repo_id,
        filename=args.hf_filename,
    )
