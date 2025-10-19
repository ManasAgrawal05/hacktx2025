#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_facenet_noise.py

Adversarial noise optimization for FaceNet embeddings, using a frozen model.

Features:
- Linear warmup + cosine annealing LR schedule
- Early stopping based on validation distance (patience-based)
- Saves per-epoch checkpoints AND best.pt when validation improves
"""

import os
import csv
import json
import math
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset_simple import build_dataloaders
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def try_load_state_dict(model: nn.Module, ckpt_path: str) -> bool:
    try:
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
    except Exception as e:
        print(f"[warn] Failed to load state_dict from '{ckpt_path}': {e}")
        return False


def load_facenet_from_hf(device, repo_id=None, filename=None, fallback_to_facenet_pytorch=True):
    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    tried_any = False
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
            tried_any = True
            if try_load_state_dict(model, ckpt_path):
                print(f"[info] Loaded FaceNet weights from HF: {rid}/{fname}")
                return model
        except Exception as e:
            print(f"[warn] Could not fetch {rid}/{fname}: {e}")

    if fallback_to_facenet_pytorch:
        print("[info] Falling back to facenet-pytorch pretrained='vggface2'.")
        model_fallback = InceptionResnetV1(pretrained="vggface2", classify=False)
        model_fallback.eval().to(device)
        for p in model_fallback.parameters():
            p.requires_grad = False
        return model_fallback

    if not tried_any:
        raise RuntimeError("No HF candidates were attempted.")
    raise RuntimeError("Failed to load FaceNet weights from the Hugging Face Hub.")


def preprocess_for_facenet(x):
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def bounded_noise(noise_raw, epsilon):
    return epsilon * torch.tanh(noise_raw)


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train(
    root_dir,
    out_root,
    num_epochs,
    epsilon,
    lr,
    seed,
    device,
    repo_id=None,
    filename=None,
    warmup_epochs=10,
    patience=5,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] Using device: {device}")

    train_loader, val_loader = build_dataloaders(root_dir, batch_size=256)
    model = load_facenet_from_hf(device=device, repo_id=repo_id, filename=filename)

    noise_raw = nn.Parameter(torch.zeros(3, 250, 250))
    nn.init.normal_(noise_raw, mean=0.0, std=1e-3)
    noise_raw = noise_raw.to(device)

    optimizer = torch.optim.Adam([noise_raw], lr=lr)

    # LR Scheduler: Warmup + Cosine
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output dirs
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
        "warmup_epochs": warmup_epochs,
        "patience": patience,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    history_rows = []
    global_step = 0
    best_val = -float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    try:
        for epoch in range(1, num_epochs + 1):
            model.eval()
            running_loss = 0.0
            running_dist = 0.0
            steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=120)
            for step, (imgs, _) in enumerate(pbar, start=1):
                imgs = imgs.to(device).float()
                with torch.no_grad():
                    e_clean = model(preprocess_for_facenet(imgs))

                noise = bounded_noise(noise_raw, epsilon)
                noise_resized = F.interpolate(
                    noise.unsqueeze(0), size=imgs.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)
                noise_batch = noise_resized.unsqueeze(0).expand(imgs.size(0), -1, -1, -1)
                imgs_noised = torch.clamp(imgs + noise_batch, 0.0, 1.0)

                e_noised = model(preprocess_for_facenet(imgs_noised))
                dist = torch.norm(e_clean - e_noised, dim=1)
                mean_dist = dist.mean()
                loss = -mean_dist

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1
                steps += 1
                running_loss += loss.item()
                running_dist += mean_dist.item()
                avg_loss = running_loss / steps
                avg_dist = running_dist / steps
                current_lr = scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    "avg_loss": f"{avg_loss:.4f}",
                    "avg_dist": f"{avg_dist:.4f}",
                    "lr": f"{current_lr:.5f}",
                    "eps": f"{epsilon:.4f}",
                })

                history_rows.append((epoch, step, float(loss.item()), float(mean_dist.item())))

            # Validation
            val_dist_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for imgs, _ in tqdm(val_loader, desc="Validating", ncols=120, leave=False):
                    imgs = imgs.to(device).float()
                    e_clean = model(preprocess_for_facenet(imgs))
                    noise = bounded_noise(noise_raw, epsilon)
                    noise_resized = F.interpolate(
                        noise.unsqueeze(0), size=imgs.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(0)
                    noise_batch = noise_resized.unsqueeze(0).expand(imgs.size(0), -1, -1, -1)
                    imgs_noised = torch.clamp(imgs + noise_batch, 0.0, 1.0)
                    e_noised = model(preprocess_for_facenet(imgs_noised))
                    dist = torch.norm(e_clean - e_noised, dim=1)
                    val_dist_sum += dist.mean().item()
                    val_steps += 1

            val_avg_dist = val_dist_sum / val_steps
            print(f"[val] Epoch {epoch}: mean distance = {val_avg_dist:.4f}")
            history_rows.append((epoch, 0, float('nan'), float(val_avg_dist)))

            # Save checkpoint for this epoch
            noise_bounded_snapshot = bounded_noise(noise_raw.detach().clone(), epsilon)
            epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "epsilon": epsilon,
                "noise_raw": noise_raw.detach().cpu(),
                "noise_bounded": noise_bounded_snapshot.cpu(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }, epoch_ckpt_path)
            print(f"[info] Saved checkpoint: {epoch_ckpt_path}")

            # Check for best validation
            if val_avg_dist > best_val:
                best_val = val_avg_dist
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "epsilon": epsilon,
                    "noise_raw": noise_raw.detach().cpu(),
                    "noise_bounded": noise_bounded_snapshot.cpu(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "val_avg_dist": best_val,
                }, best_ckpt_path)
                print(
                    f"[best] New best validation distance: {best_val:.4f} — saved to {best_ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"[info] No improvement for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    print(
                        f"[early stop] Validation distance has not improved for {patience} epochs.")
                    break

        # Final save
        final_ckpt = os.path.join(ckpt_dir, "early_stop_final.pt")
        torch.save({
            "epoch": epoch,
            "epsilon": epsilon,
            "noise_raw": noise_raw.detach().cpu(),
            "noise_bounded": bounded_noise(noise_raw.detach().clone(), epsilon).cpu(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }, final_ckpt)
        print(f"[info] Saved final checkpoint: {final_ckpt}")

        csv_path = os.path.join(run_dir, "loss_history.csv")
        _write_history_csv(csv_path, history_rows)
        print(f"[info] Wrote loss history: {csv_path}")

    except KeyboardInterrupt:
        print("\n[info] KeyboardInterrupt received. Saving partial results...")
        epoch = len({r[0] for r in history_rows}) + 1
        ckpt_path = os.path.join(ckpt_dir, f"interrupt_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "epsilon": epsilon,
            "noise_raw": noise_raw.detach().cpu(),
            "noise_bounded": bounded_noise(noise_raw.detach().clone(), epsilon).cpu(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }, ckpt_path)
        print(f"[info] Saved interrupt checkpoint: {ckpt_path}")
        csv_path = os.path.join(run_dir, "loss_history_partial.csv")
        _write_history_csv(csv_path, history_rows)
        print(f"[info] Wrote partial loss history: {csv_path}")


def _write_history_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "distance"])
        writer.writerows(rows)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial noise training against FaceNet embeddings (frozen).")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-filename", type=str, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs before cosine annealing.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement).")
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
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
    )
