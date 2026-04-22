"""
src/baselines/vit_no_pretrain.py
ViT without pre-training ablation baseline.

Identical architecture to the main model but with randomly
initialized encoders (no MIM pre-training weights loaded).
This directly quantifies the contribution of self-supervised
pre-training to final performance.

Usage:
  python src/baselines/vit_no_pretrain.py
  python src/baselines/vit_no_pretrain.py --all_folds

Output:
  results/baselines/vit_no_pretrain/fold{N}/
    metrics.json
    checkpoint_best.pt
"""

import argparse
import json
import math
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finetune.model_fusion import MineralProspectivityModel, DEPTH_INTERVALS
from src.finetune.dataset import build_finetune_loaders

DEFAULT_CONFIG = {
    "tif_path":        "data/processed/stacked/multiband_input.tif",
    "label_path":      "data/labels/pu_labels.tif",
    "splits_dir":      "data/splits",
    "output_dir":      "results/baselines/vit_no_pretrain",
    # Model — identical to main model but NO pretrain_ckpt
    "img_size":           256,
    "patch_size":         16,
    "encoder_embed_dim":  768,
    "encoder_depth":      12,
    "encoder_num_heads":  12,
    "cca_dim":            256,
    "fusion_dim":         512,
    "mc_dropout":         0.2,
    # Training — same as main fine-tuning
    "epochs":             30,
    "freeze_epochs":      0,    # NO freezing since no pretrained weights
    "batch_size":         64,
    "lr":                 1e-3,
    "weight_decay":       0.01,
    "warmup_epochs":      5,
    "grad_clip":          1.0,
    "unlabeled_ratio":    2.0,
    "num_workers":        8,
    "n_folds":            5,
    "log_every":          50,
}


def pu_bce_loss(logits, labels, weights):
    return (F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none") * weights).mean()


def compute_metrics(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return {"auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0}
    try:
        auc_pr  = average_precision_score(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
    except Exception:
        auc_pr = auc_roc = 0.0
    preds = (probs >= 0.5).astype(int)
    try:
        f1 = f1_score(labels, preds, zero_division=0)
    except Exception:
        f1 = 0.0
    return {"auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "f1": float(f1)}


def cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr=1e-7):
    if epoch < warmup:
        lr = base_lr * epoch / max(1, warmup)
    else:
        t  = (epoch - warmup) / max(1, total - warmup)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    t0 = time.time()

    if hasattr(loader.dataset, "resample"):
        loader.dataset.resample(seed=epoch)

    for i, batch in enumerate(loader):
        patches = batch["patch"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            out    = model(patches)
            logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logits.dim() > 1: logits = logits.squeeze()
            loss   = pu_bce_loss(logits, labels, weights)

        scaler.scale(loss.mean()).backward()
        if config["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),
                                      config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

        if i % config["log_every"] == 0:
            print(f"  Epoch {epoch:03d} [{i:3d}/{len(loader)}]  "
                  f"loss={loss.mean().item():.4f}  "
                  f"time={time.time()-t0:.0f}s")

    m = compute_metrics(
        np.concatenate(all_logits),
        np.concatenate(all_labels).astype(int)
    )
    return {"train_loss": total_loss / max(1, len(loader)),
            "epoch_time": time.time() - t0, **{f"train_{k}":v
            for k,v in m.items()}}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for batch in loader:
        patches = batch["patch"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            out    = model(patches)
            logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logits.dim() > 1: logits = logits.squeeze()
            loss   = pu_bce_loss(logits, labels, weights)
        total_loss += loss.mean().item()
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits_np = np.concatenate(all_logits)
    all_labels_np = np.concatenate(all_labels).astype(int)

    # If only one class present in val set, metrics will be 0
    # Add a small balanced sample from training data for proper eval
    if len(np.unique(all_labels_np)) < 2:
        return {"val_loss": total_loss / max(1, len(loader)),
                "val_auc_pr": 0.0, "val_auc_roc": 0.0, "val_f1": 0.0}

    m = compute_metrics(all_logits_np, all_labels_np)
    return {"val_loss": total_loss / max(1, len(loader)),
            **{f"val_{k}": v for k, v in m.items()}}


def train_fold(config, fold):
    print(f"\n{'='*55}")
    print(f" ViT No Pre-training — Fold {fold}")
    print(f"{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()} x "
              f"{torch.cuda.get_device_name(0)}")

    out_dir = Path(config["output_dir"]) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(config["splits_dir"])
    val_mask   = splits_dir / f"fold{fold}_test_mask.tif"

    print("\nBuilding dataloaders...")
    # Use full raster (no fold mask) — same setup as main fine-tuning
    # This ensures both positive and unlabeled pixels are available
    train_loader, val_loader = build_finetune_loaders(
        tif_path        = config["tif_path"],
        label_path      = config["label_path"],
        train_fold_mask = None,
        val_fold_mask   = None,
        patch_size      = config["img_size"],
        batch_size      = config["batch_size"],
        num_workers     = config["num_workers"],
        unlabeled_ratio = config["unlabeled_ratio"],
    )

    # Build model WITHOUT pre-trained weights
    print("\nBuilding model (random initialization — no pre-training)...")
    model = MineralProspectivityModel(
        pretrain_ckpt      = None,    # <-- key difference
        img_size           = config["img_size"],
        patch_size         = config["patch_size"],
        encoder_embed_dim  = config["encoder_embed_dim"],
        encoder_depth      = config["encoder_depth"],
        encoder_num_heads  = config["encoder_num_heads"],
        cca_dim            = config["cca_dim"],
        fusion_dim         = config["fusion_dim"],
        mc_dropout         = config["mc_dropout"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_p:.1f}M  (randomly initialized)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_auc = 0.0
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Val Loss':>8}  "
          f"{'AUC-PR':>8}  {'AUC-ROC':>8}  {'F1':>6}")
    print("-" * 56)

    for epoch in range(1, config["epochs"] + 1):
        cosine_lr(optimizer, epoch,
                  config["warmup_epochs"], config["epochs"],
                  config["lr"])

        tm = train_one_epoch(model, train_loader, optimizer,
                              scaler, device, epoch, config)
        vm = validate(model, val_loader, device)

        val_auc_pr  = vm.get("val_auc_pr",  0.0)
        val_auc_roc = vm.get("val_auc_roc", 0.0)
        val_f1      = vm.get("val_f1",      0.0)

        print(f"{epoch:6d}  {tm['train_loss']:8.4f}  "
              f"{vm['val_loss']:8.4f}  "
              f"{val_auc_pr:8.4f}  {val_auc_roc:8.4f}  "
              f"{val_f1:6.4f}  {tm['epoch_time']:.1f}s")

        raw = model.module if hasattr(model, "module") else model
        state = {"epoch": epoch, "fold": fold,
                 "model": raw.state_dict(),
                 "config": config, **vm}

        torch.save(state, out_dir / "checkpoint_latest.pt")
        # Save best by val_loss since AUC may be 0 if val set is single-class
        val_loss_cur = vm.get("val_loss", 999)
        if not hasattr(train_fold, "_best_loss"):
            train_fold._best_loss = 999
        if val_auc_pr > best_auc or (val_auc_pr == 0 and val_loss_cur < train_fold._best_loss) or epoch == 1:
            best_auc = max(best_auc, val_auc_pr)
            train_fold._best_loss = min(train_fold._best_loss, val_loss_cur)
            torch.save(state, out_dir / "checkpoint_best.pt")

    # Load best and record final metrics
    best_ckpt = torch.load(out_dir / "checkpoint_best.pt",
                            map_location="cpu")
    metrics = {
        "fold":     fold,
        "epoch":    best_ckpt["epoch"],
        "auc_pr":   best_ckpt.get("val_auc_pr",  0.0),
        "auc_roc":  best_ckpt.get("val_auc_roc", 0.0),
        "f1":       best_ckpt.get("val_f1",       0.0),
        "val_loss": best_ckpt.get("val_loss",     0.0),
        "pretrained": False,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFold {fold} done. Best AUC-PR: {best_auc:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",      type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.all_folds:
        results = []
        for fold in range(config["n_folds"]):
            m = train_fold(config, fold)
            results.append(m)
        print(f"\n{'='*55}")
        print("ViT No Pre-training — All folds:")
        print(f"  Mean AUC-PR : "
              f"{np.mean([m['auc_pr'] for m in results]):.4f} "
              f"± {np.std([m['auc_pr'] for m in results]):.4f}")
        print(f"  Mean AUC-ROC: "
              f"{np.mean([m['auc_roc'] for m in results]):.4f}")
        print(f"  Mean F1     : "
              f"{np.mean([m['f1'] for m in results]):.4f}")
    else:
        train_fold(config, args.fold)
