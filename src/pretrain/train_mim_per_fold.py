"""
src/pretrain/train_mim_per_fold.py

Per-fold MIM pre-training (leakage-free).

Reviewer #1, Comment 1 fix
--------------------------
The original train_mim.py pre-trains on ALL 42,291 km² tiles, including the
regions reserved for testing in each spatial CV fold.  This script instead
trains a separate encoder for each fold k, excluding fold k's test strip
(plus its 20 km buffer) from the pre-training pool.

The result is 5 encoder checkpoints:
    experiments/pretrain_mim_fold_aware/fold{k}/checkpoint_best.pt

The downstream fine-tuning script (train_finetune.py) must load the
matching fold-k encoder, not a single shared one.

Usage
-----
Single fold (recommended — run one per GPU in parallel):
    CUDA_VISIBLE_DEVICES=0 python src/pretrain/train_mim_per_fold.py --fold 0
    CUDA_VISIBLE_DEVICES=1 python src/pretrain/train_mim_per_fold.py --fold 1
    ...

All folds sequentially on one GPU:
    CUDA_VISIBLE_DEVICES=3 python src/pretrain/train_mim_per_fold.py --all_folds

Resume a fold:
    CUDA_VISIBLE_DEVICES=0 python src/pretrain/train_mim_per_fold.py --fold 0 --resume

Quick sanity check (5 epochs, no wandb):
    CUDA_VISIBLE_DEVICES=3 python src/pretrain/train_mim_per_fold.py --fold 0 \
        --epochs 5 --no_wandb
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pretrain.dataset_fold_aware import build_dataloaders_for_fold
from src.pretrain.model import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Data
    "tif_path"          : "data/processed/stacked/multiband_input.tif",
    "splits_dir"        : "data/splits",
    "patch_size_px"     : 256,
    "stride_px"         : 128,
    "in_channels"       : 15,
    "num_workers"       : 4,
    "noise_sigma"       : 0.01,

    # Model (ViT-Base, identical to original)
    "img_size"          : 256,
    "model_patch_size"  : 16,
    "encoder_embed_dim" : 768,
    "encoder_depth"     : 12,
    "encoder_num_heads" : 12,
    "decoder_embed_dim" : 512,
    "decoder_depth"     : 4,
    "decoder_num_heads" : 16,
    "mask_ratio"        : 0.75,

    # Training  (same hypers as original to keep comparison fair)
    "epochs"            : 300,
    "batch_size"        : 16,      # single GPU — reduced from 64
    "base_lr"           : 1.5e-4,
    "min_lr"            : 1e-6,
    "weight_decay"      : 0.05,
    "beta1"             : 0.9,
    "beta2"             : 0.95,
    "warmup_epochs"     : 40,
    "grad_clip"         : 1.0,

    # Logging / checkpointing
    "checkpoint_root"   : "experiments/pretrain_mim_fold_aware",
    "save_every"        : 10,
    "log_every"         : 50,
    "use_wandb"         : False,   # set True if wandb available
    "wandb_project"     : "mineral_mapping_pretrain",
}

N_FOLDS = 5


# ── LR schedule ──────────────────────────────────────────────────────────────
def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        t  = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ── Training / validation loops ───────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            loss, _, _ = model(batch)
        scaler.scale(loss).backward()
        if config["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if i % config["log_every"] == 0:
            print(f"  Epoch {epoch:03d} [{i:4d}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")
    return {"train_loss": total_loss / max(1, len(loader)),
            "epoch_time": time.time() - t0}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            loss, _, _ = model(batch)
        total_loss += loss.item()
    return {"val_loss": total_loss / max(1, len(loader))}


def save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"]


# ── Per-fold training ─────────────────────────────────────────────────────────
def train_fold(fold_idx, config, resume=False):
    print(f"\n{'='*60}")
    print(f"  MIM PRE-TRAINING  —  FOLD {fold_idx}  (leakage-free)")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name()}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    exclude_mask = Path(config["splits_dir"]) / f"fold{fold_idx}_test_mask.tif"
    print(f"\nExclusion mask: {exclude_mask}")

    train_loader, val_loader, n_channels = build_dataloaders_for_fold(
        tif_path          = config["tif_path"],
        exclude_mask_path = str(exclude_mask),
        fold_idx          = fold_idx,
        patch_size        = config["patch_size_px"],
        stride            = config["stride_px"],
        batch_size        = config["batch_size"],
        num_workers       = config["num_workers"],
        noise_sigma       = config["noise_sigma"],
    )

    # Linear LR scaling
    effective_batch = config["batch_size"]
    scaled_lr       = config["base_lr"] * effective_batch / 256
    print(f"Effective batch: {effective_batch}   Scaled LR: {scaled_lr:.2e}")
    print(f"Train batches  : {len(train_loader)}   Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model({
        **config,
        "in_channels" : n_channels,
        "patch_size"  : config["model_patch_size"],
    }).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_p/1e6:.1f}M")

    # ── Optimizer (weight decay on matrices only) ──────────────────────
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr    = scaled_lr,
        betas = (config["beta1"], config["beta2"]),
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Resume ────────────────────────────────────────────────────────────
    ckpt_dir    = Path(config["checkpoint_root"]) / f"fold{fold_idx}"
    start_epoch = 1
    best_val    = float("inf")

    if resume:
        latest = ckpt_dir / "checkpoint_latest.pt"
        if latest.exists():
            start_epoch = load_checkpoint(latest, model, optimizer, scaler) + 1
        else:
            print("  No checkpoint found — starting fresh")

    # ── Wandb ─────────────────────────────────────────────────────────────
    use_wandb = config["use_wandb"] and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project = config["wandb_project"],
            name    = f"mim_fold{fold_idx}_leakage_free",
            config  = {**config, "fold": fold_idx},
            resume  = "allow",
        )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({**config, "fold": fold_idx}, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'LR':>10}  {'Time':>8}")
    print("-" * 56)

    for epoch in range(start_epoch, config["epochs"] + 1):
        lr = cosine_lr(optimizer, epoch, config["warmup_epochs"],
                       config["epochs"], scaled_lr, config["min_lr"])

        tm = train_one_epoch(model, train_loader, optimizer,
                             scaler, device, epoch, config)
        vm = validate(model, val_loader, device)

        print(f"{epoch:6d}  {tm['train_loss']:12.4f}  {vm['val_loss']:10.4f}  "
              f"{lr:10.2e}  {tm['epoch_time']:8.1f}s")

        if use_wandb:
            wandb.log({"epoch": epoch, "fold": fold_idx,
                       "train_loss": tm["train_loss"],
                       "val_loss":   vm["val_loss"], "lr": lr})

        state = {
            "epoch"      : epoch,
            "fold"       : fold_idx,
            "model"      : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "scaler"     : scaler.state_dict(),
            "config"     : config,
            "train_loss" : tm["train_loss"],
            "val_loss"   : vm["val_loss"],
        }

        save_checkpoint(state, ckpt_dir / "checkpoint_latest.pt")

        if epoch % config["save_every"] == 0:
            save_checkpoint(state, ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt")

        if vm["val_loss"] < best_val:
            best_val = vm["val_loss"]
            save_checkpoint(state, ckpt_dir / "checkpoint_best.pt")
            print(f"  ★ New best val loss: {best_val:.4f}")

    print(f"\nFold {fold_idx} pre-training complete.")
    print(f"Best val loss : {best_val:.4f}")
    print(f"Best ckpt     : {ckpt_dir / 'checkpoint_best.pt'}")

    if use_wandb:
        wandb.finish()

    return best_val


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-fold leakage-free MIM pre-training"
    )
    parser.add_argument("--fold",       type=int,   default=None,
                        help="Single fold to train (0–4)")
    parser.add_argument("--all_folds",  action="store_true",
                        help="Train all 5 folds sequentially")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    if args.fold is None and not args.all_folds:
        parser.error("Specify --fold 0..4  or  --all_folds")

    config = DEFAULT_CONFIG.copy()
    if args.epochs:     config["epochs"]      = args.epochs
    if args.batch_size: config["batch_size"]  = args.batch_size
    if args.lr:         config["base_lr"]     = args.lr
    if args.workers:    config["num_workers"] = args.workers
    if args.no_wandb:   config["use_wandb"]   = False

    folds = list(range(N_FOLDS)) if args.all_folds else [args.fold]

    summary = {}
    for k in folds:
        summary[k] = train_fold(k, config, resume=args.resume)

    print(f"\n{'='*40}")
    print("  Summary")
    print(f"{'='*40}")
    for k, v in summary.items():
        print(f"  Fold {k}: best val loss = {v:.4f}")
    print(f"\nCheckpoints in: {config['checkpoint_root']}/")