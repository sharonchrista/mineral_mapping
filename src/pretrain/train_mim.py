"""
src/pretrain/train_mim.py
MIM pre-training loop. Proposal specs section 4.2.2:
  Epochs   : 300
  LR       : 1.5e-4 cosine + warmup
  Batch    : 256 effective (64 per GPU x 4 GPUs)
  Optimizer: AdamW beta1=0.9 beta2=0.95 wd=0.05
  Masking  : 75%
  Loss     : MSE on masked patches only

Usage:
  python src/pretrain/train_mim.py
  python src/pretrain/train_mim.py --epochs 300 --resume
  python src/pretrain/train_mim.py --no_wandb
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
from src.pretrain.dataset import build_dataloaders
from src.pretrain.model import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


DEFAULT_CONFIG = {
    "tif_path":           "data/processed/stacked/multiband_input.tif",
    "patch_size_px":      256,
    "stride_px":          128,
    "in_channels":        15,
    "num_workers":        8,
    "img_size":           256,
    "model_patch_size":   16,
    "encoder_embed_dim":  768,
    "encoder_depth":      12,
    "encoder_num_heads":  12,
    "decoder_embed_dim":  512,
    "decoder_depth":      4,
    "decoder_num_heads":  16,
    "mask_ratio":         0.75,
    "epochs":             300,
    "batch_size":         64,
    "base_lr":            1.5e-4,
    "min_lr":             1e-6,
    "weight_decay":       0.05,
    "beta1":              0.9,
    "beta2":              0.95,
    "warmup_epochs":      40,
    "grad_clip":          1.0,
    "noise_sigma":        0.01,
    "checkpoint_dir":     "experiments/pretrain_mim",
    "save_every":         10,
    "log_every":          50,
    "use_wandb":          True,
    "wandb_project":      "mineral_mapping_pretrain",
    "wandb_run_name":     "mim_vit_base_300ep",
}


def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            loss, _, _ = model(batch)
        scaler.scale(loss.mean()).backward()
        if config["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.mean().item()
        if i % config["log_every"] == 0:
            print(f"  Epoch {epoch:03d} [{i:4d}/{len(loader)}]  "
                  f"loss={loss.mean().item():.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"time={time.time()-t0:.0f}s")
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
        total_loss += loss.mean().item()
    return {"val_loss": total_loss / max(1, len(loader))}


def save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from: {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def train(config, resume=False):
    # Device
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Training on {n_gpus} GPU(s):")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, training on CPU")

    # Dataloaders
    print("\nBuilding dataloaders...")
    train_loader, val_loader, n_channels = build_dataloaders(
        tif_path=config["tif_path"], patch_size=config["patch_size_px"],
        stride=config["stride_px"], batch_size=config["batch_size"],
        num_workers=config["num_workers"], noise_sigma=config["noise_sigma"],
    )
    effective_batch = config["batch_size"] * max(1, torch.cuda.device_count())
    scaled_lr = config["base_lr"] * effective_batch / 256
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"  Effective batch: {effective_batch}  Scaled LR: {scaled_lr:.2e}")

    # Model
    print("\nBuilding model...")
    model = build_model({**config, "in_channels": n_channels,
                         "patch_size": config["model_patch_size"]}).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p/1e6:.1f}M")

    # Optimizer with weight decay only on weight matrices
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=scaled_lr, betas=(config["beta1"], config["beta2"]),
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    # Resume
    ckpt_dir = Path(config["checkpoint_dir"])
    start_epoch = 1
    best_val_loss = float("inf")
    if resume:
        latest = ckpt_dir / "checkpoint_latest.pt"
        if latest.exists():
            start_epoch = load_checkpoint(latest, model, optimizer, scaler) + 1
        else:
            print("  No checkpoint found, starting fresh")

    # Wandb
    use_wandb = config["use_wandb"] and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=config["wandb_project"], name=config["wandb_run_name"],
                   config=config, resume="allow")
    elif config["use_wandb"] and not WANDB_AVAILABLE:
        print("WARNING: wandb not installed, logging to console only")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nMIM pre-training: {config['epochs']} epochs\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}  {'Time':>8}")
    print("-" * 58)

    for epoch in range(start_epoch, config["epochs"] + 1):
        lr = cosine_lr(optimizer, epoch, config["warmup_epochs"],
                       config["epochs"], scaled_lr, config["min_lr"])
        tm = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        vm = validate(model, val_loader, device)

        print(f"{epoch:6d}  {tm['train_loss']:12.4f}  {vm['val_loss']:10.4f}  "
              f"{lr:10.2e}  {tm['epoch_time']:8.1f}s")

        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": tm["train_loss"],
                       "val_loss": vm["val_loss"], "lr": lr})

        raw_model = model.module if hasattr(model, "module") else model
        state = {"epoch": epoch, "model": raw_model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "config": config, "train_loss": tm["train_loss"],
                 "val_loss": vm["val_loss"]}

        save_checkpoint(state, ckpt_dir / "checkpoint_latest.pt")
        if epoch % config["save_every"] == 0:
            save_checkpoint(state, ckpt_dir / f"checkpoint_epoch{epoch:04d}.pt")
        if vm["val_loss"] < best_val_loss:
            best_val_loss = vm["val_loss"]
            save_checkpoint(state, ckpt_dir / "checkpoint_best.pt")
            print(f"  New best val loss: {best_val_loss:.4f}")

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'checkpoint_best.pt'}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--workers",    type=int,   default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.epochs:     config["epochs"]      = args.epochs
    if args.batch_size: config["batch_size"]  = args.batch_size
    if args.lr:         config["base_lr"]     = args.lr
    if args.mask_ratio: config["mask_ratio"]  = args.mask_ratio
    if args.workers:    config["num_workers"] = args.workers
    if args.no_wandb:   config["use_wandb"]   = False

    train(config, resume=args.resume)
