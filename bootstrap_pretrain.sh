#!/bin/bash
# =============================================================================
# bootstrap_pretrain.sh
# Creates all MIM pre-training scripts in the correct folder structure.
# Run from: ~/sharon/mineral_mapping/
# Usage:    bash bootstrap_pretrain.sh
# =============================================================================

set -e
BASE=~/sharon/mineral_mapping

echo "Creating folder structure..."
mkdir -p $BASE/src/pretrain
mkdir -p $BASE/experiments/pretrain_mim
touch $BASE/src/pretrain/__init__.py

# =============================================================================
# dataset.py
# =============================================================================
cat > $BASE/src/pretrain/dataset.py << 'PYEOF'
"""
src/pretrain/dataset.py
Tiles the 15-channel multiband GeoTIFF into 256x256 patches for MIM pre-training.
"""

import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class GeoTileDataset(Dataset):
    """
    Tiles multiband_input.tif into spatial patches for MIM pre-training.

    Args:
        tif_path   : path to multiband_input.tif
        patch_size : tile size in pixels (default 256)
        stride     : step between tiles (default = patch_size)
        augment    : random rotation, flip, gaussian noise
        min_valid  : minimum fraction of non-zero pixels to keep a tile
        noise_sigma: std of gaussian noise (proposal 4.2.2: 0.01)
    """

    def __init__(self, tif_path, patch_size=256, stride=None,
                 augment=True, min_valid=0.5, noise_sigma=0.01):
        self.tif_path    = Path(tif_path)
        self.patch_size  = patch_size
        self.stride      = stride if stride is not None else patch_size
        self.augment     = augment
        self.min_valid   = min_valid
        self.noise_sigma = noise_sigma

        with rasterio.open(self.tif_path) as src:
            self.data = src.read().astype("float32")   # (C, H, W)
            self.n_channels = src.count
            self.H = src.height
            self.W = src.width

        self.data = np.where(self.data == -9999.0, 0.0, self.data)
        self.tiles = self._build_tile_index()
        print(f"GeoTileDataset: {len(self.tiles)} tiles | "
              f"patch={patch_size}px | stride={self.stride}px | "
              f"channels={self.n_channels}")

    def _build_tile_index(self):
        tiles = []
        P, S = self.patch_size, self.stride
        for r in range(0, self.H - P + 1, S):
            for c in range(0, self.W - P + 1, S):
                patch = self.data[:, r:r+P, c:c+P]
                if (patch.sum(axis=0) != 0).mean() >= self.min_valid:
                    tiles.append((r, c))
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        r, c = self.tiles[idx]
        P = self.patch_size
        patch = torch.from_numpy(self.data[:, r:r+P, c:c+P].copy())
        if self.augment:
            patch = self._augment(patch)
        return patch

    def _augment(self, patch):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            patch = TF.rotate(patch, angle)
        if random.random() > 0.5:
            patch = TF.hflip(patch)
        if random.random() > 0.5:
            patch = TF.vflip(patch)
        if self.noise_sigma > 0:
            patch = torch.clamp(patch + torch.randn_like(patch) * self.noise_sigma, 0.0, 1.0)
        return patch


def build_dataloaders(tif_path, patch_size=256, stride=128, batch_size=32,
                      num_workers=8, val_fraction=0.1, noise_sigma=0.01, seed=42):
    from torch.utils.data import DataLoader, random_split

    full_ds = GeoTileDataset(tif_path=tif_path, patch_size=patch_size,
                              stride=stride, augment=True, noise_sigma=noise_sigma)
    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    val_ds.dataset.augment = False

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, full_ds.n_channels


if __name__ == "__main__":
    tif = Path("data/processed/stacked/multiband_input.tif")
    train_loader, val_loader, n_ch = build_dataloaders(
        tif, patch_size=256, stride=256, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}  dtype: {batch.dtype}  "
          f"min: {batch.min():.3f}  max: {batch.max():.3f}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print("Dataset OK")
PYEOF

echo "  OK dataset.py"

# =============================================================================
# model.py
# =============================================================================
cat > $BASE/src/pretrain/model.py << 'PYEOF'
"""
src/pretrain/model.py
Masked Image Modeling (MIM) architecture.
  Encoder : ViT-Base (12 layers, 768 dim, 12 heads)
  Decoder : lightweight (4 layers, 512 dim, 16 heads)
  Input   : 15-channel patches (256x256 px)
  Patches : 16x16 px tokens
  Masking : 75% random
  Loss    : MSE on masked patches only
"""

import torch
import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    assert embed_dim % 4 == 0
    half  = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

    h_ids = torch.arange(grid_h, dtype=torch.float32)
    w_ids = torch.arange(grid_w, dtype=torch.float32)

    h_sin = torch.sin(h_ids[:, None] * omega[None, :])
    h_cos = torch.cos(h_ids[:, None] * omega[None, :])
    w_sin = torch.sin(w_ids[:, None] * omega[None, :])
    w_cos = torch.cos(w_ids[:, None] * omega[None, :])

    h_emb = torch.cat([h_sin, h_cos], dim=-1)[:, None, :].expand(grid_h, grid_w, half)
    w_emb = torch.cat([w_sin, w_cos], dim=-1)[None, :, :].expand(grid_h, grid_w, half)

    return torch.cat([h_emb, w_emb], dim=-1).reshape(grid_h * grid_w, embed_dim)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=15, embed_dim=768):
        super().__init__()
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj        = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=15,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        n_keep  = int(N * (1 - mask_ratio))
        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)
        ids_keep     = ids_shuffle[:, :n_keep]
        x_visible    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
        mask         = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x) + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore


class MIMDecoder(nn.Module):
    def __init__(self, num_patches, encoder_embed_dim=768, decoder_embed_dim=512,
                 decoder_depth=4, decoder_num_heads=16, patch_size=16,
                 in_channels=15, mlp_ratio=4.0, grid_size=16):
        super().__init__()
        patch_dim   = patch_size * patch_size * in_channels
        self.embed  = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        pos = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latent, ids_restore):
        x = self.embed(latent)[:, 1:, :]         # remove CLS
        B, N_vis, D = x.shape
        n_masked = ids_restore.shape[1] - N_vis
        mask_tokens = self.mask_token.expand(B, n_masked, D)
        x_full = torch.gather(
            torch.cat([x, mask_tokens], dim=1), 1,
            ids_restore.unsqueeze(-1).expand(-1,-1,D)
        )
        x_full = x_full + self.pos_embed
        for block in self.blocks:
            x_full = block(x_full)
        return self.pred(self.norm(x_full))


class MaskedImageModel(nn.Module):
    """
    Full MIM model: ViT-Base encoder + lightweight decoder.
    Proposal section 4.2.1.
    """

    def __init__(self, img_size=256, patch_size=16, in_channels=15,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mask_ratio=0.75, mlp_ratio=4.0):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.grid_size   = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.encoder = ViTEncoder(img_size, patch_size, in_channels,
                                   encoder_embed_dim, encoder_depth,
                                   encoder_num_heads, mlp_ratio)
        self.decoder = MIMDecoder(self.num_patches, encoder_embed_dim,
                                   decoder_embed_dim, decoder_depth,
                                   decoder_num_heads, patch_size, in_channels,
                                   mlp_ratio, self.grid_size)

    def patchify(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        G = H // P
        return x.reshape(B, C, G, P, G, P).permute(0,2,4,3,5,1).reshape(B, G*G, P*P*C)

    def compute_loss(self, pred, target, mask):
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    def forward(self, x):
        latent, mask, ids_restore = self.encoder(x, self.mask_ratio)
        pred   = self.decoder(latent, ids_restore)
        target = self.patchify(x)
        loss   = self.compute_loss(pred, target, mask)
        return loss, pred, mask

    def encode(self, x, mask_ratio=0.0):
        latent, _, _ = self.encoder(x, mask_ratio=mask_ratio)
        return latent


def build_model(config):
    return MaskedImageModel(
        img_size           = config.get("img_size", 256),
        patch_size         = config.get("patch_size", 16),
        in_channels        = config.get("in_channels", 15),
        encoder_embed_dim  = config.get("encoder_embed_dim", 768),
        encoder_depth      = config.get("encoder_depth", 12),
        encoder_num_heads  = config.get("encoder_num_heads", 12),
        decoder_embed_dim  = config.get("decoder_embed_dim", 512),
        decoder_depth      = config.get("decoder_depth", 4),
        decoder_num_heads  = config.get("decoder_num_heads", 16),
        mask_ratio         = config.get("mask_ratio", 0.75),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = MaskedImageModel().to(device)
    total  = sum(p.numel() for p in model.parameters())
    enc    = sum(p.numel() for p in model.encoder.parameters())
    dec    = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total params  : {total/1e6:.1f}M")
    print(f"Encoder params: {enc/1e6:.1f}M")
    print(f"Decoder params: {dec/1e6:.1f}M")
    x = torch.randn(2, 15, 256, 256).to(device)
    loss, pred, mask = model(x)
    print(f"Loss: {loss.item():.4f}  Pred: {pred.shape}  Mask: {mask.shape}")
    print("Model OK")
PYEOF

echo "  OK model.py"

# =============================================================================
# train_mim.py
# =============================================================================
cat > $BASE/src/pretrain/train_mim.py << 'PYEOF'
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
        total_loss += loss.item()
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
PYEOF

echo "  OK train_mim.py"

echo ""
echo "========================================================"
echo " All pre-training scripts created:"
echo "   src/pretrain/dataset.py"
echo "   src/pretrain/model.py"
echo "   src/pretrain/train_mim.py"
echo ""
echo " Quick sanity checks:"
echo "   python src/pretrain/dataset.py"
echo "   python src/pretrain/model.py"
echo ""
echo " Start pre-training (no wandb):"
echo "   python src/pretrain/train_mim.py --no_wandb"
echo ""
echo " Start pre-training (with wandb):"
echo "   wandb login"
echo "   python src/pretrain/train_mim.py"
echo ""
echo " Resume after interruption:"
echo "   python src/pretrain/train_mim.py --resume"
echo "========================================================"