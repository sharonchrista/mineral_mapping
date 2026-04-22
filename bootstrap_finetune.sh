#!/bin/bash
# =============================================================================
# bootstrap_finetune.sh
# Deploys all fine-tuning scripts to ~/sharon/mineral_mapping/src/finetune/
# Run from: ~/sharon/mineral_mapping/
# Usage:    bash bootstrap_finetune.sh
# =============================================================================

set -e
BASE=~/sharon/mineral_mapping

echo "Creating folder structure..."
mkdir -p $BASE/src/finetune
mkdir -p $BASE/experiments/finetune
mkdir -p $BASE/results/prospectivity_maps
touch $BASE/src/finetune/__init__.py

# =============================================================================
# model_fusion.py
# =============================================================================
cat > $BASE/src/finetune/model_fusion.py << 'PYEOF'
"""
src/finetune/model_fusion.py
Multi-Modal Fusion Prospectivity Model (proposal section 4.3).
  Geological branch  : pre-trained ViT on channels 0-7
  Geophysical branch : pre-trained ViT on channels 8-11
  Feature branch     : MLP on channels 12-14
  CCA fusion         : deep CCA + 3-layer MLP
  Cross-attention    : geology <-> geophysics
  Depth heads        : 3 classifiers for 0-500m, 500-1000m, 1000-2000m
  MC Dropout         : p=0.2 for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pretrain.model import ViTEncoder

GEO_CHANNELS   = list(range(0, 8))
AERO_CHANNELS  = list(range(8, 12))
FEAT_CHANNELS  = list(range(12, 15))
DEPTH_INTERVALS = ["0_500m", "500_1000m", "1000_2000m"]


class DeepCCAFusion(nn.Module):
    def __init__(self, in_dim, cca_dim=256, out_dim=512, dropout=0.2):
        super().__init__()
        self.geo_proj = nn.Sequential(
            nn.Linear(in_dim, cca_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cca_dim*2, cca_dim), nn.LayerNorm(cca_dim),
        )
        self.aero_proj = nn.Sequential(
            nn.Linear(in_dim, cca_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cca_dim*2, cca_dim), nn.LayerNorm(cca_dim),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cca_dim*2, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),   nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),              nn.LayerNorm(out_dim),
        )

    def forward(self, geo, aero):
        gp = self.geo_proj(geo)
        ap = self.aero_proj(aero)
        return self.fusion_mlp(torch.cat([gp, ap], dim=-1)), gp, ap


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_geo = nn.Linear(dim, dim); self.k_aero = nn.Linear(dim, dim)
        self.v_aero = nn.Linear(dim, dim); self.out_geo = nn.Linear(dim, dim)
        self.q_aero = nn.Linear(dim, dim); self.k_geo = nn.Linear(dim, dim)
        self.v_geo = nn.Linear(dim, dim);  self.out_aero = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def _attn(self, q, k, v, B, N):
        H, D = self.num_heads, self.head_dim
        q = q.reshape(B,N,H,D).transpose(1,2); k = k.reshape(B,N,H,D).transpose(1,2)
        v = v.reshape(B,N,H,D).transpose(1,2)
        return (self.drop((q@k.transpose(-2,-1))*self.scale).softmax(dim=-1)@v).transpose(1,2).reshape(B,N,H*D)

    def forward(self, geo, aero):
        B, N, D = geo.shape
        geo_out  = self.norm1(geo  + self.out_geo( self._attn(self.q_geo(geo),   self.k_aero(aero), self.v_aero(aero), B, N)))
        aero_out = self.norm2(aero + self.out_aero(self._attn(self.q_aero(aero), self.k_geo(geo),   self.v_geo(geo),   B, N)))
        return geo_out, aero_out


class FeatureBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),     nn.GELU(), nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x.mean(dim=(-2,-1)))


class ProspectivityHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MineralProspectivityModel(nn.Module):
    GEO_CH = GEO_CHANNELS; AERO_CH = AERO_CHANNELS; FEAT_CH = FEAT_CHANNELS

    def __init__(self, pretrain_ckpt=None, img_size=256, patch_size=16,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 cca_dim=256, fusion_dim=512, feat_dim=128,
                 mc_dropout=0.2, n_depth_levels=3):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim

        self.geo_encoder  = ViTEncoder(img_size, patch_size, len(self.GEO_CH),
                                        encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.aero_encoder = ViTEncoder(img_size, patch_size, len(self.AERO_CH),
                                        encoder_embed_dim, encoder_depth, encoder_num_heads)
        self.cross_attn   = CrossModalAttention(encoder_embed_dim, 8, 0.1)
        self.feat_branch  = FeatureBranch(len(self.FEAT_CH), feat_dim, mc_dropout)
        self.cca_fusion   = DeepCCAFusion(encoder_embed_dim, cca_dim, fusion_dim, mc_dropout)
        self.depth_heads  = nn.ModuleList([
            ProspectivityHead(fusion_dim + feat_dim, 256, mc_dropout)
            for _ in range(n_depth_levels)
        ])
        if pretrain_ckpt is not None:
            self._load_pretrained(pretrain_ckpt)

    def _load_pretrained(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, training from scratch")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_state = {k.replace("encoder.",""): v
                     for k,v in ckpt["model"].items() if k.startswith("encoder.")}
        for enc, name in [(self.geo_encoder,"geo"), (self.aero_encoder,"aero")]:
            own = enc.state_dict(); n = 0
            for k, v in enc_state.items():
                if k not in own or "patch_embed.proj" in k: continue
                if own[k].shape != v.shape: continue
                own[k].copy_(v); n += 1
            enc.load_state_dict(own)
            print(f"  {name} encoder: {n} weight tensors loaded from pre-trained")

    def freeze_encoders(self):
        for p in list(self.geo_encoder.parameters()) + list(self.aero_encoder.parameters()):
            p.requires_grad = False
        print("  Encoders frozen")

    def unfreeze_encoders(self):
        for p in list(self.geo_encoder.parameters()) + list(self.aero_encoder.parameters()):
            p.requires_grad = True
        print("  Encoders unfrozen")

    def forward(self, x):
        geo_lat,  _, _ = self.geo_encoder( x[:, self.GEO_CH,  :, :], mask_ratio=0.0)
        aero_lat, _, _ = self.aero_encoder(x[:, self.AERO_CH, :, :], mask_ratio=0.0)
        geo_tok,  aero_tok  = self.cross_attn(geo_lat[:,1:,:], aero_lat[:,1:,:])
        geo_cls  = geo_lat[:, 0, :]
        aero_cls = aero_lat[:, 0, :]
        feat_out = self.feat_branch(x[:, self.FEAT_CH, :, :])
        fused, gp, ap = self.cca_fusion(geo_cls, aero_cls)
        combined = torch.cat([fused, feat_out], dim=-1)
        out = {"geo_proj": gp, "aero_proj": ap, "combined": combined}
        for head, depth in zip(self.depth_heads, DEPTH_INTERVALS):
            out[f"logits_{depth}"] = head(combined)
        return out

    def predict_proba(self, x, mc_passes=50):
        self.train()
        results = {d: [] for d in DEPTH_INTERVALS}
        with torch.no_grad():
            for _ in range(mc_passes):
                out = self.forward(x)
                for d in DEPTH_INTERVALS:
                    results[d].append(torch.sigmoid(out[f"logits_{d}"]))
        self.eval()
        return {d: {"mean": torch.stack(results[d],-1).mean(-1),
                    "std":  torch.stack(results[d],-1).std(-1),
                    "all_pass": torch.stack(results[d],-1)} for d in DEPTH_INTERVALS}


def cca_loss(geo_proj, aero_proj, reg=1e-4):
    B, D = geo_proj.shape
    gc = geo_proj  - geo_proj.mean(0, keepdim=True)
    ac = aero_proj - aero_proj.mean(0, keepdim=True)
    Sxy = (gc.T @ ac) / (B-1)
    Sxx = (gc.T @ gc) / (B-1) + reg * torch.eye(D, device=geo_proj.device)
    Syy = (ac.T @ ac) / (B-1) + reg * torch.eye(D, device=geo_proj.device)
    corr = torch.trace(Sxy) / (torch.sqrt(torch.trace(Sxx)) * torch.sqrt(torch.trace(Syy)) + 1e-8)
    return -corr


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MineralProspectivityModel(
        pretrain_ckpt="experiments/pretrain_mim/checkpoint_best.pt"
    ).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total:.1f}M")
    x = torch.randn(2, 15, 256, 256).to(device)
    out = model(x)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    print("Model OK")
PYEOF

echo "  OK model_fusion.py"

# =============================================================================
# dataset.py
# =============================================================================
cat > $BASE/src/finetune/dataset.py << 'PYEOF'
"""
src/finetune/dataset.py
PU-labeled dataset for prospectivity fine-tuning.
"""

import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import random


class ProspectivityDataset(Dataset):
    def __init__(self, tif_path, label_path, patch_size=256, augment=True,
                 fold_mask_path=None, unlabeled_ratio=5.0, noise_sigma=0.01):
        self.patch_size      = patch_size
        self.augment         = augment
        self.unlabeled_ratio = unlabeled_ratio
        self.noise_sigma     = noise_sigma
        self.half            = patch_size // 2

        with rasterio.open(tif_path) as src:
            self.data = src.read().astype("float32")
            self.H, self.W = src.height, src.width
        self.data = np.where(self.data == -9999.0, 0.0, self.data)

        with rasterio.open(label_path) as src:
            self.labels = src.read(1).astype("uint8")

        self.fold_mask = None
        if fold_mask_path is not None and Path(fold_mask_path).exists():
            with rasterio.open(fold_mask_path) as src:
                self.fold_mask = src.read(1).astype("uint8")

        self.pos_pixels, self.unl_pixels = self._find_pixels()
        self.samples = self._build_sample_list()

        print(f"ProspectivityDataset:")
        print(f"  Positive : {len(self.pos_pixels)} px")
        print(f"  Unlabeled: {len(self.unl_pixels)} px")
        print(f"  Samples  : {len(self.samples)}")

    def _find_pixels(self):
        H, W, P = self.H, self.W, self.half
        valid = np.zeros((H,W), dtype=bool)
        valid[P:H-P, P:W-P] = True
        if self.fold_mask is not None:
            valid = valid & (self.fold_mask > 0)
        pos = (self.labels==1) & valid
        unl = (self.labels==0) & valid
        pr, pc = np.where(pos); ur, uc = np.where(unl)
        return list(zip(pr.tolist(),pc.tolist())), list(zip(ur.tolist(),uc.tolist()))

    def _build_sample_list(self, seed=42):
        n_unl = min(len(self.unl_pixels), int(len(self.pos_pixels)*self.unlabeled_ratio))
        rng = np.random.default_rng(seed)
        sampled = self.unl_pixels if len(self.unl_pixels)<=n_unl else \
                  [self.unl_pixels[i] for i in rng.choice(len(self.unl_pixels),n_unl,replace=False)]
        samples = [(r,c,1,1.0) for r,c in self.pos_pixels] + \
                  [(r,c,0,0.1) for r,c in sampled]
        rng.shuffle(samples)
        return samples

    def resample(self, seed=None):
        self.samples = self._build_sample_list(seed=seed if seed is not None else 42)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        r,c,label,weight = self.samples[idx]
        H2 = self.half
        patch = torch.from_numpy(self.data[:, r-H2:r+H2, c-H2:c+H2].copy())
        if self.augment: patch = self._augment(patch)
        return {"patch": patch, "label": torch.tensor(label,dtype=torch.float32),
                "weight": torch.tensor(weight,dtype=torch.float32),
                "row": torch.tensor(r,dtype=torch.long),
                "col": torch.tensor(c,dtype=torch.long)}

    def _augment(self, patch):
        a = random.choice([0,90,180,270])
        if a: patch = TF.rotate(patch, a)
        if random.random()>0.5: patch = TF.hflip(patch)
        if random.random()>0.5: patch = TF.vflip(patch)
        if self.noise_sigma>0:
            patch = torch.clamp(patch + torch.randn_like(patch)*self.noise_sigma, 0.0, 1.0)
        return patch

    def get_sample_weights(self):
        return torch.tensor([s[3] for s in self.samples], dtype=torch.float32)


class FullRasterDataset(Dataset):
    def __init__(self, tif_path, patch_size=256, stride=None, min_valid=0.3):
        self.patch_size = patch_size
        self.half       = patch_size // 2
        self.stride     = stride if stride is not None else patch_size // 2
        with rasterio.open(tif_path) as src:
            self.data = src.read().astype("float32")
            self.H, self.W = src.height, src.width
            self.transform = src.transform; self.crs = src.crs
        self.data  = np.where(self.data==-9999.0, 0.0, self.data)
        self.tiles = self._build_index(min_valid)
        print(f"FullRasterDataset: {len(self.tiles)} inference tiles")

    def _build_index(self, min_valid):
        tiles = []
        H2, S = self.half, self.stride
        for r in range(H2, self.H-H2, S):
            for c in range(H2, self.W-H2, S):
                if (self.data[:, r-H2:r+H2, c-H2:c+H2].sum(0)!=0).mean() >= min_valid:
                    tiles.append((r,c))
        return tiles

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        r,c = self.tiles[idx]; H2 = self.half
        return {"patch": torch.from_numpy(self.data[:,r-H2:r+H2,c-H2:c+H2].copy()),
                "row": torch.tensor(r,dtype=torch.long),
                "col": torch.tensor(c,dtype=torch.long)}


def build_finetune_loaders(tif_path, label_path, train_fold_mask, val_fold_mask,
                            patch_size=256, batch_size=32, num_workers=8, unlabeled_ratio=5.0):
    train_ds = ProspectivityDataset(tif_path, label_path, patch_size, True,
                                     train_fold_mask, unlabeled_ratio)
    val_ds   = ProspectivityDataset(tif_path, label_path, patch_size, False,
                                     val_fold_mask, unlabeled_ratio)
    weights  = train_ds.get_sample_weights()
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=num_workers>0)
    return DataLoader(train_ds, sampler=sampler, **kw), DataLoader(val_ds, shuffle=False, **kw)
PYEOF

echo "  OK dataset.py"

# =============================================================================
# train_finetune.py
# =============================================================================
cat > $BASE/src/finetune/train_finetune.py << 'PYEOF'
"""
src/finetune/train_finetune.py
Fine-tuning loop for mineral prospectivity mapping.
  Epochs 1-10 : encoders frozen
  Epochs 11+  : full end-to-end
  Loss        : weighted BCE (PU) + CCA correlation loss
  Metrics     : AUC-PR, AUC-ROC, F1, MCC

Usage:
  python src/finetune/train_finetune.py --no_wandb
  python src/finetune/train_finetune.py --all_folds --no_wandb
"""

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finetune.model_fusion import MineralProspectivityModel, cca_loss, DEPTH_INTERVALS
from src.finetune.dataset import build_finetune_loaders

try:
    import wandb; WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEFAULT_CONFIG = {
    "tif_path":        "data/processed/stacked/multiband_input.tif",
    "label_path":      "data/labels/pu_labels.tif",
    "splits_dir":      "data/splits",
    "pretrain_ckpt":   "experiments/pretrain_mim/checkpoint_best.pt",
    "output_dir":      "experiments/finetune",
    "img_size":        256, "patch_size": 16,
    "encoder_embed_dim": 768, "encoder_depth": 12, "encoder_num_heads": 12,
    "cca_dim": 256, "fusion_dim": 512, "mc_dropout": 0.2,
    "epochs": 100, "freeze_epochs": 10, "batch_size": 32,
    "lr_head": 1e-3, "lr_encoder": 1e-5, "weight_decay": 0.01,
    "warmup_epochs": 5, "grad_clip": 1.0, "cca_lambda": 0.1,
    "unlabeled_ratio": 5.0, "num_workers": 8,
    "n_folds": 5, "fold": 0, "save_every": 10, "log_every": 20,
    "use_wandb": False, "wandb_project": "mineral_mapping_finetune",
}


def pu_bce_loss(logits, labels, weights):
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()


def compute_metrics(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    if labels.sum() == 0 or (labels==0).sum() == 0:
        return {"auc_pr":0.0,"auc_roc":0.0,"f1":0.0,"mcc":0.0,"best_f1":0.0}
    try:
        auc_pr  = average_precision_score(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
    except: auc_pr = auc_roc = 0.0
    preds = (probs>=0.5).astype(int)
    try:
        f1  = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
    except: f1 = mcc = 0.0
    return {"auc_pr":auc_pr,"auc_roc":auc_roc,"f1":f1,"mcc":mcc}


def cosine_lr(optimizer, epoch, warmup, total, base_lrs, min_lr=1e-7):
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup: lr = blr * epoch / max(1, warmup)
        else:
            t = (epoch-warmup)/max(1,total-warmup)
            lr = min_lr + 0.5*(blr-min_lr)*(1+math.cos(math.pi*t))
        pg["lr"] = lr


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0; all_logits = []; all_labels = []; t0 = time.time()
    if hasattr(loader.dataset, "resample"): loader.dataset.resample(seed=epoch)
    for i, batch in enumerate(loader):
        patches = batch["patch"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type=="cuda"):
            out    = model(patches)
            logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logits.dim()>1: logits = logits.squeeze()
            loss   = pu_bce_loss(logits, labels, weights) + \
                     config["cca_lambda"] * cca_loss(out["geo_proj"], out["aero_proj"])
        scaler.scale(loss.mean()).backward()
        if config["grad_clip"]>0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer); scaler.update()
        total_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())
        if i % config["log_every"] == 0:
            print(f"  Epoch {epoch:03d} [{i:3d}/{len(loader)}]  "
                  f"loss={loss.mean().item():.4f}  time={time.time()-t0:.0f}s")
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    m = compute_metrics(logits_np, labels_np)
    return {"train_loss": total_loss/max(1,len(loader)), "epoch_time": time.time()-t0,
            **{f"train_{k}":v for k,v in m.items()}}


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); total_loss=0.0; all_logits=[]; all_labels=[]
    for batch in loader:
        patches = batch["patch"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)
        with autocast(enabled=device.type=="cuda"):
            out    = model(patches)
            logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logits.dim()>1: logits=logits.squeeze()
            loss   = pu_bce_loss(logits, labels, weights)
        total_loss += loss.mean().item()
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())
    m = compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels).astype(int))
    return {"val_loss": total_loss/max(1,len(loader)), **{f"val_{k}":v for k,v in m.items()}}


def train_fold(config, fold):
    print(f"\n{'='*60}\n Fine-tuning Fold {fold}/{config['n_folds']-1}\n{'='*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")

    splits_dir = Path(config["splits_dir"])
    out_dir    = Path(config["output_dir"]) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_mask = splits_dir / f"fold{fold}_train_pos_mask.tif"
    val_mask   = splits_dir / f"fold{fold}_test_mask.tif"

    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_finetune_loaders(
        config["tif_path"], config["label_path"],
        train_mask if train_mask.exists() else None,
        val_mask   if val_mask.exists()   else None,
        config["img_size"], config["batch_size"],
        config["num_workers"], config["unlabeled_ratio"],
    )

    print("\nBuilding model...")
    model = MineralProspectivityModel(
        pretrain_ckpt=config["pretrain_ckpt"],
        img_size=config["img_size"], patch_size=config["patch_size"],
        encoder_embed_dim=config["encoder_embed_dim"],
        encoder_depth=config["encoder_depth"],
        encoder_num_heads=config["encoder_num_heads"],
        cca_dim=config["cca_dim"], fusion_dim=config["fusion_dim"],
        mc_dropout=config["mc_dropout"],
    ).to(device)
    if torch.cuda.device_count()>1: model = nn.DataParallel(model)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    raw_model = model.module if hasattr(model,"module") else model
    raw_model.freeze_encoders()

    enc_params  = list(raw_model.geo_encoder.parameters()) + list(raw_model.aero_encoder.parameters())
    head_params = list(raw_model.cross_attn.parameters()) + list(raw_model.feat_branch.parameters()) + \
                  list(raw_model.cca_fusion.parameters()) + list(raw_model.depth_heads.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config["lr_head"],    "weight_decay": config["weight_decay"]},
         {"params": enc_params,  "lr": 0.0,                  "weight_decay": config["weight_decay"]}]
    )
    scaler    = GradScaler(enabled=device.type=="cuda")
    base_lrs  = [config["lr_head"], config["lr_encoder"]]

    use_wandb = config["use_wandb"] and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=config["wandb_project"], name=f"fold{fold}",
                   config={**config,"fold":fold}, reinit=True)

    best_auc = 0.0
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Val Loss':>8}  {'AUC-PR':>8}  {'AUC-ROC':>8}  {'F1':>6}")
    print("-"*58)

    for epoch in range(1, config["epochs"]+1):
        if epoch == config["freeze_epochs"]+1:
            raw_model.unfreeze_encoders()
            optimizer.param_groups[1]["lr"] = config["lr_encoder"]
            print(f"  Epoch {epoch}: encoders unfrozen")
        cosine_lr(optimizer, epoch, config["warmup_epochs"], config["epochs"], base_lrs)
        tm = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        vm = validate(model, val_loader, device)
        print(f"{epoch:6d}  {tm['train_loss']:8.4f}  {vm['val_loss']:8.4f}  "
              f"{vm.get('val_auc_pr',0):8.4f}  {vm.get('val_auc_roc',0):8.4f}  "
              f"{vm.get('val_f1',0):6.4f}  {tm['epoch_time']:.1f}s")
        if use_wandb: wandb.log({"epoch":epoch,"fold":fold,**tm,**vm})
        raw = model.module if hasattr(model,"module") else model
        state = {"epoch":epoch,"fold":fold,"model":raw.state_dict(),
                 "optimizer":optimizer.state_dict(),"config":config,**vm}
        torch.save(state, out_dir/"checkpoint_latest.pt")
        if epoch % config["save_every"]==0:
            torch.save(state, out_dir/f"checkpoint_epoch{epoch:04d}.pt")
        if vm.get("val_auc_pr",0) > best_auc:
            best_auc = vm["val_auc_pr"]
            torch.save(state, out_dir/"checkpoint_best.pt")
            print(f"  New best AUC-PR: {best_auc:.4f}")

    print(f"\nFold {fold} done. Best AUC-PR: {best_auc:.4f}")
    with open(out_dir/"results.json","w") as f:
        json.dump({"fold":fold,"best_val_auc_pr":best_auc,"config":config},f,indent=2)
    if use_wandb: wandb.finish()
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--all_folds",  action="store_true")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.epochs:     config["epochs"]     = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.no_wandb:   config["use_wandb"]  = False
    if args.all_folds:
        results = [train_fold(config, f) for f in range(config["n_folds"])]
        print(f"\nAll folds: mean AUC-PR = {np.mean(results):.4f} +/- {np.std(results):.4f}")
    else:
        train_fold(config, fold=args.fold)
PYEOF

echo "  OK train_finetune.py"

# =============================================================================
# predict_map.py
# =============================================================================
cat > $BASE/src/finetune/predict_map.py << 'PYEOF'
"""
src/finetune/predict_map.py
Generate full prospectivity map GeoTIFFs with MC uncertainty.
Outputs per depth: prospectivity_mean_<depth>.tif, prospectivity_uncertainty_<depth>.tif
Plus: prospectivity_ensemble_mean.tif

Usage:
  python src/finetune/predict_map.py --fold 0
  python src/finetune/predict_map.py --fold 0 --mc_passes 50
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import rasterio
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finetune.model_fusion import MineralProspectivityModel, DEPTH_INTERVALS
from src.finetune.dataset import FullRasterDataset

console = Console()


def save_map(arr, out_path, meta, desc=""):
    profile = {"driver":"GTiff","dtype":"float32","crs":meta["crs"],
               "transform":meta["transform"],"width":meta["width"],
               "height":meta["height"],"count":1,"compress":"lzw",
               "tiled":True,"nodata":-9999.0}
    arr_out = np.where(np.isnan(arr), -9999.0, arr).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_out, 1)
        if desc: dst.update_tags(description=desc)
    valid = arr[~np.isnan(arr)]
    console.print(f"  Saved: {out_path.name}  [min={valid.min():.3f} max={valid.max():.3f}]")


def predict_map(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"]) / f"fold{config['fold']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(config["ckpt_dir"]) / f"fold{config['fold']}" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_path}[/red]"); return

    console.print(f"Loading: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = ckpt["config"]
    model = MineralProspectivityModel(
        pretrain_ckpt=None, img_size=cfg.get("img_size",256),
        patch_size=cfg.get("patch_size",16),
        encoder_embed_dim=cfg.get("encoder_embed_dim",768),
        encoder_depth=cfg.get("encoder_depth",12),
        encoder_num_heads=cfg.get("encoder_num_heads",12),
        cca_dim=cfg.get("cca_dim",256), fusion_dim=cfg.get("fusion_dim",512),
        mc_dropout=cfg.get("mc_dropout",0.2),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    console.print(f"  Epoch {ckpt['epoch']}  val_auc_pr={ckpt.get('val_auc_pr','N/A')}")

    ds = FullRasterDataset(config["tif_path"], config["patch_size"], config["patch_size"]//2)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=True)

    with rasterio.open(config["tif_path"]) as src:
        meta = {"crs":src.crs,"transform":src.transform,"height":src.height,"width":src.width}
    H, W, half = meta["height"], meta["width"], config["patch_size"]//2

    mean_maps = {d: np.zeros((H,W), dtype="float64") for d in DEPTH_INTERVALS}
    std_maps  = {d: np.zeros((H,W), dtype="float64") for d in DEPTH_INTERVALS}
    count_map = np.zeros((H,W), dtype="float64")

    for batch in track(loader, description="Predicting..."):
        patches = batch["patch"].to(device)
        rows    = batch["row"].numpy()
        cols    = batch["col"].numpy()
        mc      = model.predict_proba(patches, mc_passes=config["mc_passes"])
        for i in range(len(rows)):
            r,c = int(rows[i]), int(cols[i])
            r0,r1,c0,c1 = r-half,r+half,c-half,c+half
            count_map[r0:r1,c0:c1] += 1.0
            for d in DEPTH_INTERVALS:
                mean_maps[d][r0:r1,c0:c1] += mc[d]["mean"][i].item()
                std_maps[d][r0:r1,c0:c1]  += mc[d]["std"][i].item()

    cnt = np.where(count_map>0, count_map, np.nan)
    for d in DEPTH_INTERVALS:
        mean_maps[d] /= cnt; std_maps[d] /= cnt

    console.rule("Saving maps")
    for d in DEPTH_INTERVALS:
        save_map(mean_maps[d].astype("float32"), out_dir/f"prospectivity_mean_{d}.tif",   meta, f"Prospectivity mean depth={d}")
        save_map(std_maps[d].astype("float32"),  out_dir/f"prospectivity_uncertainty_{d}.tif", meta, f"Uncertainty depth={d}")

    ensemble = np.nanmean(np.stack([mean_maps[d] for d in DEPTH_INTERVALS],0), 0)
    save_map(ensemble.astype("float32"), out_dir/"prospectivity_ensemble_mean.tif", meta, "Ensemble mean")
    console.print(f"\nAll maps -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--mc_passes",  type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()
    predict_map({
        "tif_path":   "data/processed/stacked/multiband_input.tif",
        "output_dir": "results/prospectivity_maps",
        "ckpt_dir":   "experiments/finetune",
        "patch_size": 256, "batch_size": args.batch_size,
        "num_workers": args.workers, "mc_passes": args.mc_passes,
        "fold": args.fold,
    })
PYEOF

echo "  OK predict_map.py"

echo ""
echo "========================================================"
echo " All fine-tuning scripts created:"
echo "   src/finetune/model_fusion.py"
echo "   src/finetune/dataset.py"
echo "   src/finetune/train_finetune.py"
echo "   src/finetune/predict_map.py"
echo ""
echo " Sanity check:"
echo "   python src/finetune/model_fusion.py"
echo ""
echo " Train fold 0:"
echo "   nohup python src/finetune/train_finetune.py --fold 0 --no_wandb > experiments/finetune/fold0.log 2>&1 &"
echo "   echo PID: \$!"
echo ""
echo " Train all 5 folds:"
echo "   nohup python src/finetune/train_finetune.py --all_folds --no_wandb > experiments/finetune/all_folds.log 2>&1 &"
echo "   echo PID: \$!"
echo ""
echo " Generate prospectivity map (after training):"
echo "   python src/finetune/predict_map.py --fold 0 --mc_passes 50"
echo "========================================================"