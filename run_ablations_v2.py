"""
run_ablations_v2.py

Corrected, leakage-free ablation study for MineralProspectivityModel.

Fixes applied relative to original run_ablations.py:
  1. Uses fold-aware pre-trained encoder
     (experiments/pretrain_mim_fold_aware/fold0/checkpoint_best.pt)
     instead of the leaked single shared checkpoint.
  2. Uses correct train/val fold masks
     (fold0_train_mask.tif / fold0_test_region_mask.tif)
     instead of unfiltered raster access.
  3. Uses gradual encoder unfreeze (matches corrected train_finetune.py)
     to avoid the unfreeze instability observed in the main fine-tuning runs.
  4. Uses optimal-threshold F1 (matches corrected compute_metrics)
     instead of fixed 0.5 threshold, which previously produced F1=0.0000
     under PU weighting.
  5. Early stopping (patience=10) instead of fixed 30 epochs, consistent
     with the main fine-tuning protocol.

Variants tested (same as original):
  Full model    : complete architecture (baseline for comparison)
  A1 — no_cross_attn   : skip CrossModalAttention, use raw CLS tokens
  A2 — no_cca          : replace DeepCCAFusion with simple concat MLP
  A3 — geo_only        : geological branch only (no aero encoder)
  A4 — aero_only       : geophysical branch only (no geo encoder)
  A5 — no_pretrain     : randomly initialized encoders (ViT-NoPT)

All variants are run on the SAME fold with the SAME leakage-free
train/val split, for direct comparability.

Usage:
  cd ~/sharon/mineral_mapping
  nohup python run_ablations_v2.py --fold 0 > experiments/ablation_studies_v2/ablation_fold0.log 2>&1 &
  nohup python run_ablations_v2.py --fold 2 > experiments/ablation_studies_v2/ablation_fold2.log 2>&1 &
"""

import os, sys, json, time, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef

sys.path.insert(0, ".")
from src.pretrain.model import ViTEncoder
from src.finetune.dataset import build_finetune_loaders
from src.finetune.model_fusion import (
    CrossModalAttention, DeepCCAFusion,
    FeatureBranch, MineralProspectivityModel,
    GEO_CHANNELS, AERO_CHANNELS, FEAT_CHANNELS, DEPTH_INTERVALS
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(".")

def make_config(fold):
    return {
        "tif_path"           : "data/processed/stacked/multiband_input.tif",
        "label_path"         : "data/labels/pu_labels.tif",
        "splits_dir"         : "data/splits",
        "pretrain_ckpt"      : f"experiments/pretrain_mim_fold_aware/fold{fold}/checkpoint_best.pt",
        "train_fold_mask"    : f"data/splits/fold{fold}_train_mask.tif",
        "val_fold_mask"      : f"data/splits/fold{fold}_test_region_mask.tif",
        "img_size"           : 256,
        "patch_size"         : 16,
        "encoder_embed_dim"  : 768,
        "encoder_depth"      : 12,
        "encoder_num_heads"  : 12,
        "cca_dim"            : 256,
        "fusion_dim"         : 512,
        "feat_dim"           : 128,
        "mc_dropout"         : 0.2,
        "batch_size"         : 32,
        "epochs"             : 100,
        "freeze_epochs"      : 10,
        "unfreeze_ramp_epochs": 5,
        "lr_head"            : 1e-3,
        "lr_encoder"         : 1e-6,
        "weight_decay"       : 0.01,
        "warmup_epochs"      : 5,
        "grad_clip"          : 1.0,
        "unlabeled_ratio"    : 2.0,
        "pos_weight"         : 1.0,
        "unl_weight"         : 0.1,
        "fold"               : fold,
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.001,
        "log_every"          : 100,
        "num_workers"        : 4,
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("experiments/ablation_studies_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Loss ──────────────────────────────────────────────────────────────────────
def pu_bce_loss(logits, labels, weights):
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return (loss * weights).mean()


# ── Metrics (optimal threshold F1, matches train_finetune.py fix) ────────────
def compute_metrics(logits, labels):
    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    labels = np.array(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return {"auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0, "mcc": 0.0}

    try:
        auc_pr  = average_precision_score(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
    except Exception:
        auc_pr = auc_roc = 0.0

    best_f1, best_mcc = 0.0, 0.0
    for thresh in np.linspace(0.01, 0.99, 50):
        preds = (probs >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        try:
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_mcc = matthews_corrcoef(labels, preds)
        except Exception:
            continue

    return {"auc_pr": float(auc_pr), "auc_roc": float(auc_roc),
            "f1": float(best_f1), "mcc": float(best_mcc)}


# ── LR schedule ───────────────────────────────────────────────────────────────
def cosine_lr(optimizer, epoch, warmup, total, base_lrs, min_lr=1e-7):
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup:
            lr = blr * epoch / max(1, warmup)
        else:
            t  = (epoch - warmup) / max(1, total - warmup)
            lr = min_lr + 0.5 * (blr - min_lr) * (1 + math.cos(math.pi * t))
        pg["lr"] = lr


# ── Variant model builder ─────────────────────────────────────────────────────
def build_variant_model(variant, config):
    """
    Builds a MineralProspectivityModel variant.
    variant in {"full", "no_cross_attn", "no_cca", "geo_only",
                "aero_only", "no_pretrain"}
    """
    use_pretrain = (variant != "no_pretrain")
    pretrain_ckpt = config["pretrain_ckpt"] if use_pretrain else None

    model = MineralProspectivityModel(
        pretrain_ckpt      = pretrain_ckpt,
        img_size           = config["img_size"],
        patch_size         = config["patch_size"],
        encoder_embed_dim  = config["encoder_embed_dim"],
        encoder_depth      = config["encoder_depth"],
        encoder_num_heads  = config["encoder_num_heads"],
        cca_dim            = config["cca_dim"],
        fusion_dim         = config["fusion_dim"],
        mc_dropout         = config["mc_dropout"],
    )

    # Apply structural ablations by disabling/bypassing components
    if variant == "no_cross_attn":
        class IdentityCrossAttn(nn.Module):
            def forward(self, geo_tok, aero_tok):
                return geo_tok, aero_tok
        model.cross_attn = IdentityCrossAttn()

    elif variant == "no_cca":
        embed_dim  = config["encoder_embed_dim"]
        fusion_dim = config["fusion_dim"]
        class SimpleConcatFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(embed_dim * 2, fusion_dim),
                    nn.GELU(),
                    nn.LayerNorm(fusion_dim),
                )
            def forward(self, geo_cls, aero_cls):
                fused = self.proj(torch.cat([geo_cls, aero_cls], dim=-1))
                return fused, geo_cls, aero_cls
        model.cca_fusion = SimpleConcatFusion()

    elif variant == "geo_only":
        def geo_only_forward(x):
            geo_lat, _, _ = model.geo_encoder(x[:, model.GEO_CH, :, :], mask_ratio=0.0)
            geo_cls = geo_lat[:, 0, :]
            zero_aero_cls = torch.zeros_like(geo_cls)
            feat_out = model.feat_branch(x[:, model.FEAT_CH, :, :])
            fused, gp, ap = model.cca_fusion(geo_cls, zero_aero_cls)
            combined = torch.cat([fused, feat_out], dim=-1)
            out = {"geo_proj": gp, "aero_proj": ap, "combined": combined}
            for head, depth in zip(model.depth_heads, DEPTH_INTERVALS):
                out[f"logits_{depth}"] = head(combined)
            return out
        model.forward = geo_only_forward

    elif variant == "aero_only":
        def aero_only_forward(x):
            aero_lat, _, _ = model.aero_encoder(x[:, model.AERO_CH, :, :], mask_ratio=0.0)
            aero_cls = aero_lat[:, 0, :]
            zero_geo_cls = torch.zeros_like(aero_cls)
            feat_out = model.feat_branch(x[:, model.FEAT_CH, :, :])
            fused, gp, ap = model.cca_fusion(zero_geo_cls, aero_cls)
            combined = torch.cat([fused, feat_out], dim=-1)
            out = {"geo_proj": gp, "aero_proj": ap, "combined": combined}
            for head, depth in zip(model.depth_heads, DEPTH_INTERVALS):
                out[f"logits_{depth}"] = head(combined)
            return out
        model.forward = aero_only_forward

    # "full" and "no_pretrain" use the unmodified architecture

    return model.to(DEVICE)


# ── Training loop for one variant ────────────────────────────────────────────
def train_variant(variant, config, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"  Ablation variant: {variant}")
    print(f"{'='*60}")

    model = build_variant_model(variant, config)
    model.freeze_encoders()

    enc_params  = (list(model.geo_encoder.parameters()) +
                   list(model.aero_encoder.parameters()))
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("geo_encoder") and not n.startswith("aero_encoder")]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": config["lr_head"], "weight_decay": config["weight_decay"]},
        {"params": enc_params,  "lr": 0.0,                "weight_decay": config["weight_decay"]},
    ])
    scaler   = GradScaler(enabled=DEVICE.type == "cuda")
    base_lrs = [config["lr_head"], config["lr_encoder"]]

    best_auc = 0.0
    patience_counter = 0
    out_dir = OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):

        if epoch == config["freeze_epochs"] + 1:
            model.unfreeze_encoders()
            print(f"  Epoch {epoch}: encoders unfrozen, gradual ramp")

        cosine_lr(optimizer, epoch, config["warmup_epochs"], config["epochs"], base_lrs)

        if epoch > config["freeze_epochs"]:
            ramp_epoch = epoch - config["freeze_epochs"]
            ramp_frac  = min(1.0, ramp_epoch / max(1, config["unfreeze_ramp_epochs"]))
            optimizer.param_groups[1]["lr"] = config["lr_encoder"] * ramp_frac
        else:
            optimizer.param_groups[1]["lr"] = 0.0

        # Train
        model.train()
        total_loss = 0.0
        t0 = time.time()
        if hasattr(train_loader.dataset, "resample"):
            train_loader.dataset.resample(seed=epoch)

        for i, batch in enumerate(train_loader):
            patches = batch["patch"].to(DEVICE, non_blocking=True)
            labels  = batch["label"].to(DEVICE, non_blocking=True)
            weights = batch["weight"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type == "cuda"):
                out = model(patches)
                logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
                if logits.dim() > 1:
                    logits = logits.squeeze()
                loss = pu_bce_loss(logits, labels, weights)

            scaler.scale(loss).backward()
            if config["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if i % config["log_every"] == 0:
                print(f"    [{variant}] Epoch {epoch:03d} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")

        train_loss = total_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        all_logits, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                patches = batch["patch"].to(DEVICE, non_blocking=True)
                labels  = batch["label"].to(DEVICE, non_blocking=True)
                weights = batch["weight"].to(DEVICE, non_blocking=True)
                with autocast(enabled=DEVICE.type == "cuda"):
                    out = model(patches)
                    logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
                    if logits.dim() > 1:
                        logits = logits.squeeze()
                    loss = pu_bce_loss(logits, labels, weights)
                val_loss += loss.item()
                all_logits.append(logits.cpu().float().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        m = compute_metrics(np.concatenate(all_logits), np.concatenate(all_labels))

        print(f"  [{variant}] Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  AUC-PR={m['auc_pr']:.4f}  "
              f"AUC-ROC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
              f"time={time.time()-t0:.0f}s")

        if m["auc_pr"] > best_auc + config["early_stop_min_delta"]:
            best_auc = m["auc_pr"]
            patience_counter = 0
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"variant": variant, "epoch": epoch, **m,
                          "fold": config["fold"]}, f, indent=2)
        else:
            patience_counter += 1
            if epoch > config["freeze_epochs"] and patience_counter >= config["early_stop_patience"]:
                print(f"  [{variant}] Early stopping at epoch {epoch}")
                break

    print(f"\n  [{variant}] FINAL best AUC-PR: {best_auc:.4f}")
    return best_auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--variants", nargs="+",
                        default=["full", "no_cross_attn", "no_cca",
                                "geo_only", "aero_only", "no_pretrain"])
    args = parser.parse_args()

    config = make_config(args.fold)

    print(f"Building dataloaders for fold {args.fold} (leakage-free)...")
    train_loader, val_loader = build_finetune_loaders(
        tif_path        = config["tif_path"],
        label_path      = config["label_path"],
        train_fold_mask = config["train_fold_mask"],
        val_fold_mask   = config["val_fold_mask"],
        patch_size      = config["img_size"],
        batch_size      = config["batch_size"],
        num_workers     = config["num_workers"],
        unlabeled_ratio = config["unlabeled_ratio"],
    )

    results = {}
    for variant in args.variants:
        try:
            best_auc = train_variant(variant, config, train_loader, val_loader)
            results[variant] = best_auc
        except Exception as e:
            print(f"  [{variant}] FAILED: {e}")
            results[variant] = None

    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY (Fold {args.fold}, leakage-free)")
    print(f"{'='*60}")
    full_auc = results.get("full", None)
    for variant, auc in results.items():
        if auc is None:
            print(f"  {variant:20s}: FAILED")
        elif variant == "full":
            print(f"  {variant:20s}: {auc:.4f}  (baseline)")
        else:
            drop = full_auc - auc if full_auc else float("nan")
            print(f"  {variant:20s}: {auc:.4f}  (drop: {drop:+.4f})")

    with open(OUT_DIR / f"summary_fold{args.fold}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {OUT_DIR / f'summary_fold{args.fold}.json'}")


if __name__ == "__main__":
    main()