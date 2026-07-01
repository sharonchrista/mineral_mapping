"""
src/finetune/train_finetune.py
Fine-tuning loop for mineral prospectivity mapping.
  Epochs 1-10 : encoders frozen
  Epochs 11+  : full end-to-end
  Loss        : weighted BCE (PU) + CCA correlation loss
  Metrics     : AUC-PR, AUC-ROC, F1, MCC

Fixes applied (Reviewer #1, Comment 1):
  1. pretrain_ckpt is now fold-specific:
       experiments/pretrain_mim_fold_aware/fold{k}/checkpoint_best.pt
     instead of a single shared checkpoint. Each fold's encoder was
     pre-trained without seeing that fold's test region.
  2. build_finetune_loaders now receives the correct fold masks
     (fold{k}_train_pos_mask.tif for training,
      fold{k}_test_mask.tif   for validation)
     instead of None, None.

Usage:
  python src/finetune/train_finetune.py --fold 0 --no_wandb
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
    "tif_path":             "data/processed/stacked/multiband_input.tif",
    "label_path":           "data/labels/pu_labels.tif",
    "splits_dir":           "data/splits",
    # FIX 1: pretrain_ckpt is now a template; fold index is substituted at runtime
    # "experiments/pretrain_mim_fold_aware/fold{fold}/checkpoint_best.pt"
    "pretrain_ckpt_template": "experiments/pretrain_mim_fold_aware/fold{fold}/checkpoint_best.pt",
    "output_dir":           "experiments/finetune",
    "img_size":             256,
    "patch_size":           16,
    "encoder_embed_dim":    768,
    "encoder_depth":        12,
    "encoder_num_heads":    12,
    "cca_dim":              256,
    "fusion_dim":           512,
    "mc_dropout":           0.2,
    "epochs":               30,
    "freeze_epochs":        10,
    "batch_size":           16,       # single-GPU safe
    "lr_head":              1e-3,
    "lr_encoder":           1e-6,
    "weight_decay":         0.01,
    "warmup_epochs":        5,
    "grad_clip":            1.0,
    "cca_lambda":           0.0,
    "unlabeled_ratio":      2.0,
    "num_workers":          4,
    "n_folds":              5,
    "fold":                 0,
    "save_every":           10,
    "log_every":            50,
    "use_wandb":            False,
    "wandb_project":        "mineral_mapping_finetune",
    "unfreeze_ramp_epochs": 5,
    "early_stop_patience": 10,
    "early_stop_min_delta": 0.001,
}


# ── Loss & metrics ────────────────────────────────────────────────────────────
def pu_bce_loss(logits, labels, weights):
    return (F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none") * weights).mean()


def compute_metrics(logits, labels):
    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return {"auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0, "mcc": 0.0,
                "best_threshold": 0.5}
    try:
        auc_pr  = average_precision_score(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
    except:
        auc_pr = auc_roc = 0.0

    # Find optimal threshold for F1 (standard for PU learning eval,
    # since the unlabeled weight biases probabilities low)
    best_f1, best_thresh, best_mcc = 0.0, 0.5, 0.0
    for thresh in np.linspace(0.01, 0.99, 50):
        preds = (probs >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        try:
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_mcc = matthews_corrcoef(labels, preds)
        except:
            continue

    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": best_f1,
            "mcc": best_mcc, "best_threshold": best_thresh}


# ── LR schedule ───────────────────────────────────────────────────────────────
def cosine_lr(optimizer, epoch, warmup, total, base_lrs, min_lr=1e-7):
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        if epoch < warmup:
            lr = blr * epoch / max(1, warmup)
        else:
            t  = (epoch - warmup) / max(1, total - warmup)
            lr = min_lr + 0.5 * (blr - min_lr) * (1 + math.cos(math.pi * t))
        pg["lr"] = lr


# ── Train / validate ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
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
            if logits.dim() > 1:
                logits = logits.squeeze()
            loss = pu_bce_loss(logits, labels, weights) + \
                   config["cca_lambda"] * cca_loss(out["geo_proj"], out["aero_proj"])

        scaler.scale(loss.mean()).backward()
        if config["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
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
    return {
        "train_loss": total_loss / max(1, len(loader)),
        "epoch_time": time.time() - t0,
        **{f"train_{k}": v for k, v in m.items()}
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        patches = batch["patch"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        weights = batch["weight"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            out    = model(patches)
            logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logits.dim() > 1:
                logits = logits.squeeze()
            loss = pu_bce_loss(logits, labels, weights)
        total_loss += loss.mean().item()
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    m = compute_metrics(
        np.concatenate(all_logits),
        np.concatenate(all_labels).astype(int)
    )
    return {
        "val_loss": total_loss / max(1, len(loader)),
        **{f"val_{k}": v for k, v in m.items()}
    }


# ── Per-fold training ─────────────────────────────────────────────────────────
def train_fold(config, fold, resume=False):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning  —  Fold {fold}/{config['n_folds']-1}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits_dir = Path(config["splits_dir"])
    out_dir    = Path(config["output_dir"]) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FIX 1: fold-matched encoder checkpoint
    pretrain_ckpt = config["pretrain_ckpt_template"].format(fold=fold)
    if not Path(pretrain_ckpt).exists():
        raise FileNotFoundError(
            f"Pre-training checkpoint not found: {pretrain_ckpt}\n"
            f"Run train_mim_per_fold.py --fold {fold} first."
        )
    print(f"Pre-train ckpt : {pretrain_ckpt}")

    # FIX 2: proper fold masks for train and validation splits
    # train_fold_mask : pixels OUTSIDE test strip (train region)
    # val_fold_mask   : pixels INSIDE  test strip (held-out test region)
    train_fold_mask = str(splits_dir / f"fold{fold}_train_mask.tif")
    val_fold_mask = str(splits_dir / f"fold{fold}_test_region_mask.tif")

    print(f"Train mask     : {train_fold_mask}")
    print(f"Val mask       : {val_fold_mask}")

    # Verify masks exist
    for p in [train_fold_mask, val_fold_mask]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Fold mask not found: {p}\n"
                f"Run src/preprocessing/06_generate_labels.py first."
            )

    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_finetune_loaders(
        tif_path        = config["tif_path"],
        label_path      = config["label_path"],
        train_fold_mask = train_fold_mask,
        val_fold_mask   = val_fold_mask,
        patch_size      = config["img_size"],
        batch_size      = config["batch_size"],
        num_workers     = config["num_workers"],
        unlabeled_ratio = config["unlabeled_ratio"],
    )

    print("\nBuilding model...")
    model = MineralProspectivityModel(
        pretrain_ckpt      = pretrain_ckpt,   # FIX 1: fold-matched
        img_size           = config["img_size"],
        patch_size         = config["patch_size"],
        encoder_embed_dim  = config["encoder_embed_dim"],
        encoder_depth      = config["encoder_depth"],
        encoder_num_heads  = config["encoder_num_heads"],
        cca_dim            = config["cca_dim"],
        fusion_dim         = config["fusion_dim"],
        mc_dropout         = config["mc_dropout"],
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_p/1e6:.1f}M")

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.freeze_encoders()

    enc_params  = (list(raw_model.geo_encoder.parameters()) +
                   list(raw_model.aero_encoder.parameters()))
    head_params = (list(raw_model.cross_attn.parameters()) +
                   list(raw_model.feat_branch.parameters()) +
                   list(raw_model.cca_fusion.parameters()) +
                   list(raw_model.depth_heads.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": config["lr_head"],
         "weight_decay": config["weight_decay"]},
        {"params": enc_params,  "lr": 0.0,
         "weight_decay": config["weight_decay"]},
    ])
    scaler   = GradScaler(enabled=device.type == "cuda")
    base_lrs = [config["lr_head"], config["lr_encoder"]]

    use_wandb = config["use_wandb"] and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project = config["wandb_project"],
            name    = f"fold{fold}_leakage_free",
            config  = {**config, "fold": fold, "pretrain_ckpt": pretrain_ckpt},
            reinit  = True,
        )

    best_auc = 0.0
    patience_counter = 0
    start_epoch = 1

    if resume:
        latest_ckpt = out_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            ckpt = torch.load(latest_ckpt, map_location=device)
            raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_auc = ckpt.get("val_auc_pr", 0.0)
            if start_epoch > config["freeze_epochs"]:
                raw_model.unfreeze_encoders()
                optimizer.param_groups[1]["lr"] = config["lr_encoder"]
            ep_num = ckpt["epoch"]
            print(f"  Resumed from epoch {ep_num}, best AUC-PR so far: {best_auc:.4f}")
        else:
            print(f"  No checkpoint found at {latest_ckpt}, starting fresh")

    print(f"\n{'Epoch':>6}  {'TrnLoss':>8}  {'ValLoss':>8}  "
          f"{'AUC-PR':>8}  {'AUC-ROC':>8}  {'F1':>6}  {'Time':>6}")
    print("-" * 64)

    for epoch in range(start_epoch, config["epochs"] + 1):

        # Unfreeze encoders after freeze_epochs, with gradual LR ramp
        if epoch == config["freeze_epochs"] + 1:
            raw_model.unfreeze_encoders()
            print(f"  Epoch {epoch}: encoders unfrozen, ramping LR over "
                  f"{config['unfreeze_ramp_epochs']} epochs")

        if epoch > config["freeze_epochs"]:
            ramp_epoch = epoch - config["freeze_epochs"]
            ramp_total = max(1, config["unfreeze_ramp_epochs"])
            ramp_frac  = min(1.0, ramp_epoch / ramp_total)
            optimizer.param_groups[1]["lr"] = config["lr_encoder"] * ramp_frac
        else:
            optimizer.param_groups[1]["lr"] = 0.0

        cosine_lr(optimizer, epoch, config["warmup_epochs"],
                  config["epochs"], base_lrs)
        # Re-apply ramp after cosine_lr so it doesn't override the ramp
        if epoch > config["freeze_epochs"]:
            ramp_epoch = epoch - config["freeze_epochs"]
            ramp_total = max(1, config["unfreeze_ramp_epochs"])
            ramp_frac  = min(1.0, ramp_epoch / ramp_total)
            optimizer.param_groups[1]["lr"] = config["lr_encoder"] * ramp_frac
        else:
            optimizer.param_groups[1]["lr"] = 0.0

        tm = train_one_epoch(model, train_loader, optimizer,
                             scaler, device, epoch, config)
        vm = validate(model, val_loader, device)

        print(f"{epoch:6d}  {tm['train_loss']:8.4f}  {vm['val_loss']:8.4f}  "
              f"{vm.get('val_auc_pr',0):8.4f}  {vm.get('val_auc_roc',0):8.4f}  "
              f"{vm.get('val_f1',0):6.4f}  {tm['epoch_time']:6.1f}s")

        if use_wandb:
            wandb.log({"epoch": epoch, "fold": fold, **tm, **vm})

        raw = model.module if hasattr(model, "module") else model
        state = {
            "epoch": epoch, "fold": fold,
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "pretrain_ckpt": pretrain_ckpt,
            **vm,
        }
        torch.save(state, out_dir / "checkpoint_latest.pt")

        if epoch % config["save_every"] == 0:
            torch.save(state, out_dir / f"checkpoint_epoch{epoch:04d}.pt")

        if vm.get("val_auc_pr", 0) > best_auc + config["early_stop_min_delta"]:
            best_auc = vm["val_auc_pr"]
            patience_counter = 0
            torch.save(state, out_dir / "checkpoint_best.pt")
            print(f"  ★ New best AUC-PR: {best_auc:.4f}")
        else:
            patience_counter += 1
            if epoch > config["freeze_epochs"] and patience_counter >= config["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break

    print(f"\nFold {fold} complete.  Best AUC-PR: {best_auc:.4f}")

    results = {
        "fold":              fold,
        "best_val_auc_pr":   best_auc,
        "pretrain_ckpt":     pretrain_ckpt,
        "train_mask":        train_fold_mask,
        "val_mask":          val_fold_mask,
        "config":            config,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if use_wandb:
        wandb.finish()

    return best_auc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fold-aware fine-tuning for mineral prospectivity mapping"
    )
    parser.add_argument("--fold",       type=int,  default=0)
    parser.add_argument("--all_folds",  action="store_true")
    parser.add_argument("--epochs",     type=int,  default=None)
    parser.add_argument("--batch_size", type=int,  default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.epochs:     config["epochs"]     = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.no_wandb:   config["use_wandb"]  = False

    if args.all_folds:
        results = []
        for f in range(config["n_folds"]):
            results.append(train_fold(config, f, resume=args.resume))
        print(f"\n{'='*40}")
        print(f"  All folds complete")
        print(f"{'='*40}")
        for f, r in enumerate(results):
            print(f"  Fold {f}: AUC-PR = {r:.4f}")
        print(f"  Mean : {np.mean(results):.4f} ± {np.std(results):.4f}")
    else:
        train_fold(config, fold=args.fold, resume=args.resume)