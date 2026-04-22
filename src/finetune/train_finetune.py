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
    "epochs": 30, "freeze_epochs": 10, "batch_size": 64,
    "lr_head": 1e-3, "lr_encoder": 1e-5, "weight_decay": 0.01,
    "warmup_epochs": 5, "grad_clip": 1.0, "cca_lambda": 0.0,
    "unlabeled_ratio": 2.0, "num_workers": 8,
    "n_folds": 5, "fold": 0, "save_every": 10, "log_every": 50,
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
        None,
        None,
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
