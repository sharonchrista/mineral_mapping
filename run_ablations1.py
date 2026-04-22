"""
run_ablations.py
Ablation study for MineralProspectivityModel.
Runs 4 variants sequentially on fold 0, saves metrics to
experiments/ablation_studies/<variant>/metrics.json

Variants:
  A1 — no_cross_attn   : skip CrossModalAttention, use raw CLS tokens
  A2 — no_cca          : replace DeepCCAFusion with simple concat MLP
  A3 — geo_only        : geological branch only (no aero encoder)
  A4 — aero_only       : geophysical branch only (no geo encoder)

Run:
  cd ~/sharon/mineral_mapping
  source activate.sh
  nohup python run_ablations.py > experiments/ablation_studies/ablation.log 2>&1 &
"""

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, ".")
from src.pretrain.model import ViTEncoder
from src.finetune.dataset import ProspectivityDataset
from src.finetune.model_fusion import (
    CrossModalAttention, DeepCCAFusion,
    FeatureBranch, ProspectivityHead,
    GEO_CHANNELS, AERO_CHANNELS, FEAT_CHANNELS, DEPTH_INTERVALS
)
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
BASE   = Path(".")
CONFIG = {
    "tif_path"         : "data/processed/stacked/multiband_input.tif",
    "label_path"       : "data/labels/pu_labels.tif",
    "pretrain_ckpt"    : "experiments/pretrain_mim/checkpoint_best.pt",
    "img_size"         : 256,
    "patch_size"       : 16,
    "encoder_embed_dim": 768,
    "encoder_depth"    : 12,
    "encoder_num_heads": 12,
    "cca_dim"          : 256,
    "fusion_dim"       : 512,
    "feat_dim"         : 128,
    "mc_dropout"       : 0.2,
    "batch_size"       : 64,
    "epochs"           : 30,
    "freeze_epochs"    : 10,
    "lr_head"          : 1e-3,
    "lr_encoder"       : 1e-5,
    "unlabeled_ratio"  : 2.0,
    "pos_weight"       : 1.0,
    "unl_weight"       : 0.1,
    "fold"             : 0,
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU, no NCCL needed

# ── Loss ──────────────────────────────────────────────────────────────────────
def pu_bce_loss(logits, labels, weights):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none")
    return (loss * weights).mean()

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(logits, labels):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return {"auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0}
    return {
        "auc_pr" : float(average_precision_score(labels, probs)),
        "auc_roc": float(roc_auc_score(labels, probs)),
        "f1"     : float(f1_score(labels, (probs >= 0.5).astype(int),
                                  zero_division=0)),
    }

# ── Load pretrained weights into encoder ──────────────────────────────────────
def load_pretrained(encoder, ckpt_path, name="enc"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_state = {k.replace("encoder.", ""): v
                 for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    own = encoder.state_dict(); n = 0
    for k, v in enc_state.items():
        if k not in own or "patch_embed.proj" in k:
            continue
        if own[k].shape != v.shape:
            continue
        own[k].copy_(v); n += 1
    encoder.load_state_dict(own)
    print(f"  [{name}] loaded {n} pretrained tensors")

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION MODEL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class AblationModel(nn.Module):
    """
    Unified ablation model controlled by flags:
      use_cross_attn : bool  — whether to apply CrossModalAttention
      use_cca        : bool  — whether to use DeepCCAFusion (False = concat MLP)
      use_geo        : bool  — whether to include geological branch
      use_aero       : bool  — whether to include geophysical branch
    """
    def __init__(self, cfg, use_cross_attn=True, use_cca=True,
                 use_geo=True, use_aero=True):
        super().__init__()
        D   = cfg["encoder_embed_dim"]
        cdim = cfg["cca_dim"]
        fdim = cfg["fusion_dim"]
        feat = cfg["feat_dim"]
        drop = cfg["mc_dropout"]

        self.use_cross_attn = use_cross_attn
        self.use_cca        = use_cca
        self.use_geo        = use_geo
        self.use_aero       = use_aero

        # Encoders — only build what we need
        if use_geo:
            self.geo_encoder = ViTEncoder(
                cfg["img_size"], cfg["patch_size"], len(GEO_CHANNELS),
                D, cfg["encoder_depth"], cfg["encoder_num_heads"])
            load_pretrained(self.geo_encoder,
                            cfg["pretrain_ckpt"], "geo")

        if use_aero:
            self.aero_encoder = ViTEncoder(
                cfg["img_size"], cfg["patch_size"], len(AERO_CHANNELS),
                D, cfg["encoder_depth"], cfg["encoder_num_heads"])
            load_pretrained(self.aero_encoder,
                            cfg["pretrain_ckpt"], "aero")

        # Cross-modal attention (only meaningful if both branches present)
        if use_cross_attn and use_geo and use_aero:
            self.cross_attn = CrossModalAttention(D, 8, 0.1)

        # Feature branch always included
        self.feat_branch = FeatureBranch(len(FEAT_CHANNELS), feat, drop)

        # Fusion — determine input dim based on active branches
        n_branches = int(use_geo) + int(use_aero)
        if use_cca and n_branches == 2:
            # Standard deep CCA fusion
            self.cca_fusion = DeepCCAFusion(D, cdim, fdim, drop)
            head_in = fdim + feat
        else:
            # Simple concat MLP (no CCA projection)
            concat_dim = D * n_branches
            self.concat_mlp = nn.Sequential(
                nn.Linear(concat_dim, fdim), nn.GELU(),
                nn.LayerNorm(fdim), nn.Dropout(drop),
                nn.Linear(fdim, fdim), nn.GELU(),
                nn.LayerNorm(fdim),
            )
            head_in = fdim + feat

        self.depth_heads = nn.ModuleList([
            ProspectivityHead(head_in, 256, drop)
            for _ in range(3)
        ])

    def freeze_encoders(self):
        params = []
        if self.use_geo:  params += list(self.geo_encoder.parameters())
        if self.use_aero: params += list(self.aero_encoder.parameters())
        for p in params: p.requires_grad = False

    def unfreeze_encoders(self):
        params = []
        if self.use_geo:  params += list(self.geo_encoder.parameters())
        if self.use_aero: params += list(self.aero_encoder.parameters())
        for p in params: p.requires_grad = True

    def forward(self, x):
        # Encode active branches
        if self.use_geo:
            geo_lat,  _, _ = self.geo_encoder(
                x[:, GEO_CHANNELS, :, :], mask_ratio=0.0)
            geo_cls = geo_lat[:, 0, :]
        if self.use_aero:
            aero_lat, _, _ = self.aero_encoder(
                x[:, AERO_CHANNELS, :, :], mask_ratio=0.0)
            aero_cls = aero_lat[:, 0, :]

        # Cross-modal attention on patch tokens
        if self.use_cross_attn and self.use_geo and self.use_aero:
            _, _ = self.cross_attn(
                geo_lat[:, 1:, :], aero_lat[:, 1:, :])

        # Feature branch
        feat_out = self.feat_branch(x[:, FEAT_CHANNELS, :, :])

        # Fusion
        if self.use_cca and self.use_geo and self.use_aero:
            fused, _, _ = self.cca_fusion(geo_cls, aero_cls)
        elif self.use_geo and self.use_aero:
            # Simple concat MLP (no CCA)
            fused = self.concat_mlp(torch.cat([geo_cls, aero_cls], dim=-1))
        elif self.use_geo:
            fused = self.concat_mlp(geo_cls)
        else:
            fused = self.concat_mlp(aero_cls)

        combined = torch.cat([fused, feat_out], dim=-1)
        out = {}
        for head, depth in zip(self.depth_heads, DEPTH_INTERVALS):
            out[f"logits_{depth}"] = head(combined)
        return out

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def build_loader(cfg, split="train"):
    fold = cfg["fold"]
    ds = ProspectivityDataset(
        tif_path        = cfg["tif_path"],
        label_path      = cfg["label_path"],
        patch_size      = cfg["img_size"],
        augment         = (split == "train"),
        unlabeled_ratio = cfg["unlabeled_ratio"],
    )
    return DataLoader(ds, batch_size=cfg["batch_size"],
                      shuffle=(split == "train"),
                      num_workers=4, pin_memory=True,
                      drop_last=(split == "train"))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    depth = DEPTH_INTERVALS[0]
    for batch in loader:
        patches = batch["patch"].to(DEVICE)
        labels  = batch["label"].to(DEVICE)
        weights = batch["weight"].to(DEVICE)
        out     = model(patches)
        logits  = out[f"logits_{depth}"]
        if logits.dim() > 1: logits = logits.squeeze()
        loss = pu_bce_loss(logits, labels, weights)
        total_loss += loss.item()
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    m = compute_metrics(logits_np, labels_np)
    m["val_loss"] = total_loss / max(1, len(loader))
    return m

def train_ablation(variant_name, model_kwargs, cfg):
    print(f"\n{'='*60}")
    print(f"  ABLATION: {variant_name}")
    print(f"{'='*60}")
    out_dir = Path(f"experiments/ablation_studies/{variant_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single GPU only
    model = AblationModel(cfg, **model_kwargs).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train_loader = build_loader(cfg, "train")
    val_loader   = build_loader(cfg, "val")

    scaler = GradScaler()
    depth  = DEPTH_INTERVALS[0]

    # Phase 1: freeze encoders
    m = model
    m.freeze_encoders()

    # Optimizer - will be recreated at unfreeze
    def make_opt(unfreeze=False):
        head_params = [p for n, p in model.named_parameters()
                       if "encoder" not in n and p.requires_grad]
        if not unfreeze:
            return torch.optim.AdamW(head_params, lr=cfg["lr_head"],
                                     weight_decay=0.05)
        enc_params = [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
        return torch.optim.AdamW([
            {"params": enc_params,  "lr": cfg["lr_encoder"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ], weight_decay=0.05)

    optimizer = make_opt(False)
    best_auc  = 0.0
    log_rows  = []

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch == cfg["freeze_epochs"] + 1:
            m.unfreeze_encoders()
            optimizer = make_opt(True)
            print(f"  [Epoch {epoch}] Encoders unfrozen")

        model.train()
        t0 = time.time()
        for batch in train_loader:
            patches = batch["patch"].to(DEVICE, non_blocking=True)
            labels  = batch["label"].to(DEVICE, non_blocking=True)
            weights = batch["weight"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                out    = model(patches)
                logits = out[f"logits_{depth}"]
                if logits.dim() > 1: logits = logits.squeeze()
                loss   = pu_bce_loss(logits, labels, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        vm = evaluate(model, val_loader)
        elapsed = time.time() - t0
        row = {"epoch": epoch, **vm}
        log_rows.append(row)

        print(f"  Ep {epoch:03d}  "
              f"auc_pr={vm['auc_pr']:.4f}  "
              f"auc_roc={vm['auc_roc']:.4f}  "
              f"f1={vm['f1']:.4f}  "
              f"val_loss={vm['val_loss']:.4f}  "
              f"t={elapsed:.0f}s")

        state = {
            "epoch": epoch, "variant": variant_name,
            "config": cfg, "model": model.state_dict(),
            **vm
        }
        torch.save(state, out_dir / "checkpoint_latest.pt")
        if vm["auc_pr"] > best_auc or epoch == 1:
            best_auc = vm["auc_pr"]
            torch.save(state, out_dir / "checkpoint_best.pt")

    # Save final metrics
    final = {
        "variant"  : variant_name,
        "fold"     : cfg["fold"],
        "epochs"   : cfg["epochs"],
        "best_auc_pr" : best_auc,
        "final"    : vm,
        "log"      : log_rows,
        "model_kwargs": model_kwargs,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"  DONE — best AUC-PR: {best_auc:.4f}")
    print(f"  Saved to {out_dir}/metrics.json")
    return final

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None,
        choices=["A1_no_cross_attn","A2_no_cca",
                 "A3_geo_only","A4_aero_only"],
        help="Run single variant for parallel GPU execution")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"GPUs  : {torch.cuda.device_count()}")

    VARIANTS = [
        # name               cross_attn  cca    geo    aero
        ("A1_no_cross_attn", False,      True,  True,  True),
        ("A2_no_cca",        True,       False, True,  True),
        ("A3_geo_only",      False,      False, True,  False),
        ("A4_aero_only",     False,      False, False, True),
    ]

    if args.variant:
        VARIANTS = [v for v in VARIANTS if v[0] == args.variant]
        if not VARIANTS:
            print(f"Unknown variant: {args.variant}"); sys.exit(1)

    all_results = {}
    for name, xattn, cca, geo, aero in VARIANTS:
        kwargs = {"use_cross_attn": xattn, "use_cca": cca,
                  "use_geo": geo, "use_aero": aero}
        result = train_ablation(name, kwargs, CONFIG)
        all_results[name] = result["best_auc_pr"]

    print("\n" + "="*60)
    print("ABLATION SUMMARY (fold 0, best AUC-PR)")
    print("="*60)
    for vname, val in all_results.items():
        print(f"  {vname:<32} AUC-PR = {val:.4f}")

    summary_path = Path("experiments/ablation_studies/summary.json")
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {summary_path}")