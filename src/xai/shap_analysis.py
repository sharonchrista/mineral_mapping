"""
src/xai/shap_analysis.py
SHAP feature importance analysis for the mineral prospectivity model.

Computes:
  1. Global SHAP values per channel (bar chart — Fig 7a)
  2. SHAP beeswarm plot (Fig 7b)
  3. Spatial SHAP maps per top channel (Fig 7c-f)

Usage:
  python src/xai/shap_analysis.py
  python src/xai/shap_analysis.py --fold 0 --n_samples 500

Output:
  results/shap/shap_values.npy
  results/shap/shap_summary.json
  results/shap/figure7.png / .pdf / .eps
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finetune.model_fusion import MineralProspectivityModel, DEPTH_INTERVALS
from src.finetune.dataset import ProspectivityDataset

# =============================================================================
# CHANNEL NAMES — must match channel_manifest.json order
# =============================================================================
CHANNEL_NAMES = [
    "Lithology 25K",           # 0
    "Fault dist. 25K",         # 1
    "Fold dist. 25K",          # 2
    "Shear zone dist. 25K",    # 3
    "Dyke dist. 25K",          # 4
    "Lithology 50K",           # 5
    "Fault dist. 50K",         # 6
    "Fold dist. 50K",          # 7
    "TMI",                     # 8
    "TMI 1VD",                 # 9
    "TMI ASA",                 # 10
    "TMI TDR",                 # 11
    "Fault density (KDE)",     # 12
    "Structural complexity",   # 13
    "Contact density",         # 14
]

# Channel groups for coloring
CHANNEL_GROUPS = {
    "Geological 25K":    [0, 1, 2, 3, 4],
    "Geological 50K":    [5, 6, 7],
    "Aeromagnetic":      [8, 9, 10, 11],
    "Derived features":  [12, 13, 14],
}

GROUP_COLORS = {
    "Geological 25K":   "#E07B4A",
    "Geological 50K":   "#F0C060",
    "Aeromagnetic":     "#4FAEAC",
    "Derived features": "#A86BAA",
}

DEFAULT_CONFIG = {
    "tif_path":      "data/processed/stacked/multiband_input.tif",
    "label_path":    "data/labels/pu_labels.tif",
    "fold":          0,
    "ckpt_dir":      "experiments/finetune",
    "output_dir":    "results/shap",
    "n_samples":     300,     # number of patches to compute SHAP on
    "n_background":  50,      # background samples for SHAP kernel
    "batch_size":    8,
    "img_size":      256,
    "patch_size":    16,
    "encoder_embed_dim": 768,
    "encoder_depth":     12,
    "encoder_num_heads": 12,
    "cca_dim":       256,
    "fusion_dim":    512,
    "mc_dropout":    0.2,
}


# =============================================================================
# SHAP WRAPPER — channel-level attribution
# =============================================================================

class ChannelAggregator(nn.Module):
    """
    Wraps the prospectivity model to accept channel-level scalar inputs
    (global average per channel) instead of full spatial patches.
    This allows SHAP KernelExplainer to attribute importance to each
    of the 15 input channels.

    Input:  (N, 15)  — mean value per channel per sample
    Output: (N, 1)   — prospectivity logit for 0-500m head
    """

    def __init__(self, model, img_size=256, device="cpu"):
        super().__init__()
        self.model    = model
        self.img_size = img_size
        self.device   = device

    def forward_numpy(self, x_np):
        """
        x_np: (N, 15) numpy array of channel means
        Returns: (N,) numpy array of prospectivity scores
        """
        self.model.eval()
        results = []
        batch_size = 4
        for i in range(0, len(x_np), batch_size):
            batch_ch = x_np[i:i+batch_size]
            # Expand each channel scalar to a full spatial patch
            # Shape: (B, 15, H, W) with each channel filled with its mean
            B = len(batch_ch)
            patches = np.zeros((B, 15, self.img_size, self.img_size),
                                dtype=np.float32)
            for b in range(B):
                for c in range(15):
                    patches[b, c, :, :] = batch_ch[b, c]

            with torch.no_grad():
                t = torch.from_numpy(patches).to(self.device)
                out = self.model(t)
                logits = out[f"logits_{DEPTH_INTERVALS[0]}"]
                if logits.dim() > 1:
                    logits = logits.squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
            results.append(probs)
        return np.concatenate(results)


def extract_channel_means(patches):
    """
    patches: (N, 15, H, W) tensor or array
    Returns: (N, 15) numpy array of per-channel spatial means
    """
    if isinstance(patches, torch.Tensor):
        patches = patches.numpy()
    return patches.mean(axis=(2, 3))   # (N, 15)


# =============================================================================
# SPATIAL SHAP — gradient-based per-pixel attribution
# =============================================================================

def compute_gradient_shap(model, patches, device, n_stochastic=20):
    """
    Compute gradient x input attributions per channel using
    integrated gradients approximation.
    Returns: (N, 15, H, W) attribution maps, (N, 15) channel importance
    """
    model.eval()
    all_attrs   = []
    all_ch_imp  = []

    for i in range(len(patches)):
        patch = patches[i:i+1].to(device).requires_grad_(False)

        # Baseline: zero input
        baseline = torch.zeros_like(patch)
        attrs    = torch.zeros_like(patch)

        # Integrated gradients: average gradient over interpolated inputs
        steps = 20
        for alpha in np.linspace(0, 1, steps):
            inp = (baseline + alpha * (patch - baseline)).detach()
            inp.requires_grad_(True)
            out    = model(inp)
            logit  = out[f"logits_{DEPTH_INTERVALS[0]}"]
            if logit.dim() > 1:
                logit = logit.squeeze()
            score  = torch.sigmoid(logit).sum()
            score.backward()
            attrs += inp.grad.detach() / steps

        # Multiply by input (integrated gradients)
        attrs = attrs * (patch - baseline)
        all_attrs.append(attrs.squeeze(0).cpu().numpy())

        # Channel importance: mean |attribution| per channel
        ch_imp = np.abs(attrs.squeeze(0).cpu().numpy()).mean(axis=(1, 2))
        all_ch_imp.append(ch_imp)

    return np.stack(all_attrs), np.stack(all_ch_imp)


# =============================================================================
# MAIN
# =============================================================================

def run_shap(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model (fold {config['fold']})...")
    ckpt_path = (Path(config["ckpt_dir"]) /
                 f"fold{config['fold']}" / "checkpoint_best.pt")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = MineralProspectivityModel(
        pretrain_ckpt=None,
        img_size          = config["img_size"],
        patch_size        = config["patch_size"],
        encoder_embed_dim = config["encoder_embed_dim"],
        encoder_depth     = config["encoder_depth"],
        encoder_num_heads = config["encoder_num_heads"],
        cca_dim           = config["cca_dim"],
        fusion_dim        = config["fusion_dim"],
        mc_dropout        = config["mc_dropout"],
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded: epoch {ckpt['epoch']}  "
          f"val_auc_pr={ckpt.get('val_auc_pr',0):.4f}")

    # ── Load samples ──────────────────────────────────────────────────────
    print(f"\nLoading dataset ({config['n_samples']} samples)...")
    ds = ProspectivityDataset(
        tif_path   = config["tif_path"],
        label_path = config["label_path"],
        patch_size = config["img_size"],
        augment    = False,
        unlabeled_ratio = 2.0,
    )

    # Sample balanced: ~50% positive, ~50% unlabeled
    pos_idx = [i for i, s in enumerate(ds.samples) if s[2] == 1]
    unl_idx = [i for i, s in enumerate(ds.samples) if s[2] == 0]

    rng     = np.random.default_rng(42)
    n_half  = config["n_samples"] // 2
    sel_pos = rng.choice(pos_idx, min(n_half, len(pos_idx)), replace=False)
    sel_unl = rng.choice(unl_idx, min(n_half, len(unl_idx)), replace=False)
    sel_idx = np.concatenate([sel_pos, sel_unl])
    rng.shuffle(sel_idx)

    subset = Subset(ds, sel_idx.tolist())
    loader = DataLoader(subset, batch_size=config["batch_size"],
                        shuffle=False, num_workers=4)

    # Collect patches and labels
    all_patches = []
    all_labels  = []
    print("  Collecting patches...")
    for batch in loader:
        all_patches.append(batch["patch"])
        all_labels.append(batch["label"])
    all_patches = torch.cat(all_patches, dim=0)   # (N, 15, H, W)
    all_labels  = torch.cat(all_labels,  dim=0)   # (N,)
    print(f"  Patches: {all_patches.shape}")
    print(f"  Positive: {(all_labels==1).sum().item()}  "
          f"Unlabeled: {(all_labels==0).sum().item()}")

    # ── Compute integrated gradients ──────────────────────────────────────
    print(f"\nComputing integrated gradients...")
    _, ch_importance = compute_gradient_shap(
        model, all_patches, device
    )
    # ch_importance: (N, 15)

    # Separate by label
    pos_mask = all_labels.numpy() == 1
    unl_mask = ~pos_mask

    ch_imp_pos = ch_importance[pos_mask].mean(axis=0)   # (15,)
    ch_imp_unl = ch_importance[unl_mask].mean(axis=0)
    ch_imp_all = ch_importance.mean(axis=0)

    print("\nChannel importance (mean |grad×input|):")
    for i, (name, imp) in enumerate(zip(CHANNEL_NAMES, ch_imp_all)):
        print(f"  Ch {i:2d}  {name:30s}  {imp:.6f}")

    # Save values
    np.save(output_dir / "channel_importance.npy", ch_importance)
    summary = {
        "channel_names":      CHANNEL_NAMES,
        "importance_all":     ch_imp_all.tolist(),
        "importance_positive":ch_imp_pos.tolist(),
        "importance_unlabeled":ch_imp_unl.tolist(),
        "n_samples":          len(all_patches),
        "n_positive":         int(pos_mask.sum()),
        "fold":               config["fold"],
    }
    with open(output_dir / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir / 'shap_summary.json'}")

    # ── Generate Figure 7 ─────────────────────────────────────────────────
    print("\nGenerating Figure 7...")
    generate_figure7(
        ch_imp_all, ch_imp_pos, ch_imp_unl,
        ch_importance, all_labels.numpy(),
        all_patches, output_dir, config, device, model
    )


def generate_figure7(ch_imp_all, ch_imp_pos, ch_imp_unl,
                      ch_importance, labels,
                      all_patches, output_dir, config, device, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os

    fig = plt.figure(figsize=(16, 10), dpi=300, facecolor="white")
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                            left=0.06, right=0.97,
                            top=0.91, bottom=0.07)

    ax_bar   = fig.add_subplot(gs[0, :2])   # top-left wide: bar chart
    ax_bee   = fig.add_subplot(gs[0, 2])    # top-right: scatter per channel
    ax_sp    = [fig.add_subplot(gs[1, i]) for i in range(3)]  # spatial maps

    # Assign colors by group
    colors = []
    group_of = {}
    for name, idxs in CHANNEL_GROUPS.items():
        for i in idxs:
            group_of[i] = name
    for i in range(15):
        colors.append(GROUP_COLORS[group_of[i]])

    # ── Panel (a): Horizontal bar chart ───────────────────────────────────
    order = np.argsort(ch_imp_all)
    y_pos = np.arange(15)

    ax_bar.barh(y_pos, ch_imp_all[order],
                color=[colors[i] for i in order],
                edgecolor="white", linewidth=0.4,
                height=0.65, alpha=0.85, label="All samples", zorder=3)
    ax_bar.barh(y_pos, ch_imp_pos[order],
                color=[colors[i] for i in order],
                edgecolor="white", linewidth=0.4,
                height=0.65, alpha=0.45, hatch="///",
                label="Positive samples only", zorder=4)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([CHANNEL_NAMES[i] for i in order], fontsize=8)
    ax_bar.set_xlabel("Mean |Gradient × Input| attribution", fontsize=9)
    ax_bar.set_title("(a) Channel feature importance\n(Integrated Gradients)",
                      fontsize=9.5, fontweight="bold", pad=5)
    ax_bar.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5,
                color="#CCCCCC")
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.tick_params(labelsize=8)

    # Group legend
    group_handles = [
        mpatches.Patch(facecolor=c, label=g, alpha=0.85)
        for g, c in GROUP_COLORS.items()
    ]
    group_handles.append(
        mpatches.Patch(facecolor="gray", alpha=0.45,
                       hatch="///", label="Positive samples")
    )
    ax_bar.legend(handles=group_handles, fontsize=7.5,
                  loc="lower right", framealpha=0.9,
                  edgecolor="#DDDDDD")

    # ── Panel (b): Per-sample scatter (channel importance vs label) ───────
    top3 = np.argsort(ch_imp_all)[-3:][::-1]
    pos_mask = labels == 1
    unl_mask = ~pos_mask

    for ti, ch_idx in enumerate(top3):
        vals = ch_importance[:, ch_idx]
        color = colors[ch_idx]
        ax_bee.scatter(
            vals[pos_mask],
            np.random.normal(ti, 0.08, pos_mask.sum()),
            c=color, s=10, alpha=0.6, zorder=3,
            marker="o", label=f"Ch{ch_idx} pos" if ti==0 else ""
        )
        ax_bee.scatter(
            vals[unl_mask],
            np.random.normal(ti, 0.08, unl_mask.sum()),
            c=color, s=8, alpha=0.3, zorder=2,
            marker="^", label=f"Ch{ch_idx} unl" if ti==0 else ""
        )

    ax_bee.set_yticks(range(len(top3)))
    ax_bee.set_yticklabels(
        [CHANNEL_NAMES[i] for i in top3], fontsize=8
    )
    ax_bee.set_xlabel("Attribution value", fontsize=9)
    ax_bee.set_title("(b) Per-sample attribution\n(top 3 channels)",
                      fontsize=9.5, fontweight="bold", pad=5)
    ax_bee.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5,
                color="#CCCCCC")
    ax_bee.spines[["top","right"]].set_visible(False)
    ax_bee.tick_params(labelsize=8)

    from matplotlib.lines import Line2D
    ax_bee.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, label="Positive"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.5, label="Unlabeled"),
    ], fontsize=7.5, loc="lower right",
       framealpha=0.9, edgecolor="#DDDDDD")

    # ── Panels (c-e): Spatial attribution maps for top 3 channels ─────────
    # Compute spatial attributions for a few positive samples
    pos_indices = np.where(labels == 1)[0][:20]
    sample_patches = all_patches[pos_indices].to(device)

    print("  Computing spatial attributions for top channels...")
    spatial_attrs, _ = compute_gradient_shap(
        model, sample_patches, device
    )
    # spatial_attrs: (20, 15, H, W)
    # Average over samples
    mean_spatial = np.abs(spatial_attrs).mean(axis=0)   # (15, H, W)

    for pi, ch_idx in enumerate(top3):
        ax = ax_sp[pi]
        attr_map = mean_spatial[ch_idx]   # (H, W)
        input_ch = all_patches[pos_indices[0], ch_idx].numpy()  # (H,W)

        # Show input channel as background
        ax.imshow(input_ch, cmap="gray", alpha=0.5,
                  interpolation="nearest")
        # Overlay attribution
        im = ax.imshow(attr_map, cmap="hot", alpha=0.7,
                       interpolation="bilinear")

        ax.set_title(f"({'cde'[pi]}) Attribution: {CHANNEL_NAMES[ch_idx]}",
                      fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xlabel("Pixel (W)", fontsize=7.5)
        if pi == 0:
            ax.set_ylabel("Pixel (H)", fontsize=7.5)
        ax.tick_params(labelsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04,
                             pad=0.02, shrink=0.85)
        cbar.set_label("|Grad × Input|", fontsize=7, labelpad=2)
        cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Figure 7 — Explainability Analysis: Channel Feature Importance\n"
        "Method: Integrated Gradients  |  "
        f"n={len(labels)} samples (fold {config['fold']})  |  "
        "Depth: 0–500 m",
        fontsize=10, fontweight="bold", y=0.97
    )

    import os
    out_dir = Path(output_dir)
    for fmt, dpi in [("png",300),("pdf",300),("eps",300)]:
        p = out_dir / f"figure7.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  {p.name}  ({os.path.getsize(p)/1e6:.1f} MB)")
    plt.close(fig)
    print("Figure 7 complete.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int,   default=0)
    parser.add_argument("--n_samples",  type=int,   default=300)
    parser.add_argument("--n_background", type=int, default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["fold"]       = args.fold
    config["n_samples"]  = args.n_samples
    config["batch_size"] = args.batch_size

    run_shap(config)
