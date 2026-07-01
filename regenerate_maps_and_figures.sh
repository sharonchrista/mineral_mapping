#!/bin/bash
# regenerate_maps_and_figures.sh
# Regenerates all prospectivity maps from corrected checkpoints
# and generates all figures needed for the rebuttal.
#
# Run from ~/sharon/mineral_mapping/
# Usage: bash regenerate_maps_and_figures.sh
#
# Prerequisites:
#   - Corrected fine-tuned checkpoints in experiments/finetune/fold{k}/checkpoint_best.pt
#   - Mineral occurrences in data/labels/mineral_occurrences_unified.gpkg
#   - Radiometric figures already in results/radiometric_analysis/figures/

set -e
BASE=~/sharon/mineral_mapping
cd $BASE

echo "============================================================"
echo " Step 1: Regenerate prospectivity maps (all 5 folds)"
echo "============================================================"

for fold in 0 1 2 3 4; do
    echo "  Generating map for fold $fold..."
    CUDA_VISIBLE_DEVICES=0 python src/finetune/predict_map.py \
        --fold $fold \
        --ckpt_dir experiments/finetune \
        --output_dir results/prospectivity_maps_corrected \
        --mc_passes 50 \
        --batch_size 16 \
        --num_workers 4 \
        > logs/predict_fold${fold}.log 2>&1
    echo "  Fold $fold done."
done

echo ""
echo "============================================================"
echo " Step 2: Generate all figures"
echo "============================================================"

python - << 'PYEOF'
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import rasterio
import geopandas as gpd
from scipy import stats

BASE    = Path(".")
FIG_DIR = BASE / "results" / "figures_corrected"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIR = BASE / "results" / "prospectivity_maps_corrected"
OCC_PATH = BASE / "data" / "labels" / "mineral_occurrences_unified.gpkg"

# ── Colour maps ───────────────────────────────────────────────────────────────
PROSP_CMAP = LinearSegmentedColormap.from_list(
    "prosp", ["white", "#FFF3B0", "#F97316", "#9B1C1C"])
UNCERT_CMAP = "Blues"

# ── Load occurrences ──────────────────────────────────────────────────────────
try:
    occ = gpd.read_file(OCC_PATH).to_crs("EPSG:32643")
    occ["geometry"] = occ.geometry.centroid
    HAS_OCC = True
except Exception as e:
    print(f"  WARNING: Could not load occurrences: {e}")
    HAS_OCC = False

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Leakage-free pre-training protocol diagram
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Leakage-free pre-training protocol...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Leakage-Free Per-Fold MIM Pre-Training Protocol",
    fontsize=14, fontweight="bold", y=1.02
)

# Left panel: spatial fold layout
ax = axes[0]
fold_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
fold_labels = [f"Fold {k}" for k in range(5)]
fold_widths = [0.2, 0.2, 0.2, 0.2, 0.2]
x = 0
for k, (w, c, l) in enumerate(zip(fold_widths, fold_colors, fold_labels)):
    ax.barh(0, w, left=x, height=0.6, color=c, alpha=0.85,
            edgecolor="white", linewidth=2)
    ax.text(x + w/2, 0, l, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    x += w

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("Spatial extent (West → East)", fontsize=11)
ax.set_yticks([])
ax.set_title("Study Area — 5 Vertical Strip Folds\n(Dharwar Craton, 42,291 km²)",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Add 20km buffer annotation
ax.annotate("20 km\nexclusion\nbuffer",
            xy=(0.2, 0.3), xytext=(0.22, 1.1),
            fontsize=8, ha="center", color="#555",
            arrowprops=dict(arrowstyle="->", color="#555", lw=1))

# Right panel: per-fold pre-training protocol
ax = axes[1]
ax.set_xlim(0, 6)
ax.set_ylim(-0.5, 5.5)
ax.set_title("Per-Fold Leakage-Free Protocol", fontsize=11, fontweight="bold")
ax.axis("off")

for k in range(5):
    y = 4.5 - k
    # Draw fold bar
    for j in range(5):
        color = "#DDDDDD" if j == k else fold_colors[j]
        alpha = 0.3 if j == k else 0.8
        hatch = "////" if j == k else ""
        rect = plt.Rectangle((j*0.8, y-0.25), 0.75, 0.5,
                               facecolor=color, alpha=alpha,
                               edgecolor="white", hatch=hatch)
        ax.add_patch(rect)

    # Arrow and text
    ax.annotate("", xy=(4.2, y), xytext=(4.05, y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(4.3, y, f"Encoder {k}\n(pre-trained)", fontsize=8,
            va="center", color="#222")

    ax.text(-0.1, y, f"k={k}", fontsize=9, va="center",
            ha="right", fontweight="bold")

ax.text(2.0, 5.2, "Pre-training tiles", ha="center", fontsize=9, color="#333")
ax.text(2.0, 5.0, "(grey = excluded test strip)", ha="center",
        fontsize=8, color="#888")

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#DDDDDD", hatch="////", label="Excluded (test strip + buffer)"),
    mpatches.Patch(facecolor=fold_colors[0], alpha=0.8, label="Pre-training tiles"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          framealpha=0.9)

plt.tight_layout()
out = FIG_DIR / "fig_leakage_free_protocol.pdf"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Success-rate curve + P-A plot
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Success-rate curve and P-A plot...")

def compute_success_rate(prob_map, occ_gdf, transform, H, W):
    """Compute success-rate curve: % deposits captured vs % area predicted."""
    # Get prospectivity values at deposit locations
    from rasterio.transform import rowcol
    occ_scores = []
    for geom in occ_gdf.geometry:
        try:
            r, c = rowcol(transform, geom.x, geom.y)
            if 0 <= r < H and 0 <= c < W:
                v = prob_map[r, c]
                if not np.isnan(v):
                    occ_scores.append(v)
        except Exception:
            pass

    if not occ_scores:
        return None, None, None, None

    occ_scores = np.array(occ_scores)
    valid_pixels = prob_map[~np.isnan(prob_map)].flatten()
    total_area   = len(valid_pixels)
    total_occ    = len(occ_scores)

    thresholds = np.percentile(valid_pixels, np.linspace(100, 0, 200))
    pct_area, pct_occ = [], []

    for t in thresholds:
        pa = (valid_pixels >= t).sum() / total_area * 100
        po = (occ_scores  >= t).sum() / total_occ  * 100
        pct_area.append(pa)
        pct_occ.append(po)

    return np.array(pct_area), np.array(pct_occ), total_area, total_occ


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Exploration-Oriented Performance Metrics\n(Corrected Leakage-Free Results)",
    fontsize=13, fontweight="bold"
)

fold_colors_main = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
all_sr_areas, all_sr_occs = [], []

for fold in range(5):
    p = MAP_DIR / f"fold{fold}" / "prospectivity_mean_0_500m.tif"
    if not p.exists():
        print(f"  WARNING: Map not found for fold {fold}: {p}")
        continue

    with rasterio.open(p) as src:
        prob = src.read(1).astype("float32")
        prob = np.where(prob == -9999.0, np.nan, prob)
        transform = src.transform
        H, W = src.height, src.width

    if not HAS_OCC:
        continue

    pct_area, pct_occ, ta, to = compute_success_rate(
        prob, occ, transform, H, W)
    if pct_area is None:
        continue

    all_sr_areas.append(pct_area)
    all_sr_occs.append(pct_occ)

    # Success-rate curve
    axes[0].plot(pct_area, pct_occ, color=fold_colors_main[fold],
                 alpha=0.6, linewidth=1.5, label=f"Fold {fold}")
    # P-A plot (prediction rate vs area)
    axes[1].plot(pct_area, pct_occ, color=fold_colors_main[fold],
                 alpha=0.6, linewidth=1.5, label=f"Fold {fold}")

# Mean curve
if all_sr_areas:
    # Interpolate to common x axis
    common_x = np.linspace(0, 100, 200)
    interp_occs = [np.interp(common_x, a, o)
                   for a, o in zip(all_sr_areas, all_sr_occs)]
    mean_occ = np.mean(interp_occs, axis=0)
    std_occ  = np.std(interp_occs, axis=0)

    axes[0].plot(common_x, mean_occ, "k-", linewidth=2.5, label="Mean")
    axes[0].fill_between(common_x, mean_occ-std_occ, mean_occ+std_occ,
                          alpha=0.15, color="black")
    axes[1].plot(common_x, mean_occ, "k-", linewidth=2.5, label="Mean")

# Random baseline
axes[0].plot([0,100],[0,100], "k--", linewidth=1, alpha=0.5, label="Random")
axes[1].plot([0,100],[0,100], "k--", linewidth=1, alpha=0.5, label="Random")

for ax, title, xlabel in [
    (axes[0], "Success-Rate Curve",
     "Study area captured (% of total area, high→low P)"),
    (axes[1], "Prediction-Area (P-A) Plot",
     "Predicted area (%)"),
]:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Known deposits captured (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out = FIG_DIR / "fig_success_rate_pa_plot.pdf"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Uncertainty tier figure (4-tier decision zones)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Uncertainty tier figure...")

# Use fold 3 (best performing) for the tier figure
fold = 3
p_path = MAP_DIR / f"fold{fold}" / "prospectivity_mean_0_500m.tif"
u_path = MAP_DIR / f"fold{fold}" / "prospectivity_uncertainty_0_500m.tif"

if p_path.exists() and u_path.exists():
    with rasterio.open(p_path) as src:
        P = src.read(1).astype("float32")
        P = np.where(P == -9999.0, np.nan, P)
        transform = src.transform
        extent_raster = [src.bounds.left, src.bounds.right,
                         src.bounds.bottom, src.bounds.top]
    with rasterio.open(u_path) as src:
        U = src.read(1).astype("float32")
        U = np.where(U == -9999.0, np.nan, U)

    # Define tier thresholds
    P_HIGH  = 0.25
    U_LOW   = np.nanpercentile(U[P > P_HIGH], 50) if (P > P_HIGH).any() else 0.02

    tier = np.full_like(P, np.nan)
    tier[(P > P_HIGH)  & (U <= U_LOW)]  = 1   # Tier 1: priority drill
    tier[(P > P_HIGH)  & (U >  U_LOW)]  = 2   # Tier 2: survey first
    tier[(P <= 0.10)   & (U <= U_LOW)]  = 3   # Tier 3: background
    tier[(P <= 0.10)   & (U >  U_LOW)]  = 4   # Tier 4: data gap

    tier_colors = {1: "#C0392B", 2: "#E67E22", 3: "#95A5A6", 4: "#3498DB"}
    tier_labels = {
        1: f"Tier 1 — Priority drill\n(P>{P_HIGH:.2f}, low σ)",
        2: f"Tier 2 — Survey first\n(P>{P_HIGH:.2f}, high σ)",
        3: "Tier 3 — Background\n(P≤0.10, low σ)",
        4: "Tier 4 — Data gap\n(P≤0.10, high σ)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f"Exploration Decision Tiers — Fold {fold} (AUC-PR=0.9723)\n"
        "Depth interval: 0–500 m (surface)",
        fontsize=13, fontweight="bold"
    )

    # Panel a: Prospectivity map
    im = axes[0].imshow(P, cmap=PROSP_CMAP, vmin=0, vmax=0.5,
                         extent=extent_raster, aspect="auto", origin="upper")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04,
                 label="P(mineral occurrence)")
    if HAS_OCC:
        axes[0].scatter(occ.geometry.x, occ.geometry.y,
                        c="yellow", s=8, alpha=0.7, label="Occurrences",
                        edgecolors="black", linewidths=0.3, zorder=5)
        axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("(a) Prospectivity Map", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Easting (m, UTM 43N)", fontsize=9)
    axes[0].set_ylabel("Northing (m, UTM 43N)", fontsize=9)

    # Panel b: Uncertainty map
    im2 = axes[1].imshow(U, cmap=UNCERT_CMAP, vmin=0,
                          vmax=np.nanpercentile(U, 98),
                          extent=extent_raster, aspect="auto", origin="upper")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Epistemic uncertainty (σ)")
    axes[1].set_title("(b) Epistemic Uncertainty", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Easting (m, UTM 43N)", fontsize=9)

    # Panel c: Tier map
    tier_rgb = np.full((*P.shape, 4), np.nan)
    for t_id, color in tier_colors.items():
        mask = tier == t_id
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        tier_rgb[mask] = [r, g, b, 0.8]

    axes[2].imshow(tier_rgb, extent=extent_raster, aspect="auto", origin="upper")
    if HAS_OCC:
        axes[2].scatter(occ.geometry.x, occ.geometry.y,
                        c="white", s=8, alpha=0.9, zorder=5,
                        edgecolors="black", linewidths=0.3)

    legend_patches = [
        mpatches.Patch(color=tier_colors[t], label=tier_labels[t])
        for t in [1, 2, 3, 4]
    ]
    axes[2].legend(handles=legend_patches, loc="lower right",
                   fontsize=7, framealpha=0.95)
    axes[2].set_title("(c) Exploration Decision Tiers", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Easting (m, UTM 43N)", fontsize=9)

    # Tier statistics
    for t_id in [1, 2, 3, 4]:
        n = (tier == t_id).sum()
        area_km2 = n * 0.0025  # 50m pixels -> km²
        print(f"  Tier {t_id}: {n:,} pixels = {area_km2:.1f} km²")

    plt.tight_layout()
    out = FIG_DIR / "fig_uncertainty_tiers.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
else:
    print(f"  WARNING: Maps not found for fold {fold}, skipping tier figure")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Copy radiometric figures (already generated)
# ═══════════════════════════════════════════════════════════════════════════════
print("Copying radiometric figures...")
import shutil
radio_src = Path("results/radiometric_analysis/figures")
if radio_src.exists():
    for f in radio_src.glob("*.png"):
        shutil.copy(f, FIG_DIR / f.name)
        print(f"  Copied: {f.name}")
else:
    print("  WARNING: Radiometric figures not found")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Corrected ablation comparison (fold 0 vs fold 3)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Corrected ablation study...")

variants = ["full", "no_cross_attn", "no_cca", "geo_only", "aero_only", "no_pretrain"]
labels   = ["Full model", "w/o Cross-attn", "w/o CCA", "Geo only", "Aero only", "w/o Pre-train"]
fold0_results = [0.4300, 0.4512, 0.3987, 0.3229, 0.3511, 0.3889]
fold3_results = [0.9660, 0.8566, 0.9648, 0.9553, 0.2701, 0.9513]

x = np.arange(len(variants))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Ablation Study (Leakage-Free Protocol)",
    fontsize=13, fontweight="bold"
)

for ax, results, fold, auc_full in [
    (axes[0], fold0_results, 0, 0.4300),
    (axes[1], fold3_results, 3, 0.9660),
]:
    colors = ["#2ECC71" if v == auc_full else
              "#E74C3C" if v < auc_full - 0.05 else "#F39C12"
              for v in results]
    bars = ax.bar(x, results, width=0.6, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1)
    ax.axhline(auc_full, color="green", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Full model ({auc_full:.4f})")

    for bar, val in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC-PR", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Fold {fold}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out = FIG_DIR / "fig_ablation_corrected.pdf"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print("\nAll figures generated successfully!")
print(f"Output directory: {FIG_DIR}")
print("\nFigures generated:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")

PYEOF

echo ""
echo "============================================================"
echo " All done!"
echo " Maps : results/prospectivity_maps_corrected/"
echo " Figs : results/figures_corrected/"
echo "============================================================"