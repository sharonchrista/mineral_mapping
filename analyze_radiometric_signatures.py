"""
analyze_radiometric_signatures.py

Radiometric (K/U/Th) signature analysis across deposit types in the Dharwar Craton.

Purpose (Reviewer 1, Comment 3)
--------------------------------
Reviewer 1 questioned whether gold, iron, chromite, and polymetallic deposits
in the Dharwar Craton share a common metallogenic system that justifies using
a single set of evidential layers for all deposit types.

This script extracts K, U, and Th radiometric signatures at each mineral
occurrence location and tests whether the four deposit types cluster in similar
radiometric domains — consistent with a shared Archean greenstone-hosted
tectono-magmatic framework.

Outputs
-------
1. figures/radiometric_boxplots.png
     K / U / Th distributions by deposit type (box + strip plot)
2. figures/radiometric_ternary.png
     K-U-Th ternary diagram coloured by deposit type
3. figures/radiometric_map.png
     Spatial map of radiometric channels + deposit locations
4. figures/radiometric_pca.png
     PCA of K/U/Th space showing deposit type clustering
5. results/radiometric_statistics.csv
     Per-deposit-type mean, median, std for K, U, Th
6. results/radiometric_kruskal.txt
     Kruskal-Wallis test results (are distributions different?)
7. results/radiometric_summary.txt
     Plain-language summary for use in rebuttal letter

Usage
-----
Run from ~/sharon/mineral_mapping/ with the project venv active:

    python analyze_radiometric_signatures.py \
        --k_band    data/processed/aeromagnetic/spectrometric_K.tif \
        --u_band    data/processed/aeromagnetic/spectrometric_U.tif \
        --th_band   data/processed/aeromagnetic/spectrometric_Th.tif \
        --deposits  data/labels/mineral_occurrences_unified.gpkg \
        --outdir    results/radiometric_analysis

If your spectrometric data is a single multiband TIF, use:
    --spec_tif  data/processed/aeromagnetic/spectrometric.tif \
    --k_idx 1 --u_idx 2 --th_idx 3

If deposit type is stored under a different column name, use:
    --type_col  "DEPOSIT_TYPE"
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Deposit type colour palette ───────────────────────────────────────────────
DEPOSIT_COLORS = {
    "gold":         "#FFD700",
    "iron":         "#C0392B",
    "chromite":     "#27AE60",
    "polymetallic": "#8E44AD",
    "unknown":      "#95A5A6",
}

TARGET_CRS = CRS.from_epsg(32643)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_spectrometric_bands(args):
    """Load K, U, Th arrays and their rasterio transform/CRS."""
    if args.spec_tif:
        with rasterio.open(args.spec_tif) as src:
            k   = src.read(args.k_idx).astype("float32")
            u   = src.read(args.u_idx).astype("float32")
            th  = src.read(args.th_idx).astype("float32")
            transform = src.transform
            crs       = src.crs
            nodata    = src.nodata
    else:
        bands = {}
        for name, path in [("k", args.k_band), ("u", args.u_band),
                            ("th", args.th_band)]:
            with rasterio.open(path) as src:
                bands[name] = src.read(1).astype("float32")
                if name == "k":
                    transform = src.transform
                    crs       = src.crs
                    nodata    = src.nodata
        k, u, th = bands["k"], bands["u"], bands["th"]

    # Mask nodata
    nd = nodata if nodata is not None else -9999.0
    for arr in [k, u, th]:
        arr[arr == nd] = np.nan

    print(f"  Spectrometric raster: {k.shape}  CRS: {crs}")
    return k, u, th, transform, crs


def load_deposits(args, target_crs):
    """Load mineral occurrences GeoPackage and normalise deposit type column."""
    gdf = gpd.read_file(args.deposits)

    # Reproject to match raster CRS
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Normalise to point geometry
    gdf["geometry"] = gdf.geometry.centroid

    # Normalise deposit type column
    type_col = args.type_col
    if type_col not in gdf.columns:
        # Try common alternatives
        for alt in ["deposit_type", "DEPOSIT_TYPE", "type", "TYPE",
                    "commodity", "COMMODITY", "mineral", "MINERAL"]:
            if alt in gdf.columns:
                type_col = alt
                print(f"  Using column '{type_col}' for deposit type")
                break
        else:
            print(f"  WARNING: type column '{args.type_col}' not found. "
                  f"Available: {list(gdf.columns)}")
            print(f"  Assigning all deposits to 'unknown'")
            gdf["_dep_type"] = "unknown"
            type_col = "_dep_type"

    gdf["dep_type"] = gdf[type_col].str.lower().str.strip()

    # Map to canonical categories
    def categorise(t):
        if pd.isna(t):           return "unknown"
        if "gold" in t:          return "gold"
        if "iron" in t or "fe" in t: return "iron"
        if "chrom" in t:         return "chromite"
        if any(x in t for x in ["poly", "copper", "cu", "zinc", "zn",
                                  "lead", "pb", "silver", "ag"]):
            return "polymetallic"
        return "unknown"

    gdf["dep_type"] = gdf["dep_type"].apply(categorise)

    counts = gdf["dep_type"].value_counts()
    print(f"  Deposit counts: {dict(counts)}")
    return gdf


def sample_raster_at_points(arr, transform, points_gdf, buffer_px=2):
    """
    Sample raster values at point locations.
    Uses a buffer_px window and takes the median to reduce noise.
    Returns array of sampled values (NaN where outside raster or all-nodata).
    """
    from rasterio.transform import rowcol
    H, W = arr.shape
    values = []
    for geom in points_gdf.geometry:
        x, y = geom.x, geom.y
        row, col = rowcol(transform, x, y)
        r0 = max(0, row - buffer_px)
        r1 = min(H, row + buffer_px + 1)
        c0 = max(0, col - buffer_px)
        c1 = min(W, col + buffer_px + 1)
        if r0 >= H or c0 >= W or r1 <= 0 or c1 <= 0:
            values.append(np.nan)
            continue
        window = arr[r0:r1, c0:c1]
        valid  = window[~np.isnan(window)]
        values.append(float(np.median(valid)) if len(valid) > 0 else np.nan)
    return np.array(values)


# ── Figure 1: Boxplots ────────────────────────────────────────────────────────
def plot_boxplots(df, outdir):
    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic", "unknown"]
                 if d in df["dep_type"].values]
    colors    = [DEPOSIT_COLORS[d] for d in dep_types]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Radiometric Signatures by Deposit Type — Dharwar Craton",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, channel, label, unit in zip(
        axes,
        ["K_pct", "U_ppm", "Th_ppm"],
        ["Potassium (K)", "Uranium (U)", "Thorium (Th)"],
        ["%", "ppm", "ppm"]
    ):
        data_by_type = [df.loc[df["dep_type"] == d, channel].dropna().values
                        for d in dep_types]

        bp = ax.boxplot(
            data_by_type, patch_artist=True, notch=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay individual points
        for i, (data, color) in enumerate(zip(data_by_type, colors), 1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data,
                       color=color, alpha=0.5, s=15, zorder=3)

        ax.set_xticks(range(1, len(dep_types) + 1))
        ax.set_xticklabels([d.capitalize() for d in dep_types],
                            rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"{label} ({unit})", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = outdir / "radiometric_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: Ternary diagram ─────────────────────────────────────────────────
def plot_ternary(df, outdir):
    """K-U-Th ternary plot coloured by deposit type."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw ternary frame
    triangle = plt.Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                             fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(triangle)

    # Axis labels
    offset = 0.04
    ax.text(0 - offset,       0 - offset,       "K (%)",   ha="center", fontsize=11)
    ax.text(1 + offset,       0 - offset,       "U (ppm)", ha="center", fontsize=11)
    ax.text(0.5,  np.sqrt(3)/2 + offset*1.5, "Th (ppm)", ha="center", fontsize=11)

    def to_ternary(k, u, th):
        """Normalise K/U/Th to ternary coordinates."""
        total = k + u + th
        total = np.where(total == 0, np.nan, total)
        a = k  / total   # K  → left vertex
        b = u  / total   # U  → right vertex
        c = th / total   # Th → top vertex
        x = b + c / 2
        y = c * np.sqrt(3) / 2
        return x, y

    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic"]
                 if d in df["dep_type"].values]

    for dep in dep_types:
        sub = df[df["dep_type"] == dep].dropna(
            subset=["K_pct", "U_ppm", "Th_ppm"])
        if len(sub) == 0:
            continue
        # Normalise channels to 0-1 for ternary
        k_n  = (sub["K_pct"]  - df["K_pct"].min())  / (df["K_pct"].max()  - df["K_pct"].min()  + 1e-9)
        u_n  = (sub["U_ppm"]  - df["U_ppm"].min())  / (df["U_ppm"].max()  - df["U_ppm"].min()  + 1e-9)
        th_n = (sub["Th_ppm"] - df["Th_ppm"].min()) / (df["Th_ppm"].max() - df["Th_ppm"].min() + 1e-9)
        x, y = to_ternary(k_n.values, u_n.values, th_n.values)
        ax.scatter(x, y, c=DEPOSIT_COLORS[dep], s=40, alpha=0.7,
                   label=dep.capitalize(), edgecolors="white", linewidths=0.3)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.set_title("K–U–Th Ternary Diagram by Deposit Type\nDharwar Craton",
                  fontsize=12, fontweight="bold")

    out = outdir / "radiometric_ternary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: PCA plot ────────────────────────────────────────────────────────
def plot_pca(df, outdir):
    sub = df.dropna(subset=["K_pct", "U_ppm", "Th_ppm"])
    if len(sub) < 10:
        print("  Skipping PCA — insufficient data points")
        return

    X      = sub[["K_pct", "U_ppm", "Th_ppm"]].values
    X_std  = StandardScaler().fit_transform(X)
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_std)

    fig, ax = plt.subplots(figsize=(8, 6))

    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic"]
                 if d in sub["dep_type"].values]

    for dep in dep_types:
        mask = sub["dep_type"].values == dep
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=DEPOSIT_COLORS[dep], s=45, alpha=0.75,
                   label=dep.capitalize(),
                   edgecolors="white", linewidths=0.4)

    # Loadings
    loadings = pca.components_.T
    feature_names = ["K (%)", "U (ppm)", "Th (ppm)"]
    scale = np.abs(coords).max() * 0.6
    for i, fname in enumerate(feature_names):
        ax.annotate("", xy=(loadings[i, 0]*scale, loadings[i, 1]*scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(loadings[i, 0]*scale*1.15, loadings[i, 1]*scale*1.15,
                fname, fontsize=9, ha="center")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA of K–U–Th Radiometric Space by Deposit Type\nDharwar Craton",
                  fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = outdir / "radiometric_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: Spatial map ─────────────────────────────────────────────────────
def plot_spatial_map(k_arr, u_arr, th_arr, transform, gdf, outdir):
    """RGB composite of K/U/Th with deposit locations overlaid."""
    def norm(arr):
        a = np.nanpercentile(arr, 2)
        b = np.nanpercentile(arr, 98)
        return np.clip((arr - a) / (b - a + 1e-9), 0, 1)

    rgb = np.stack([norm(k_arr), norm(u_arr), norm(th_arr)], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    H, W = k_arr.shape
    extent = rasterio.transform.array_bounds(H, W, transform)
    # extent = (left, bottom, right, top)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, extent=[extent[0], extent[2], extent[1], extent[3]],
               origin="upper", aspect="equal")

    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic"]
                 if d in gdf["dep_type"].values]

    for dep in dep_types:
        sub = gdf[gdf["dep_type"] == dep]
        ax.scatter(sub.geometry.x, sub.geometry.y,
                   c=DEPOSIT_COLORS[dep], s=25, alpha=0.85,
                   label=dep.capitalize(), edgecolors="black",
                   linewidths=0.4, zorder=5)

    ax.set_xlabel("Easting (m, UTM Zone 43N)", fontsize=10)
    ax.set_ylabel("Northing (m, UTM Zone 43N)", fontsize=10)
    ax.set_title(
        "Aerogeophysical Radiometric RGB Composite (R=K, G=U, B=Th)\n"
        "with Mineral Occurrences — Dharwar Craton",
        fontsize=11, fontweight="bold"
    )

    # Legend — deposits
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9,
               title="Deposit type", title_fontsize=9)

    # Inset: channel legend
    channel_patches = [
        mpatches.Patch(color="#FF4444", label="Red   = K (%)"),
        mpatches.Patch(color="#44FF44", label="Green = U (ppm)"),
        mpatches.Patch(color="#4444FF", label="Blue  = Th (ppm)"),
    ]
    legend2 = ax.legend(handles=channel_patches, loc="upper left",
                         framealpha=0.9, fontsize=8, title="RGB channels",
                         title_fontsize=8)
    ax.add_artist(legend2)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9,
               title="Deposit type", title_fontsize=9)

    out = outdir / "radiometric_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Statistics & Kruskal-Wallis ───────────────────────────────────────────────
def compute_statistics(df, outdir):
    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic"]
                 if d in df["dep_type"].values]

    # Per-type descriptive statistics
    rows = []
    for dep in dep_types:
        sub = df[df["dep_type"] == dep]
        for ch, unit in [("K_pct", "%"), ("U_ppm", "ppm"), ("Th_ppm", "ppm")]:
            vals = sub[ch].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "deposit_type": dep,
                "channel":      ch,
                "unit":         unit,
                "n":            len(vals),
                "mean":         np.mean(vals),
                "median":       np.median(vals),
                "std":          np.std(vals),
                "q25":          np.percentile(vals, 25),
                "q75":          np.percentile(vals, 75),
            })

    stats_df = pd.DataFrame(rows)
    stats_path = outdir / "radiometric_statistics.csv"
    stats_df.to_csv(stats_path, index=False, float_format="%.4f")
    print(f"  Saved: {stats_path}")

    # Kruskal-Wallis test (non-parametric ANOVA)
    kw_lines = ["Kruskal-Wallis Test Results",
                "=" * 50,
                "H0: Distributions are identical across deposit types",
                ""]

    for ch, label in [("K_pct",  "Potassium K (%)"),
                       ("U_ppm",  "Uranium U (ppm)"),
                       ("Th_ppm", "Thorium Th (ppm)")]:
        groups = [df.loc[df["dep_type"] == d, ch].dropna().values
                  for d in dep_types if len(df.loc[df["dep_type"] == d, ch].dropna()) > 2]
        if len(groups) < 2:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        sig = "*** SIGNIFICANT" if p_val < 0.05 else "not significant"
        kw_lines.append(f"{label}")
        kw_lines.append(f"  H = {h_stat:.3f},  p = {p_val:.4f}  ({sig})")
        kw_lines.append("")

    kw_path = outdir / "radiometric_kruskal.txt"
    kw_path.write_text("\n".join(kw_lines))
    print(f"  Saved: {kw_path}")

    return stats_df, kw_lines


# ── Plain-language rebuttal summary ──────────────────────────────────────────
def write_rebuttal_summary(df, stats_df, kw_lines, outdir):
    dep_types = [d for d in ["gold", "iron", "chromite", "polymetallic"]
                 if d in df["dep_type"].values]

    lines = [
        "RADIOMETRIC ANALYSIS SUMMARY FOR REBUTTAL",
        "=" * 60,
        "",
        "USE IN RESPONSE TO: Reviewer 1, Comment 3",
        "(Justification for mixed deposit types sharing a single",
        " evidence layer set)",
        "",
        "KEY FINDINGS",
        "-" * 40,
    ]

    # Summarise per-channel overlap
    for ch, label, unit in [("K_pct",  "K",  "%"),
                              ("U_ppm",  "U",  "ppm"),
                              ("Th_ppm", "Th", "ppm")]:
        lines.append(f"\n{label} ({unit}):")
        for dep in dep_types:
            sub = stats_df[(stats_df["deposit_type"] == dep) &
                           (stats_df["channel"] == ch)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            lines.append(f"  {dep:15s}: mean={r['mean']:.3f}, "
                         f"median={r['median']:.3f}, std={r['std']:.3f}, n={int(r['n'])}")

    lines += ["", "STATISTICAL TESTS", "-" * 40]
    lines += kw_lines[3:]   # skip header

    lines += [
        "",
        "INTERPRETATION FOR REBUTTAL",
        "-" * 40,
        "",
        "If K/U/Th distributions overlap substantially across deposit types",
        "(i.e. Kruskal-Wallis p > 0.05 for most channels), this supports",
        "the argument that all four deposit types occupy similar lithological",
        "domains in the Dharwar Craton greenstone framework, justifying a",
        "single evidence layer set.",
        "",
        "If distributions differ significantly, acknowledge this as a",
        "limitation and note that deposit-type-stratified modelling is a",
        "direction for future work.",
        "",
        "SUGGESTED REBUTTAL TEXT (fill in actual values)",
        "-" * 40,
        "",
        "To address the reviewer's concern regarding the heterogeneity of",
        "deposit types, we conducted an independent analysis of aerogeophysical",
        "radiometric signatures (K, U, Th) at all 417 mineral occurrence",
        "locations using the AIkosh aerogeophysical spectrometric dataset",
        "(Karnataka and Andhra Pradesh). The K, U and Th values extracted",
        "at gold, iron, chromite, and polymetallic occurrences show",
        "[DESCRIBE OVERLAP OR SEPARATION FROM YOUR RESULTS].",
        "A Kruskal-Wallis test [confirms/does not confirm] statistically",
        "significant differences between deposit type distributions",
        "(K: H=XX, p=XX; U: H=XX, p=XX; Th: H=XX, p=XX).",
        "These findings are consistent with all four deposit types being",
        "hosted within the Archean greenstone sequences of the Dharwar",
        "Craton, where gold mineralisation is spatially associated with",
        "BIF-hosted iron formations and ultramafic-hosted chromite deposits",
        "within the same structural corridors (Chitradurga, Hutti-Maski,",
        "and Sandur schist belts). The broadly similar radiometric domains",
        "support the use of a unified evidence layer set for regional-scale",
        "MPM, while acknowledging that deposit-type-specific modelling",
        "remains a direction for future work.",
    ]

    summary_path = outdir / "radiometric_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"  Saved: {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Radiometric signature analysis for deposit type justification"
    )
    # Spectrometric input — single multiband TIF OR separate band TIFs
    parser.add_argument("--spec_tif", default=None,
                        help="Single multiband spectrometric TIF")
    parser.add_argument("--k_idx",   type=int, default=1, help="Band index for K")
    parser.add_argument("--u_idx",   type=int, default=2, help="Band index for U")
    parser.add_argument("--th_idx",  type=int, default=3, help="Band index for Th")
    parser.add_argument("--k_band",  default=None, help="Separate K band TIF")
    parser.add_argument("--u_band",  default=None, help="Separate U band TIF")
    parser.add_argument("--th_band", default=None, help="Separate Th band TIF")
    # Deposits
    parser.add_argument("--deposits", default="data/labels/mineral_occurrences_unified.gpkg",
                        help="Mineral occurrences GeoPackage")
    parser.add_argument("--type_col", default="deposit_type",
                        help="Column name for deposit type")
    # Output
    parser.add_argument("--outdir", default="results/radiometric_analysis",
                        help="Output directory")
    parser.add_argument("--buffer_px", type=int, default=2,
                        help="Pixel buffer for sampling (default 2 = 100m at 50m res)")
    args = parser.parse_args()

    # Validate inputs
    if args.spec_tif is None and (args.k_band is None or
                                    args.u_band is None or
                                    args.th_band is None):
        parser.error(
            "Provide either --spec_tif OR all three of --k_band --u_band --th_band"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    fig_dir = outdir / "figures"

    print("\n" + "="*60)
    print("  Radiometric Signature Analysis — Dharwar Craton")
    print("="*60 + "\n")

    # 1. Load data
    print("Loading spectrometric bands...")
    k, u, th, transform, crs = load_spectrometric_bands(args)

    print("\nLoading mineral occurrences...")
    gdf = load_deposits(args, TARGET_CRS)

    # Reproject raster CRS if needed for sampling
    if crs != TARGET_CRS:
        print(f"  WARNING: Spectrometric CRS ({crs}) differs from target "
              f"({TARGET_CRS}). Reprojecting deposits to match raster CRS.")
        gdf = gdf.to_crs(crs)

    # 2. Sample radiometric values at deposit locations
    print("\nSampling radiometric values at deposit locations...")
    gdf["K_pct"]  = sample_raster_at_points(k,  transform, gdf, args.buffer_px)
    gdf["U_ppm"]  = sample_raster_at_points(u,  transform, gdf, args.buffer_px)
    gdf["Th_ppm"] = sample_raster_at_points(th, transform, gdf, args.buffer_px)

    df = gdf[["dep_type", "K_pct", "U_ppm", "Th_ppm"]].copy()
    n_valid = df.dropna().shape[0]
    print(f"  Valid samples: {n_valid}/{len(df)}")

    # 3. Figures
    print("\nGenerating figures...")
    plot_boxplots(df, fig_dir)
    plot_ternary(df, fig_dir)
    plot_pca(df, fig_dir)
    plot_spatial_map(k, u, th, transform, gdf, fig_dir)

    # 4. Statistics
    print("\nComputing statistics...")
    stats_df, kw_lines = compute_statistics(df, outdir)

    # 5. Rebuttal summary
    write_rebuttal_summary(df, stats_df, kw_lines, outdir)

    # 6. Save sampled data
    csv_path = outdir / "deposit_radiometric_values.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    print(f"\nDone. All outputs in: {outdir}/")
    print("\nNext step: check results/radiometric_analysis/radiometric_summary.txt")
    print("for the plain-language rebuttal text template.")


if __name__ == "__main__":
    main()