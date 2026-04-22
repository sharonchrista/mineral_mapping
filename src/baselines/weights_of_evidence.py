"""
src/baselines/weights_of_evidence.py
Weights of Evidence baseline — fixed for normalized/sparse channels.
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, matthews_corrcoef
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CONFIG = {
    "tif_path":   "data/processed/stacked/multiband_input.tif",
    "label_path": "data/labels/pu_labels.tif",
    "splits_dir": "data/splits",
    "output_dir": "results/baselines/woe",
    "n_folds":    5,
    "n_bins":     10,
}

CHANNEL_NAMES = [
    "Lithology_25K", "Fault_dist_25K", "Fold_dist_25K",
    "Shear_dist_25K", "Dyke_dist_25K",
    "Lithology_50K", "Fault_dist_50K", "Fold_dist_50K",
    "TMI", "TMI_1VD", "TMI_ASA", "TMI_TDR",
    "Fault_density_KDE", "Structural_complexity", "Contact_density",
]


def compute_woe(data_ch, labels):
    """
    Compute best W+/W-/C for one channel across quantile thresholds.
    Uses non-zero values for threshold computation to handle sparse channels.
    """
    valid   = ~np.isnan(data_ch)
    flat_ch = data_ch[valid].astype("float32")
    flat_lb = labels[valid].astype("int32")

    N_total  = len(flat_lb)
    N_dep    = float((flat_lb == 1).sum())
    N_nondep = N_total - N_dep

    if N_dep < 10 or N_nondep < 100:
        return None, 0.0, 0.0, 0.0, 0.0

    # Use non-zero values for threshold range
    nonzero  = flat_ch[flat_ch > 0]
    if len(nonzero) < 100:
        # Channel is nearly all zero — try above/below median
        thresholds = [float(np.median(flat_ch))]
    else:
        thresholds = list(np.percentile(
            nonzero, np.linspace(10, 90, 9)
        ))
    thresholds = sorted(set(thresholds))

    best_absC = -1
    best_Wp = best_Wm = best_C = best_sC = 0.0
    best_thr = None

    for thr in thresholds:
        B      = flat_ch >= thr
        N_B    = float(B.sum())
        N_notB = N_total - N_B
        if N_B < 10 or N_notB < 10:
            continue

        eps      = 0.5
        N_DB     = max(float((B  & (flat_lb==1)).sum()), eps)
        N_DnotB  = max(float((~B & (flat_lb==1)).sum()), eps)
        N_NDB    = max(N_B    - N_DB,    eps)
        N_NDnotB = max(N_notB - N_DnotB, eps)

        pBD   = N_DB    / N_dep
        pBnD  = N_NDB   / N_nondep
        pnBD  = N_DnotB / N_dep
        pnBnD = N_NDnotB/ N_nondep

        if pBnD <= 0 or pnBnD <= 0:
            continue

        Wp  = np.log(pBD  / pBnD)
        Wm  = np.log(pnBD / pnBnD)
        C   = Wp - Wm
        var = (1/N_DB + 1/N_NDB + 1/N_DnotB + 1/N_NDnotB)
        sC  = C / (np.sqrt(var) + 1e-8)

        if abs(C) > best_absC:
            best_absC = abs(C)
            best_Wp   = Wp
            best_Wm   = Wm
            best_C    = C
            best_sC   = sC
            best_thr  = thr

    if best_thr is None:
        return None, 0.0, 0.0, 0.0, 0.0

    binary_map = np.full(data_ch.shape, np.nan, dtype="float32")
    binary_map[valid] = (flat_ch >= best_thr).astype("float32")

    return binary_map, best_Wp, best_Wm, best_C, best_sC


def train_fold(config, fold):
    print(f"\n{'='*50}\n WoE Baseline — Fold {fold}\n{'='*50}")
    out_dir = Path(config["output_dir"]) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    with rasterio.open(config["tif_path"]) as src:
        data = src.read().astype("float32")
        H, W = src.height, src.width
        crs  = src.crs
        tf   = src.transform
    data = np.where(data == -9999.0, np.nan, data)

    with rasterio.open(config["label_path"]) as src:
        labels = src.read(1).astype("uint8")

    # Load val mask for evaluation only
    val_mask = None
    vm = Path(config["splits_dir"]) / f"fold{fold}_test_mask.tif"
    if vm.exists():
        with rasterio.open(vm) as src:
            val_mask = src.read(1).astype("uint8")

    # Compute WoE on full raster (no fold mask)
    print("Computing WoE weights (full raster)...")
    woe_maps = []
    woe_info = {}

    for ch in range(data.shape[0]):
        binary_map, Wp, Wm, C, sC = compute_woe(data[ch], labels)
        name = CHANNEL_NAMES[ch]
        woe_info[name] = {
            "W_plus": float(Wp), "W_minus": float(Wm),
            "C": float(C), "studentized_C": float(sC),
        }
        sig = "SIG" if abs(sC) >= 1.5 else "   "
        print(f"  Ch {ch:2d} {name:28s}  "
              f"W+={Wp:+.3f}  W-={Wm:+.3f}  "
              f"C={C:+.3f}  sC={sC:+.3f}  {sig}")

        if binary_map is not None:
            woe_ch = np.where(
                np.isnan(binary_map), np.nan,
                np.where(binary_map == 1, Wp, Wm)
            )
            woe_maps.append(woe_ch)

    print(f"\n  Valid channels: {len(woe_maps)} / {data.shape[0]}")

    if not woe_maps:
        print("ERROR: No valid WoE maps computed")
        return {"auc_pr":0.0,"auc_roc":0.0,"f1":0.0,"fold":fold}

    # Posterior probability
    valid_px    = ~np.isnan(data[0])
    N_total     = valid_px.sum()
    N_dep       = (labels[valid_px] == 1).sum()
    prior_odds  = N_dep / max(N_total - N_dep, 1)
    prior_logit = np.log(prior_odds + 1e-8)

    woe_stack  = np.stack(woe_maps, axis=0)
    woe_sum    = np.nansum(woe_stack, axis=0)
    post_logit = prior_logit + woe_sum
    post_prob  = 1 / (1 + np.exp(-post_logit))
    post_prob  = np.where(np.isnan(woe_sum), np.nan, post_prob)

    # Evaluate on full raster (sample for speed)
    # Use all positive pixels + random sample of unlabeled
    rng       = np.random.default_rng(42 + fold)
    all_valid = ~np.isnan(post_prob)
    pos_mask  = (labels == 1) & all_valid
    unl_mask  = (labels == 0) & all_valid
    pos_r, pos_c = np.where(pos_mask)
    unl_r, unl_c = np.where(unl_mask)
    n_unl = min(len(unl_r), len(pos_r) * 10)
    idx   = rng.choice(len(unl_r), n_unl, replace=False)
    rows  = np.concatenate([pos_r, unl_r[idx]])
    cols  = np.concatenate([pos_c, unl_c[idx]])
    y_true  = labels[rows, cols].astype(int)
    y_score = post_prob[rows, cols]
    y_pred  = (y_score >= 0.5).astype(int)



    metrics = {"fold": fold}
    try:
        metrics["auc_pr"]  = float(average_precision_score(y_true, y_score))
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
    except Exception as e:
        print(f"  Metric error: {e}")
        metrics["auc_pr"] = metrics["auc_roc"] = 0.0
    try:
        metrics["f1"]  = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["f1"] = metrics["mcc"] = 0.0

    print(f"\n  AUC-PR : {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  F1     : {metrics['f1']:.4f}")

    # Save prediction raster
    pred_out = np.where(np.isnan(post_prob),
                         -9999.0, post_prob).astype("float32")
    profile = {
        "driver":"GTiff","dtype":"float32","crs":crs,
        "transform":tf,"width":W,"height":H,
        "count":1,"compress":"lzw","nodata":-9999.0,
    }
    with rasterio.open(out_dir/"predictions.tif","w",**profile) as dst:
        dst.write(pred_out, 1)

    with open(out_dir/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir/"woe_weights.json","w") as f:
        json.dump(woe_info, f, indent=2)

    print(f"  Saved: {out_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",      type=int, default=None)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.all_folds or args.fold is None:
        all_metrics = []
        for fold in range(config["n_folds"]):
            m = train_fold(config, fold)
            all_metrics.append(m)
        print(f"\n{'='*50}")
        print("WoE All folds:")
        import numpy as np
        print(f"  Mean AUC-PR : "
              f"{np.mean([m['auc_pr'] for m in all_metrics]):.4f} "
              f"± {np.std([m['auc_pr'] for m in all_metrics]):.4f}")
        print(f"  Mean AUC-ROC: "
              f"{np.mean([m['auc_roc'] for m in all_metrics]):.4f}")
        print(f"  Mean F1     : "
              f"{np.mean([m['f1'] for m in all_metrics]):.4f}")
    else:
        train_fold(config, args.fold)
