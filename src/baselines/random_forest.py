"""
src/baselines/random_forest.py
Random Forest baseline for mineral prospectivity mapping.

Uses the same 15-channel input stack but processes pixel-level features
(no spatial context) — standard ML approach for comparison.

Usage:
  python src/baselines/random_forest.py
  python src/baselines/random_forest.py --fold 0

Output:
  results/baselines/random_forest/fold{N}/
    metrics.json
    predictions.tif
    feature_importance.json
"""

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_CONFIG = {
    "tif_path":    "data/processed/stacked/multiband_input.tif",
    "label_path":  "data/labels/pu_labels.tif",
    "splits_dir":  "data/splits",
    "output_dir":  "results/baselines/random_forest",
    "n_folds":     5,
    # RF hyperparameters
    "n_estimators":  500,
    "max_depth":     None,
    "min_samples_leaf": 5,
    "n_jobs":        -1,
    "random_state":  42,
    # Sampling
    "n_pos_train":   5000,    # positive training pixels
    "n_unl_ratio":   5,       # unlabeled:positive ratio
    "n_pred_sample": 200000,  # pixels to predict for map
}

CHANNEL_NAMES = [
    "Lithology_25K", "Fault_dist_25K", "Fold_dist_25K",
    "Shear_dist_25K", "Dyke_dist_25K",
    "Lithology_50K", "Fault_dist_50K", "Fold_dist_50K",
    "TMI", "TMI_1VD", "TMI_ASA", "TMI_TDR",
    "Fault_density_KDE", "Structural_complexity", "Contact_density",
]


def load_data(tif_path, label_path):
    with rasterio.open(tif_path) as src:
        data   = src.read().astype("float32")   # (15, H, W)
        H, W   = src.height, src.width
        bounds = src.bounds
        crs    = src.crs
        tf     = src.transform
    data = np.where(data == -9999.0, np.nan, data)

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype("uint8")

    return data, labels, H, W, bounds, crs, tf


def load_fold_mask(splits_dir, fold, mask_type):
    p = Path(splits_dir) / f"fold{fold}_{mask_type}.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1).astype("uint8")


def sample_pixels(data, labels, fold_mask, n_pos, n_unl_ratio, rng):
    """Sample pixel coordinates for training."""
    H, W = labels.shape
    valid = ~np.isnan(data).any(axis=0)   # no nodata

    pos_mask = (labels == 1) & valid
    unl_mask = (labels == 0) & valid
    if fold_mask is not None:
        pos_mask &= (fold_mask > 0)
        unl_mask &= (fold_mask > 0)

    pos_r, pos_c = np.where(pos_mask)
    unl_r, unl_c = np.where(unl_mask)

    n_pos_actual = min(n_pos, len(pos_r))
    n_unl_actual = min(n_pos_actual * n_unl_ratio, len(unl_r))

    pos_idx = rng.choice(len(pos_r), n_pos_actual, replace=False)
    unl_idx = rng.choice(len(unl_r), n_unl_actual, replace=False)

    rows = np.concatenate([pos_r[pos_idx], unl_r[unl_idx]])
    cols = np.concatenate([pos_c[pos_idx], unl_c[unl_idx]])
    lbls = np.concatenate([
        np.ones(n_pos_actual),
        np.zeros(n_unl_actual)
    ])
    # PU weights
    wts  = np.concatenate([
        np.ones(n_pos_actual),
        np.full(n_unl_actual, 0.1)
    ])

    # Extract features
    X = data[:, rows, cols].T   # (N, 15)
    return X, lbls, wts, rows, cols


def compute_metrics(y_true, y_score, y_pred):
    m = {}
    try:
        m["auc_pr"]  = float(average_precision_score(y_true, y_score))
        m["auc_roc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        m["auc_pr"] = m["auc_roc"] = 0.0
    # Optimal threshold F1 (fixed 0.5 threshold fails under PU weighting)
    best_f1, best_mcc, best_thresh = 0.0, 0.0, 0.5
    for thresh in np.linspace(0.01, 0.99, 50):
        preds = (y_score >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        try:
            f1 = float(f1_score(y_true, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_mcc = float(matthews_corrcoef(y_true, preds))
        except Exception:
            continue
    y_pred = (y_score >= best_thresh).astype(int)
    try:
        m["f1"]  = best_f1
        m["mcc"] = best_mcc
    except Exception:
        m["f1"] = m["mcc"] = 0.0
    return m


def train_fold(config, fold):
    print(f"\n{'='*50}\n RF Baseline — Fold {fold}\n{'='*50}")
    out_dir = Path(config["output_dir"]) / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42 + fold)

    print("Loading data...")
    data, labels, H, W, bounds, crs, tf = load_data(
        config["tif_path"], config["label_path"]
    )

    train_mask = load_fold_mask(config["splits_dir"], fold,
                                "train_pos_mask")
    val_mask   = load_fold_mask(config["splits_dir"], fold,
                                "test_region_mask")

    # Training set — use pos mask for positives, unl mask for unlabeled
    unl_mask_path = Path(config["splits_dir"]) / f"fold{fold}_train_unl_mask.tif"
    unl_mask = None
    if unl_mask_path.exists():
        with rasterio.open(unl_mask_path) as src:
            unl_mask = src.read(1).astype("uint8")

    print("Sampling training pixels...")
    valid = ~np.isnan(data).any(axis=0)
    # Positive pixels from pos mask
    pos_use = (labels==1) & valid
    if train_mask is not None:
        pos_use &= (train_mask > 0)
    pos_r, pos_c = np.where(pos_use)
    n_pos = min(config["n_pos_train"], len(pos_r))
    pi    = rng.choice(len(pos_r), n_pos, replace=False)

    # Unlabeled pixels from unl mask
    unl_use = (labels==0) & valid
    if unl_mask is not None:
        unl_use &= (unl_mask > 0)
    unl_r, unl_c = np.where(unl_use)
    n_unl = min(n_pos * config["n_unl_ratio"], len(unl_r))
    ui    = rng.choice(len(unl_r), int(n_unl), replace=False)

    rows = np.concatenate([pos_r[pi], unl_r[ui]])
    cols = np.concatenate([pos_c[pi], unl_c[ui]])
    y_tr = np.concatenate([np.ones(n_pos), np.zeros(int(n_unl))])
    w_tr = np.concatenate([np.ones(n_pos), np.full(int(n_unl), 0.1)])
    X_tr = data[:, rows, cols].T
    print(f"  Train: {len(X_tr)} pixels  "
          f"pos={int(y_tr.sum())}  unl={int((y_tr==0).sum())}")

    # Validation set — restricted to the held-out test fold region only
    print("Sampling validation pixels (test fold region only)...")
    valid   = ~np.isnan(data).any(axis=0)
    if val_mask is not None:
        valid = valid & (val_mask > 0)
    else:
        print("  WARNING: val_mask is None, falling back to full raster")
    pos_r, pos_c = np.where((labels==1) & valid)
    unl_r, unl_c = np.where((labels==0) & valid)
    n_unl = min(len(unl_r), len(pos_r)*5)
    idx   = rng.choice(len(unl_r), n_unl, replace=False)
    vrows = np.concatenate([pos_r, unl_r[idx]])
    vcols = np.concatenate([pos_c, unl_c[idx]])
    y_val = np.concatenate([np.ones(len(pos_r)), np.zeros(n_unl)])
    X_val = data[:, vrows, vcols].T
    for c in range(X_val.shape[1]):
        X_val[:, c] = np.where(np.isnan(X_val[:,c]),
                               np.nanmedian(X_val[:,c]), X_val[:,c])
    print(f"  Val:   {len(X_val)} pixels  "
          f"pos={int(y_val.sum())}  unl={int((y_val==0).sum())}")

    # Impute NaN with column median
    col_medians = np.nanmedian(X_tr, axis=0)
    for c in range(X_tr.shape[1]):
        X_tr[:, c]  = np.where(np.isnan(X_tr[:, c]),
                                col_medians[c], X_tr[:, c])
        X_val[:, c] = np.where(np.isnan(X_val[:, c]),
                                col_medians[c], X_val[:, c])

    # Normalize
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    # impute X_val NaN before scaling
    for c in range(X_val.shape[1]):
        X_val[:, c] = np.where(np.isnan(X_val[:,c]),
                                col_medians[c], X_val[:,c])
    X_val = scaler.transform(X_val)

    # Train RF
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators    = config["n_estimators"],
        max_depth       = config["max_depth"],
        min_samples_leaf= config["min_samples_leaf"],
        n_jobs          = config["n_jobs"],
        random_state    = config["random_state"],
        class_weight    = None,
    )
    rf.fit(X_tr, y_tr, sample_weight=w_tr)

    # Evaluate
    print("Evaluating...")
    proba = rf.predict_proba(X_val)
    if proba.shape[1] == 1:
        # Only one class in training — RF predicts constant
        print("  WARNING: RF trained on single class, using constant scores")
        y_score = proba[:, 0]
    else:
        y_score = proba[:, 1]
    y_pred  = (y_score >= 0.5).astype(int)
    metrics = compute_metrics(y_val.astype(int), y_score, y_pred)
    metrics["fold"] = fold
    metrics["n_train"] = len(X_tr)
    metrics["n_val"]   = len(X_val)

    print(f"  AUC-PR : {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  F1     : {metrics['f1']:.4f}")
    print(f"  MCC    : {metrics['mcc']:.4f}")

    # Feature importance
    fi = {name: float(imp) for name, imp in
          zip(CHANNEL_NAMES, rf.feature_importances_)}
    fi_sorted = dict(sorted(fi.items(),
                             key=lambda x: x[1], reverse=True))
    print("\n  Top 5 features:")
    for i, (name, imp) in enumerate(list(fi_sorted.items())[:5]):
        print(f"    {name}: {imp:.4f}")

    # Generate prediction map (sampled subset of study area)
    print("Generating prediction map...")
    valid_mask = ~np.isnan(data).any(axis=0)
    vr, vc     = np.where(valid_mask)
    if len(vr) > config["n_pred_sample"]:
        idx = rng.choice(len(vr), config["n_pred_sample"], replace=False)
        vr, vc = vr[idx], vc[idx]

    X_pred = data[:, vr, vc].T
    for c in range(X_pred.shape[1]):
        X_pred[:, c] = np.where(np.isnan(X_pred[:, c]),
                                  col_medians[c], X_pred[:, c])
    X_pred = scaler.transform(X_pred)
    proba_map = rf.predict_proba(X_pred)
    y_pred_map = proba_map[:, 1] if proba_map.shape[1] > 1 else proba_map[:, 0]

    pred_raster = np.full((H, W), -9999.0, dtype="float32")
    pred_raster[vr, vc] = y_pred_map

    profile = {
        "driver": "GTiff", "dtype": "float32", "crs": crs,
        "transform": tf, "width": W, "height": H,
        "count": 1, "compress": "lzw", "nodata": -9999.0,
    }
    with rasterio.open(out_dir / "predictions.tif", "w", **profile) as dst:
        dst.write(pred_raster, 1)

    # Save results
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "feature_importance.json", "w") as f:
        json.dump(fi_sorted, f, indent=2)

    print(f"\nResults saved: {out_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, default=None)
    parser.add_argument("--all_folds",  action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.all_folds or args.fold is None:
        all_metrics = []
        for fold in range(config["n_folds"]):
            m = train_fold(config, fold)
            all_metrics.append(m)
        print(f"\n{'='*50}")
        print("All folds complete:")
        print(f"  Mean AUC-PR : "
              f"{np.mean([m['auc_pr'] for m in all_metrics]):.4f} "
              f"± {np.std([m['auc_pr'] for m in all_metrics]):.4f}")
        print(f"  Mean AUC-ROC: "
              f"{np.mean([m['auc_roc'] for m in all_metrics]):.4f}")
        print(f"  Mean F1     : "
              f"{np.mean([m['f1'] for m in all_metrics]):.4f}")
    else:
        train_fold(config, args.fold)
