#!/usr/bin/env python3
"""
05_feature_engineering.py
Generate derived features: fault density KDE, intersection density,
lithological contact density, structural complexity index.
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.spatial import cKDTree
import rasterio
from rich.console import Console

console = Console()

ROOT       = Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
RASTERIZED = PROCESSED / "rasterized"
FEATURES   = PROCESSED / "features"
FEAT_GEO25 = FEATURES / "geo_25k"
FEAT_GEO50 = FEATURES / "geo_50k"
for d in [FEAT_GEO25, FEAT_GEO50]: d.mkdir(parents=True, exist_ok=True)

TARGET_RES = 50.0


def load_study_extent():
    p = PROCESSED / "study_extent.json"
    with open(p) as f: return json.load(f)


def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None: arr[arr == nodata] = np.nan
    return arr, profile


def save_feature(arr, out_path, profile, desc=""):
    p = profile.copy()
    p.update({"dtype": "float32", "count": 1, "compress": "lzw", "nodata": -9999.0})
    arr_out = np.where(np.isnan(arr), -9999.0, arr).astype("float32")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr_out, 1)
    console.print(f"  [green]✓[/green] {out_path.name}  "
                  f"[min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}]")


def fault_density_kde(binary, bandwidth_m=2000.0):
    sigma_px = bandwidth_m / TARGET_RES
    valid = np.where(np.isnan(binary), 0.0, binary)
    density = gaussian_filter(valid, sigma=sigma_px)
    d_max = density.max()
    if d_max > 0: density /= d_max
    return density.astype("float32")


def fault_intersection_density(dist_rasters, threshold_m=200.0):
    masks = [(d <= threshold_m) & (~np.isnan(d)) for d in dist_rasters]
    if len(masks) < 2: return np.zeros(dist_rasters[0].shape, dtype="float32")
    count = np.stack(masks, axis=0).sum(axis=0).astype("float32")
    density = gaussian_filter(count, sigma=5.0)
    d_max = density.max()
    if d_max > 0: density /= d_max
    return density.astype("float32")


def lithological_contact_density(litho, bandwidth_m=1000.0):
    valid = np.where(np.isnan(litho), 0, litho).astype("int32")
    contacts = np.zeros_like(valid, dtype="float32")
    contacts[:-1,:] += (valid[:-1,:] != valid[1:,:]).astype("float32")
    contacts[:,:-1] += (valid[:,:-1] != valid[:,1:]).astype("float32")
    contacts = np.clip(contacts, 0, 1)
    density = gaussian_filter(contacts, sigma=bandwidth_m/TARGET_RES)
    d_max = density.max()
    if d_max > 0: density /= d_max
    return density.astype("float32")


def structural_complexity_index(feature_stack, window_m=5000.0):
    window_px = int(window_m / TARGET_RES) | 1
    complexity = np.zeros(feature_stack[0].shape, dtype="float32")
    for feat in feature_stack:
        valid = np.where(np.isnan(feat), 0.0, feat)
        lm = uniform_filter(valid, size=window_px)
        lsm = uniform_filter(valid**2, size=window_px)
        complexity += np.maximum(lsm - lm**2, 0)
    c_max = complexity.max()
    if c_max > 0: complexity /= c_max
    return complexity.astype("float32")


def distance_to_intersection_points(fault_dist, shear_dist, prox_m=150.0):
    is_fault = (fault_dist <= prox_m) & (~np.isnan(fault_dist))
    is_shear = (shear_dist <= prox_m) & (~np.isnan(shear_dist))
    mask = is_fault & is_shear
    if not mask.any():
        return np.zeros_like(fault_dist)
    ir, ic = np.where(mask)
    all_r, all_c = np.mgrid[0:fault_dist.shape[0], 0:fault_dist.shape[1]]
    all_coords = np.column_stack([all_r.ravel(), all_c.ravel()])
    tree = cKDTree(np.column_stack([ir, ic]))
    dists_px, _ = tree.query(all_coords, k=1)
    dists_m = (dists_px * TARGET_RES).reshape(fault_dist.shape).astype("float32")
    nan_mask = np.isnan(fault_dist) | np.isnan(shear_dist)
    dists_m[nan_mask] = np.nan
    return dists_m


def process_scale(raster_dir, feat_dir, scale):
    console.rule(f"[bold cyan]Feature Engineering: {scale}[/bold cyan]")
    tifs = {p.stem: p for p in raster_dir.glob("*.tif")}
    if not tifs:
        console.print(f"  [yellow]⚠ No rasters in {raster_dir}[/yellow]"); return

    ref_path = next(iter(tifs.values()))
    _, ref_profile = load_raster(ref_path)

    dist_rasters = {}
    fault_dist = shear_dist = None
    for stem, path in tifs.items():
        if "distance_edt" in stem:
            arr, _ = load_raster(path)
            dist_rasters[stem] = arr
            if "fault" in stem.lower(): fault_dist = arr
            if "shear" in stem.lower(): shear_dist = arr

    for stem, dist_arr in dist_rasters.items():
        binary = (dist_arr <= TARGET_RES).astype("float32")
        save_feature(fault_density_kde(binary, 2000.0),
                     feat_dir / f"{stem}_kde_density.tif", ref_profile)

    if len(dist_rasters) >= 2:
        save_feature(fault_intersection_density(list(dist_rasters.values()), 200.0),
                     feat_dir / "fault_intersection_density.tif", ref_profile)

    for stem, path in tifs.items():
        if "lithology_code" in stem:
            arr, _ = load_raster(path)
            save_feature(lithological_contact_density(arr, 1000.0),
                         feat_dir / f"{stem}_contact_density.tif", ref_profile)

    if dist_rasters:
        save_feature(structural_complexity_index(list(dist_rasters.values()), 5000.0),
                     feat_dir / "structural_complexity_index.tif", ref_profile)

    if fault_dist is not None and shear_dist is not None:
        save_feature(distance_to_intersection_points(fault_dist, shear_dist, 150.0),
                     feat_dir / "dist_to_fault_shear_intersection.tif", ref_profile)

    console.print(f"\n  [bold]{scale}[/bold]: {len(list(feat_dir.glob('*.tif')))} features generated")


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 05: Feature Engineering ═══[/bold cyan]\n")
    process_scale(RASTERIZED / "geo_25k", FEAT_GEO25, "1:25K")
    process_scale(RASTERIZED / "geo_50k", FEAT_GEO50, "1:50K")
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/06_generate_labels.py[/cyan]")
