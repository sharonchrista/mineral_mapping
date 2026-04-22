#!/usr/bin/env python3
"""
06_generate_labels.py
Generate Positive-Unlabeled labels with 500m buffers and 5-fold spatial CV splits.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rich.console import Console
from rich.table import Table

console = Console()

ROOT       = Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
ALIGNED    = PROCESSED / "aligned"
LABELS_DIR = ROOT / "data" / "labels"
SPLITS_DIR = ROOT / "data" / "splits"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS    = CRS.from_epsg(32643)
TARGET_RES    = 50.0
BUFFER_M      = 500.0
N_FOLDS       = 5
FOLD_BUFFER_M = 20_000
LABEL_POS     = 1
LABEL_UNL     = 0


def load_study_extent():
    with open(PROCESSED / "study_extent.json") as f: return json.load(f)


def get_profile(extent):
    transform = from_bounds(extent["minx"], extent["miny"],
                             extent["maxx"], extent["maxy"],
                             extent["width_px"], extent["height_px"])
    return {"driver": "GTiff", "dtype": "uint8", "crs": TARGET_CRS,
            "transform": transform, "width": extent["width_px"],
            "height": extent["height_px"], "count": 1, "compress": "lzw", "nodata": 255}


def load_mineral_occurrences():
    console.rule("[bold cyan]Loading Mineral Occurrences[/bold cyan]")
    gdfs = []
    for search_dir in [ALIGNED / "mineral_occurrences",
                       ALIGNED / "geological_25k", ALIGNED / "geological_50k"]:
        for path in search_dir.glob("*.gpkg"):
            name_lower = path.stem.lower()
            if search_dir.name == "mineral_occurrences" or \
               any(k in name_lower for k in ["mine","quarry","mineral","occurrence"]):
                try:
                    gdf = gpd.read_file(path)
                    gdf["source"] = path.stem
                    gdfs.append(gdf)
                    console.print(f"  [green]✓[/green] {path.name}: {len(gdf)} features")
                except Exception as e:
                    console.print(f"  [red]✗ {path.name}: {e}[/red]")
    if not gdfs:
        console.print("  [yellow]⚠ No occurrences found[/yellow]")
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=TARGET_CRS)
    combined = combined[~combined.geometry.isna() & ~combined.geometry.is_empty]
    combined["geometry"] = combined.geometry.centroid
    console.print(f"  [bold]Total: {len(combined)} occurrences[/bold]")
    return combined


def generate_positive_raster(occurrences, extent, profile):
    console.rule("[bold cyan]Generating PU Labels (500m buffer)[/bold cyan]")
    label_arr = np.zeros((extent["height_px"], extent["width_px"]), dtype="uint8")
    if occurrences.empty:
        console.print("  [yellow]⚠ No occurrences — all unlabeled[/yellow]")
        return label_arr
    buffered = occurrences.copy()
    buffered["geometry"] = occurrences.geometry.buffer(BUFFER_M)
    buffered = buffered.clip(box(extent["minx"], extent["miny"], extent["maxx"], extent["maxy"]))
    shapes = [(g, LABEL_POS) for g in buffered.geometry if g is not None and not g.is_empty]
    if shapes:
        label_arr = rasterize(shapes=shapes, out_shape=(extent["height_px"], extent["width_px"]),
                               transform=profile["transform"], fill=LABEL_UNL,
                               dtype="uint8", all_touched=True)
    n_pos = (label_arr == LABEL_POS).sum()
    total = label_arr.size
    console.print(f"  Positive : {n_pos:,} px ({100*n_pos/total:.2f}%)")
    console.print(f"  Unlabeled: {total-n_pos:,} px ({100*(total-n_pos)/total:.2f}%)")
    return label_arr


def create_spatial_splits(label_arr, extent, profile):
    console.rule("[bold cyan]Creating 5-Fold Spatial CV Splits[/bold cyan]")
    rows, cols = label_arr.shape
    x_min, x_max = extent["minx"], extent["maxx"]
    strip_width = (x_max - x_min) / N_FOLDS
    buffer_px = int(FOLD_BUFFER_M / TARGET_RES)
    splits = []
    for fold_idx in range(N_FOLDS):
        test_x_min = x_min + fold_idx * strip_width
        test_x_max = test_x_min + strip_width
        col_start = int((test_x_min - x_min) / TARGET_RES)
        col_end   = int((test_x_max - x_min) / TARGET_RES)
        test_mask  = np.zeros((rows, cols), dtype=bool)
        test_mask[:, col_start:col_end] = True
        train_mask = np.zeros((rows, cols), dtype=bool)
        train_mask[:, :max(0, col_start - buffer_px)] = True
        train_mask[:, min(cols, col_end + buffer_px):]  = True
        test_pos  = test_mask  & (label_arr == LABEL_POS)
        train_pos = train_mask & (label_arr == LABEL_POS)
        train_unl = train_mask & (label_arr == LABEL_UNL)
        for mask, name in [(test_pos.astype("uint8"),  f"fold{fold_idx}_test_mask.tif"),
                           (train_pos.astype("uint8"), f"fold{fold_idx}_train_pos_mask.tif"),
                           (train_unl.astype("uint8"), f"fold{fold_idx}_train_unl_mask.tif")]:
            p = profile.copy(); p["dtype"] = "uint8"
            with rasterio.open(SPLITS_DIR / name, "w", **p) as dst:
                dst.write(mask, 1)
        splits.append({"fold": fold_idx, "n_test_pos": int(test_pos.sum()),
                        "n_train_pos": int(train_pos.sum()), "n_train_unl": int(train_unl.sum())})
        console.print(f"  Fold {fold_idx}: test={splits[-1]['n_test_pos']} | "
                      f"train={splits[-1]['n_train_pos']} pos + {splits[-1]['n_train_unl']:,} unlabeled")
    return splits


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 06: Generate Labels & Splits ═══[/bold cyan]\n")
    extent = load_study_extent()
    profile = get_profile(extent)
    occurrences = load_mineral_occurrences()
    label_arr = generate_positive_raster(occurrences, extent, profile)
    with rasterio.open(LABELS_DIR / "pu_labels.tif", "w", **profile) as dst:
        dst.write(label_arr, 1)
    console.print(f"  [green]✓[/green] Labels saved: {LABELS_DIR / 'pu_labels.tif'}")
    if not occurrences.empty:
        occurrences.to_file(LABELS_DIR / "mineral_occurrences_unified.gpkg", driver="GPKG")
    splits = create_spatial_splits(label_arr, extent, profile)
    with open(SPLITS_DIR / "split_metadata.json", "w") as f:
        json.dump({"n_folds": N_FOLDS, "buffer_m": BUFFER_M,
                   "fold_buffer_m": FOLD_BUFFER_M, "folds": splits}, f, indent=2)
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/07_stack_channels.py[/cyan]")
