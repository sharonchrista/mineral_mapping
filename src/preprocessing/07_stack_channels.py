#!/usr/bin/env python3
"""
07_stack_channels.py
Stack all 15 processed layers into a single multi-band GeoTIFF for model input.
"""

import json
import numpy as np
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rich.console import Console
from rich.table import Table

console = Console()

ROOT       = Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
RASTERIZED = PROCESSED / "rasterized"
AEROMAG    = PROCESSED / "aeromagnetic" / "normalized"
FEATURES   = PROCESSED / "features"
STACKED    = PROCESSED / "stacked"
STACKED.mkdir(parents=True, exist_ok=True)

TARGET_CRS = CRS.from_epsg(32643)
TARGET_RES = 50.0
N_CHANNELS = 15


def load_study_extent():
    with open(PROCESSED / "study_extent.json") as f: return json.load(f)


def find_best_match(search_dir, keywords, exclude=[]):
    candidates = list(search_dir.glob("*.tif"))
    for kw in keywords:
        matches = [f for f in candidates if kw.lower() in f.stem.lower()
                   and not any(ex.lower() in f.stem.lower() for ex in exclude)]
        if matches: return matches[0]
    return None


def resample_to_reference(src_path, ref_h, ref_w, ref_transform):
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None: arr[arr == nodata] = np.nan
        if src.crs == TARGET_CRS and arr.shape == (ref_h, ref_w):
            return arr
        out = np.full((ref_h, ref_w), np.nan, dtype="float32")
        reproject(source=arr, destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_transform, dst_crs=TARGET_CRS,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        return out


def normalize_channel(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0: return arr
    p2, p98 = np.percentile(valid, [2, 98])
    if p98 - p2 < 1e-8: return np.zeros_like(arr)
    return np.clip((arr - p2) / (p98 - p2), 0, 1).astype("float32")


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 07: Stack All Channels ═══[/bold cyan]\n")
    extent = load_study_extent()
    H, W = extent["height_px"], extent["width_px"]
    transform = from_bounds(extent["minx"], extent["miny"],
                             extent["maxx"], extent["maxy"], W, H)

    r25 = RASTERIZED / "geo_25k"
    r50 = RASTERIZED / "geo_50k"
    f25 = FEATURES   / "geo_25k"

    channel_defs = [
        (0,  "lithology_25k",        find_best_match(r25, ["lithology_code"]),           True),
        (1,  "fault_dist_25k",       find_best_match(r25, ["fault"], ["fold","shear"]),   True),
        (2,  "fold_dist_25k",        find_best_match(r25, ["fold"]),                      True),
        (3,  "shear_dist_25k",       find_best_match(r25, ["shear"]),                     True),
        (4,  "dyke_dist_25k",        find_best_match(r25, ["dyke","dike"]),               True),
        (5,  "lithology_50k",        find_best_match(r50, ["lithology_code"]),            True),
        (6,  "fault_dist_50k",       find_best_match(r50, ["fault"], ["fold","shear"]),   True),
        (7,  "fold_dist_50k",        find_best_match(r50, ["fold"]),                      True),
        (8,  "tmi",                  find_best_match(AEROMAG, ["tmi_norm"], ["1vd","asa","tdr","regional","residual"]), False),
        (9,  "tmi_1vd",              find_best_match(AEROMAG, ["1vd_norm"]),              False),
        (10, "tmi_asa",              find_best_match(AEROMAG, ["asa_norm"]),              False),
        (11, "tmi_tdr",              find_best_match(AEROMAG, ["tdr_norm"]),              False),
        (12, "fault_density_kde",    find_best_match(f25, ["kde_density","fault"]),       True),
        (13, "structural_complexity",find_best_match(f25, ["structural_complexity"]),     True),
        (14, "contact_density",      find_best_match(f25, ["contact_density"]),           True),
    ]

    stack = np.full((N_CHANNELS, H, W), 0.0, dtype="float32")

    table = Table(title="Channel Stack", show_lines=True)
    for col in ["Ch", "Name", "Status", "NaN%"]: table.add_column(col)

    for idx, name, path, do_norm in channel_defs:
        if path is None or not path.exists():
            table.add_row(str(idx), name, "[yellow]MISSING → zeros[/yellow]", "100%")
            continue
        try:
            arr = resample_to_reference(path, H, W, transform)
            if do_norm: arr = normalize_channel(arr)
            nan_pct = 100 * np.isnan(arr).sum() / arr.size
            arr = np.where(np.isnan(arr), np.nanmean(arr) if not np.all(np.isnan(arr)) else 0.0, arr)
            stack[idx] = arr
            table.add_row(str(idx), name, "[green]OK[/green]", f"{nan_pct:.1f}%")
        except Exception as e:
            table.add_row(str(idx), name, f"[red]ERROR: {str(e)[:30]}[/red]", "")

    console.print(table)

    out_path = STACKED / "multiband_input.tif"
    profile = {"driver": "GTiff", "dtype": "float32", "crs": TARGET_CRS,
               "transform": transform, "width": W, "height": H, "count": N_CHANNELS,
               "compress": "lzw", "tiled": True, "nodata": -9999.0}

    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(N_CHANNELS):
            dst.write(stack[i], i + 1)
            dst.update_tags(i + 1, name=channel_defs[i][1])

    size_mb = out_path.stat().st_size / 1e6
    console.print(f"\n  [green]✓[/green] {out_path}  ({size_mb:.1f} MB)")
    console.print(f"  Shape: {stack.shape}  (channels × height × width)")

    manifest = {"n_channels": N_CHANNELS, "shape": list(stack.shape),
                "crs": "EPSG:32643", "resolution_m": TARGET_RES,
                "channels": [{"idx": i, "name": n} for i, n, _, _ in channel_defs]}
    with open(STACKED / "channel_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"  [green]✓[/green] Manifest: {STACKED / 'channel_manifest.json'}")

    console.print("""
[bold green]✅  All preprocessing complete![/bold green]
  data/processed/stacked/multiband_input.tif  ← 15-channel model input

[bold]Next:[/bold] [cyan]python src/pretrain/train_mim.py[/cyan]
""")
