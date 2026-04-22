#!/usr/bin/env python3
"""
03_rasterize_geology.py
Convert aligned geological vector layers to 50m raster grids.
- Structural features (faults, folds, shear zones) → Euclidean Distance Transform
- Lithology → categorical integer codes
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box
from rich.console import Console
from rich.progress import track

console = Console()

ROOT       = Path(__file__).resolve().parents[2]
PROCESSED  = ROOT / "data" / "processed"
ALIGNED    = PROCESSED / "aligned"
RASTER_OUT = PROCESSED / "rasterized"

GEO25_ALIGNED = ALIGNED / "geological_25k"
GEO50_ALIGNED = ALIGNED / "geological_50k"
RASTER_25K    = RASTER_OUT / "geo_25k"
RASTER_50K    = RASTER_OUT / "geo_50k"

for d in [RASTER_25K, RASTER_50K]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_CRS = CRS.from_epsg(32643)
TARGET_RES = 50.0

LAYER_TYPE_MAP = {
    "fault": "distance", "fold": "distance", "shear": "distance",
    "dyke": "distance", "dike": "distance", "contact": "distance",
    "lineament": "distance", "litholog": "categorical", "litho": "categorical",
    "geology": "categorical", "rock": "categorical", "formation": "categorical",
    "mine": "binary", "quarry": "binary", "mineral": "binary", "occurrence": "binary",
}

LITHO_COL_CANDIDATES = ["LITHOLOGY","lithology","LITHO","litho","ROCK_TYPE",
                         "rock_type","FORMATION","formation","UNIT","unit"]


def load_study_extent():
    p = PROCESSED / "study_extent.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    tifs = list(ALIGNED.rglob("*.tif"))
    if tifs:
        with rasterio.open(tifs[0]) as src:
            b = src.bounds
            return {"minx": b.left, "miny": b.bottom, "maxx": b.right, "maxy": b.top,
                    "width_px": src.width, "height_px": src.height, "resolution_m": TARGET_RES}
    raise FileNotFoundError("Run 02_reproject_align.py first.")


def get_profile(extent):
    transform = from_bounds(extent["minx"], extent["miny"],
                             extent["maxx"], extent["maxy"],
                             extent["width_px"], extent["height_px"])
    return {"driver": "GTiff", "dtype": "float32", "crs": TARGET_CRS,
            "transform": transform, "width": extent["width_px"],
            "height": extent["height_px"], "count": 1,
            "compress": "lzw", "tiled": True, "nodata": -9999.0}


def detect_layer_type(name):
    name_lower = name.lower()
    for kw, lt in LAYER_TYPE_MAP.items():
        if kw in name_lower: return lt
    return "distance"


def detect_litho_column(gdf):
    for col in LITHO_COL_CANDIDATES:
        if col in gdf.columns: return col
    for col in gdf.columns:
        if any(k in col.lower() for k in ["lith","rock","form","unit","geol"]): return col
    return None


def save_raster(arr, out_path, profile):
    p = profile.copy(); p["dtype"] = "float32"
    arr_out = np.where(np.isnan(arr), -9999.0, arr).astype("float32")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(arr_out, 1)


def rasterize_distance(gdf, profile, out_path):
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not shapes: return
    binary = rasterize(shapes=shapes, out_shape=(profile["height"], profile["width"]),
                       transform=profile["transform"], fill=0, dtype="uint8", all_touched=True)
    dist_m = (distance_transform_edt(binary == 0) * TARGET_RES).astype("float32")
    save_raster(dist_m, out_path, profile)


def rasterize_binary(gdf, profile, out_path):
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not shapes: return
    arr = rasterize(shapes=shapes, out_shape=(profile["height"], profile["width"]),
                    transform=profile["transform"], fill=0, dtype="float32", all_touched=True)
    save_raster(arr, out_path, profile)


def rasterize_categorical(gdf, profile, out_path, cat_map_path):
    litho_col = detect_litho_column(gdf)
    if litho_col is None:
        rasterize_binary(gdf, profile, out_path); return
    categories = sorted(gdf[litho_col].dropna().unique())
    cat_map = {cat: idx + 1 for idx, cat in enumerate(categories)}
    with open(cat_map_path, "w") as f: json.dump(cat_map, f, indent=2)
    arr = np.zeros((profile["height"], profile["width"]), dtype="float32")
    for cat, code in cat_map.items():
        subset = gdf[gdf[litho_col] == cat]
        if subset.empty: continue
        shapes = [(g, float(code)) for g in subset.geometry if g is not None and not g.is_empty]
        if shapes:
            burned = rasterize(shapes=shapes, out_shape=(profile["height"], profile["width"]),
                               transform=profile["transform"], fill=0, dtype="float32", all_touched=True)
            arr = np.where(burned > 0, burned, arr)
    save_raster(arr, out_path, profile)
    console.print(f"    {len(categories)} lithology units")


def process_scale(aligned_dir, out_dir, scale, extent):
    console.rule(f"[bold cyan]Rasterizing {scale}[/bold cyan]")
    profile = get_profile(extent)
    vectors = list(aligned_dir.glob("*.gpkg")) + list(aligned_dir.glob("*.shp"))
    if not vectors:
        console.print(f"  [yellow]⚠ No aligned vectors in {aligned_dir}[/yellow]"); return
    study_box = box(extent["minx"], extent["miny"], extent["maxx"], extent["maxy"])
    for vec_path in track(vectors, description=f"Rasterizing {scale}..."):
        try:
            gdf = gpd.read_file(vec_path)
            if gdf.empty or gdf.geometry.isna().all(): continue
            gdf = gdf.clip(study_box)
            if gdf.empty: continue
            layer_type = detect_layer_type(vec_path.stem)
            stem = vec_path.stem.replace("_utm43n", "")
            if layer_type == "distance":
                out_path = out_dir / f"{stem}_distance_edt.tif"
                if not out_path.exists(): rasterize_distance(gdf, profile, out_path)
                console.print(f"  [green]✓[/green] {vec_path.name} → EDT ({len(gdf)} features)")
            elif layer_type == "categorical":
                out_path = out_dir / f"{stem}_lithology_code.tif"
                cat_map_path = out_dir / f"{stem}_categories.json"
                if not out_path.exists(): rasterize_categorical(gdf, profile, out_path, cat_map_path)
                console.print(f"  [green]✓[/green] {vec_path.name} → categorical")
            elif layer_type == "binary":
                out_path = out_dir / f"{stem}_binary.tif"
                if not out_path.exists(): rasterize_binary(gdf, profile, out_path)
                console.print(f"  [green]✓[/green] {vec_path.name} → binary")
        except Exception as e:
            console.print(f"  [red]✗ {vec_path.name}: {e}[/red]")


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 03: Rasterize Geology ═══[/bold cyan]\n")
    extent = load_study_extent()
    process_scale(GEO25_ALIGNED, RASTER_25K, "1:25K", extent)
    process_scale(GEO50_ALIGNED, RASTER_50K, "1:50K", extent)
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/04_aeromagnetic_processing.py[/cyan]")
