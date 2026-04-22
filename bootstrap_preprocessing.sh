#!/bin/bash
# =============================================================================
# bootstrap_preprocessing.sh
# Creates all preprocessing scripts in the correct folder structure.
# Run from: ~/sharon/mineral_mapping/
# Usage:    bash bootstrap_preprocessing.sh
# =============================================================================

set -e
BASE=~/sharon/mineral_mapping

echo "Creating folder structure..."
mkdir -p $BASE/src/preprocessing
touch $BASE/src/__init__.py
touch $BASE/src/preprocessing/__init__.py

# =============================================================================
# 01_unzip_inspect.py
# =============================================================================
cat > $BASE/src/preprocessing/01_unzip_inspect.py << 'PYEOF'
#!/usr/bin/env python3
"""
01_unzip_inspect.py
Unzip raw data files and inspect CRS, bounds, layers, and data types.
Run from: ~/sharon/mineral_mapping/
Usage:    python src/preprocessing/01_unzip_inspect.py
"""

import zipfile
import json
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

ROOT        = Path(__file__).resolve().parents[2]
RAW         = ROOT / "data" / "raw"
AEROMAG_DIR = RAW / "aeromagnetic"
GEO25_DIR   = RAW / "geological_25k"
GEO50_DIR   = RAW / "geological_50k"
MIN_OCC_DIR = RAW / "mineral_occurrences"
PROCESSED   = ROOT / "data" / "processed"
DOCS        = ROOT / "docs"

PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

ZIPS = {
    "aeromag": AEROMAG_DIR / "aeromag_tmi_v1.zip",
    "geo_25k": GEO25_DIR   / "geology_25k_v1.zip",
    "geo_50k": GEO50_DIR   / "geology_50k_v1.zip",
}

EXTRACT_DIRS = {
    "aeromag": AEROMAG_DIR  / "extracted",
    "geo_25k": GEO25_DIR    / "extracted",
    "geo_50k": GEO50_DIR    / "extracted",
}

TARGET_CRS = "EPSG:32643"


def unzip_all():
    console.rule("[bold cyan]Step 1 — Unzipping Raw Data[/bold cyan]")
    for name, zip_path in ZIPS.items():
        out_dir = EXTRACT_DIRS[name]
        if not zip_path.exists():
            console.print(f"  [yellow]⚠ Not found:[/yellow] {zip_path}")
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            console.print(f"  [dim]Already extracted:[/dim] {name}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"  Extracting [cyan]{name}[/cyan] ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
            members = z.namelist()
        console.print(f"  [green]✓[/green] {len(members)} files → {out_dir}")


def inspect_raster(path):
    with rasterio.open(path) as src:
        return {
            "file": path.name, "crs": str(src.crs),
            "bands": src.count, "width": src.width, "height": src.height,
            "res_x": round(src.res[0], 2), "res_y": round(src.res[1], 2),
            "dtype": src.dtypes[0], "bounds": [round(x, 4) for x in src.bounds],
            "nodata": src.nodata,
        }


def inspect_vector(path):
    gdf = gpd.read_file(path)
    return {
        "file": path.name, "crs": str(gdf.crs), "features": len(gdf),
        "geom_type": gdf.geom_type.value_counts().to_dict(),
        "columns": list(gdf.columns), "bounds": [round(x, 4) for x in gdf.total_bounds],
    }


def inspect_all():
    console.rule("[bold cyan]Step 2 — Inspecting Extracted Data[/bold cyan]")
    summary = {}
    for name, extract_dir in EXTRACT_DIRS.items():
        if not extract_dir.exists():
            console.print(f"  [yellow]⚠ Skipping {name} — not extracted[/yellow]")
            continue
        console.print(f"\n[bold]{name.upper()}[/bold]  ({extract_dir})")
        rasters = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff")) + \
                  list(extract_dir.rglob("*.asc")) + list(extract_dir.rglob("*.grd"))
        vectors = list(extract_dir.rglob("*.shp")) + list(extract_dir.rglob("*.geojson")) + \
                  list(extract_dir.rglob("*.gpkg"))
        files_only = [f for f in extract_dir.rglob("*") if f.is_file()]
        console.print(f"  Total files: {len(files_only)}  |  Rasters: {len(rasters)}  |  Vectors: {len(vectors)}")
        dataset_info = {"rasters": [], "vectors": []}
        if rasters:
            table = Table(title="Raster Files", show_lines=True)
            for col in ["File", "CRS", "Bands", "Width", "Height", "Res(m)", "DType"]:
                table.add_column(col)
            for r in rasters:
                try:
                    info = inspect_raster(r)
                    dataset_info["rasters"].append(info)
                    table.add_row(info["file"], info["crs"], str(info["bands"]),
                                  str(info["width"]), str(info["height"]),
                                  f"{info['res_x']}×{info['res_y']}", info["dtype"])
                except Exception as e:
                    table.add_row(r.name, f"ERROR: {e}", "", "", "", "", "")
            console.print(table)
        if vectors:
            table = Table(title="Vector Files", show_lines=True)
            for col in ["File", "CRS", "Features", "Geometry", "Key Columns"]:
                table.add_column(col, max_width=40)
            for v in vectors:
                try:
                    info = inspect_vector(v)
                    dataset_info["vectors"].append(info)
                    table.add_row(info["file"], info["crs"], str(info["features"]),
                                  str(info["geom_type"]), ", ".join(info["columns"][:6]))
                except Exception as e:
                    table.add_row(v.name, f"ERROR: {e}", "", "", "")
            console.print(table)
        if not rasters and not vectors:
            console.print("  [yellow]No raster/vector files found. Files:[/yellow]")
            for f in files_only[:20]:
                console.print(f"    {f.relative_to(extract_dir)}")
        summary[name] = dataset_info
    return summary


def check_crs_alignment(summary):
    console.rule("[bold cyan]Step 3 — CRS Alignment Check[/bold cyan]")
    all_crs = {}
    for name, data in summary.items():
        for r in data.get("rasters", []):
            all_crs[f"{name}/{r['file']}"] = r["crs"]
        for v in data.get("vectors", []):
            all_crs[f"{name}/{v['file']}"] = v["crs"]
    if not all_crs:
        console.print("  [yellow]No files inspected yet[/yellow]")
        return
    table = Table(title="CRS Summary", show_lines=True)
    table.add_column("Dataset/File", style="cyan")
    table.add_column("Current CRS")
    table.add_column("Needs Reproject?")
    for fname, crs in all_crs.items():
        needs = "[red]YES[/red]" if crs != TARGET_CRS else "[green]NO[/green]"
        table.add_row(fname, crs, needs)
    console.print(table)
    console.print(f"\n  [bold]Target CRS:[/bold] {TARGET_CRS} (UTM Zone 43N)")


def save_summary(summary):
    out_path = DOCS / "data_inspection.json"
    def serialize(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [serialize(i) for i in obj]
        return obj
    with open(out_path, "w") as f:
        json.dump(serialize(summary), f, indent=2)
    console.print(f"\n  [green]✓[/green] Summary saved → {out_path}")


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 01: Unzip & Inspect ═══[/bold cyan]\n")
    unzip_all()
    summary = inspect_all()
    check_crs_alignment(summary)
    save_summary(summary)
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/02_reproject_align.py[/cyan]")
PYEOF

echo "  ✓ 01_unzip_inspect.py"

# =============================================================================
# 02_reproject_align.py
# =============================================================================
cat > $BASE/src/preprocessing/02_reproject_align.py << 'PYEOF'
#!/usr/bin/env python3
"""
02_reproject_align.py
Reproject all rasters and vectors to UTM Zone 43N (EPSG:32643) at 50m resolution.
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import geopandas as gpd
from rich.console import Console
from rich.progress import track

console = Console()

ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ALIGNED   = PROCESSED / "aligned"

AEROMAG_EXT = RAW / "aeromagnetic"   / "extracted"
GEO25_EXT   = RAW / "geological_25k" / "extracted"
GEO50_EXT   = RAW / "geological_50k" / "extracted"
MIN_OCC_DIR = RAW / "mineral_occurrences"

ALIGNED_AEROMAG = ALIGNED / "aeromagnetic"
ALIGNED_GEO25   = ALIGNED / "geological_25k"
ALIGNED_GEO50   = ALIGNED / "geological_50k"
ALIGNED_MINOCC  = ALIGNED / "mineral_occurrences"

for d in [ALIGNED_AEROMAG, ALIGNED_GEO25, ALIGNED_GEO50, ALIGNED_MINOCC]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_CRS = CRS.from_epsg(32643)
TARGET_RES = 50.0


def reproject_raster(src_path, dst_path, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=TARGET_RES)
        kwargs = src.meta.copy()
        kwargs.update({"crs": TARGET_CRS, "transform": transform,
                       "width": width, "height": height, "driver": "GTiff",
                       "compress": "lzw", "tiled": True})
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=TARGET_CRS, resampling=resampling)
    return {"width": width, "height": height}


def reproject_vector(src_path, dst_path):
    gdf = gpd.read_file(src_path).to_crs(TARGET_CRS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dst_path, driver="GPKG")
    return {"features": len(gdf), "columns": list(gdf.columns)}


def process_aeromagnetic():
    console.rule("[bold cyan]Aeromagnetic Data[/bold cyan]")
    rasters = list(AEROMAG_EXT.rglob("*.tif")) + list(AEROMAG_EXT.rglob("*.asc")) + \
              list(AEROMAG_EXT.rglob("*.grd")) + list(AEROMAG_EXT.rglob("*.tiff"))
    if not rasters:
        console.print(f"  [yellow]⚠ No rasters in {AEROMAG_EXT}[/yellow]")
        return
    for r in track(rasters, description="Reprojecting TMI..."):
        dst = ALIGNED_AEROMAG / f"{r.stem}_utm43n_50m.tif"
        if dst.exists():
            console.print(f"  [dim]Already done: {dst.name}[/dim]"); continue
        try:
            info = reproject_raster(r, dst)
            console.print(f"  [green]✓[/green] {r.name} → {info['width']}×{info['height']} px")
        except Exception as e:
            console.print(f"  [red]✗ {r.name}: {e}[/red]")


def process_geological(ext_dir, out_dir, scale):
    console.rule(f"[bold cyan]Geological Map {scale}[/bold cyan]")
    vectors = list(ext_dir.rglob("*.shp")) + list(ext_dir.rglob("*.geojson")) + \
              list(ext_dir.rglob("*.gpkg"))
    if not vectors:
        console.print(f"  [yellow]⚠ No vectors in {ext_dir}[/yellow]"); return
    for v in track(vectors, description=f"Reprojecting {scale}..."):
        dst = out_dir / f"{v.stem}_utm43n.gpkg"
        if dst.exists():
            console.print(f"  [dim]Already done: {dst.name}[/dim]"); continue
        try:
            info = reproject_vector(v, dst)
            console.print(f"  [green]✓[/green] {v.name} ({info['features']} features)")
        except Exception as e:
            console.print(f"  [red]✗ {v.name}: {e}[/red]")


def process_mineral_occurrences():
    console.rule("[bold cyan]Mineral Occurrences[/bold cyan]")
    vectors = list(MIN_OCC_DIR.rglob("*.shp")) + list(MIN_OCC_DIR.rglob("*.geojson")) + \
              list(MIN_OCC_DIR.rglob("*.gpkg")) + list(MIN_OCC_DIR.rglob("*.csv"))
    if not vectors:
        console.print(f"  [yellow]⚠ No files in {MIN_OCC_DIR}[/yellow]"); return
    for v in vectors:
        if v.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(v)
            lat_cols = [c for c in df.columns if c.lower() in ["lat","latitude","y"]]
            lon_cols = [c for c in df.columns if c.lower() in ["lon","longitude","long","x"]]
            if lat_cols and lon_cols:
                gdf = gpd.GeoDataFrame(df,
                    geometry=gpd.points_from_xy(df[lon_cols[0]], df[lat_cols[0]]),
                    crs="EPSG:4326")
                dst = ALIGNED_MINOCC / f"{v.stem}_utm43n.gpkg"
                gdf.to_crs(TARGET_CRS).to_file(dst, driver="GPKG")
                console.print(f"  [green]✓[/green] {v.name} ({len(gdf)} occurrences)")
        else:
            dst = ALIGNED_MINOCC / f"{v.stem}_utm43n.gpkg"
            if dst.exists():
                console.print(f"  [dim]Already done: {dst.name}[/dim]"); continue
            try:
                info = reproject_vector(v, dst)
                console.print(f"  [green]✓[/green] {v.name} ({info['features']} features)")
            except Exception as e:
                console.print(f"  [red]✗ {v.name}: {e}[/red]")


def compute_study_extent():
    console.rule("[bold cyan]Computing Study Area Extent[/bold cyan]")
    all_bounds = []
    for tif in ALIGNED.rglob("*.tif"):
        with rasterio.open(tif) as src:
            b = src.bounds
            all_bounds.append((b.left, b.bottom, b.right, b.top))
    for gpkg in ALIGNED.rglob("*.gpkg"):
        try:
            b = gpd.read_file(gpkg).total_bounds
            all_bounds.append(tuple(b))
        except: pass
    if not all_bounds:
        console.print("  [yellow]⚠ No aligned files yet[/yellow]"); return None
    min_x = max(b[0] for b in all_bounds)
    min_y = max(b[1] for b in all_bounds)
    max_x = min(b[2] for b in all_bounds)
    max_y = min(b[3] for b in all_bounds)
    area_km2 = ((max_x - min_x) * (max_y - min_y)) / 1e6
    extent = {"crs": "EPSG:32643", "minx": min_x, "miny": min_y,
              "maxx": max_x, "maxy": max_y, "area_km2": area_km2,
              "width_px": int((max_x - min_x) / TARGET_RES),
              "height_px": int((max_y - min_y) / TARGET_RES),
              "resolution_m": TARGET_RES}
    with open(PROCESSED / "study_extent.json", "w") as f:
        json.dump(extent, f, indent=2)
    console.print(f"  Area: {area_km2:.0f} km²  |  Grid: {extent['width_px']}×{extent['height_px']} px")
    console.print(f"  [green]✓[/green] Saved: {PROCESSED / 'study_extent.json'}")
    return extent


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 02: Reproject & Align ═══[/bold cyan]\n")
    process_aeromagnetic()
    process_geological(GEO25_EXT, ALIGNED_GEO25, "1:25K")
    process_geological(GEO50_EXT, ALIGNED_GEO50, "1:50K")
    process_mineral_occurrences()
    compute_study_extent()
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/03_rasterize_geology.py[/cyan]")
PYEOF

echo "  ✓ 02_reproject_align.py"

# =============================================================================
# 03_rasterize_geology.py
# =============================================================================
cat > $BASE/src/preprocessing/03_rasterize_geology.py << 'PYEOF'
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
PYEOF

echo "  ✓ 03_rasterize_geology.py"

# =============================================================================
# 04_aeromagnetic_processing.py
# =============================================================================
cat > $BASE/src/preprocessing/04_aeromagnetic_processing.py << 'PYEOF'
#!/usr/bin/env python3
"""
04_aeromagnetic_processing.py
Process TMI: SVD filtering, regional/residual separation, 1VD, ASA, TDR derivatives.
"""

import json
import numpy as np
from pathlib import Path
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
import rasterio
from rich.console import Console

console = Console()

ROOT        = Path(__file__).resolve().parents[2]
PROCESSED   = ROOT / "data" / "processed"
ALIGNED     = PROCESSED / "aligned"
AEROMAG_AL  = ALIGNED / "aeromagnetic"
AEROMAG_OUT = PROCESSED / "aeromagnetic"
AEROMAG_OUT.mkdir(parents=True, exist_ok=True)
(AEROMAG_OUT / "normalized").mkdir(exist_ok=True)

TARGET_RES = 50.0


def svd_filter(arr, rank=50):
    console.print(f"    SVD filter rank={rank} ...")
    nan_mask = np.isnan(arr)
    a = np.where(nan_mask, np.nanmean(arr), arr)
    U, s, Vt = np.linalg.svd(a, full_matrices=False)
    s[rank:] = 0.0
    out = (U @ np.diag(s) @ Vt).astype("float32")
    out[nan_mask] = np.nan
    return out


def polynomial_trend(arr, degree=3):
    console.print(f"    Polynomial trend degree={degree} ...")
    rows, cols = arr.shape
    nan_mask = np.isnan(arr)
    y, x = np.meshgrid(np.linspace(-1,1,rows), np.linspace(-1,1,cols), indexing="ij")
    terms = []
    for i in range(degree+1):
        for j in range(degree+1-i):
            terms.append((x**j)*(y**i))
    A = np.column_stack([t.ravel() for t in terms])
    valid = ~nan_mask.ravel()
    coeffs, _, _, _ = np.linalg.lstsq(A[valid], arr.ravel()[valid], rcond=None)
    regional = (A @ coeffs).reshape(rows, cols).astype("float32")
    residual = (arr - regional).astype("float32")
    regional[nan_mask] = np.nan; residual[nan_mask] = np.nan
    return regional, residual


def wavenumber_grids(rows, cols, dx):
    kx = fftfreq(cols, d=dx) * 2 * np.pi
    ky = fftfreq(rows, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX**2 + KY**2)
    KR[0,0] = 1e-10
    return KX, KY, KR


def first_vertical_derivative(arr, dx=TARGET_RES):
    console.print("    Computing 1VD ...")
    nan_mask = np.isnan(arr)
    a = np.where(nan_mask, 0.0, arr)
    _, _, KR = wavenumber_grids(*a.shape, dx)
    out = np.real(ifft2(fft2(a) * KR)).astype("float32")
    out[nan_mask] = np.nan; return out


def analytical_signal_amplitude(arr, dx=TARGET_RES):
    console.print("    Computing ASA ...")
    nan_mask = np.isnan(arr)
    a = np.where(nan_mask, 0.0, arr)
    KX, KY, KR = wavenumber_grids(*a.shape, dx)
    F = fft2(a)
    dTdx = np.real(ifft2(F * 1j * KX))
    dTdy = np.real(ifft2(F * 1j * KY))
    dTdz = np.real(ifft2(F * KR))
    out = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2).astype("float32")
    out[nan_mask] = np.nan; return out


def tilt_derivative(arr, dx=TARGET_RES):
    console.print("    Computing TDR ...")
    nan_mask = np.isnan(arr)
    a = np.where(nan_mask, 0.0, arr)
    KX, KY, KR = wavenumber_grids(*a.shape, dx)
    F = fft2(a)
    dTdx = np.real(ifft2(F * 1j * KX))
    dTdy = np.real(ifft2(F * 1j * KY))
    dTdz = np.real(ifft2(F * KR))
    horiz = np.maximum(np.sqrt(dTdx**2 + dTdy**2), 1e-10)
    out = np.arctan2(dTdz, horiz).astype("float32")
    out[nan_mask] = np.nan; return out


def upward_continuation(arr, height, dx=TARGET_RES):
    console.print(f"    Upward continuation h={height}m ...")
    nan_mask = np.isnan(arr)
    a = np.where(nan_mask, 0.0, arr)
    _, _, KR = wavenumber_grids(*a.shape, dx)
    out = np.real(ifft2(fft2(a) * np.exp(-KR * height))).astype("float32")
    out[nan_mask] = np.nan; return out


def robust_normalize(arr):
    valid = arr[~np.isnan(arr)]
    p2, p98 = np.percentile(valid, [2, 98])
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1).astype("float32")


def save_raster(arr, out_path, template_path, desc=""):
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
    profile.update({"dtype": "float32", "count": 1, "compress": "lzw",
                    "tiled": True, "nodata": float("nan")})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    console.print(f"    [green]✓[/green] {out_path.name}")


def process_tmi(tmi_path):
    console.print(f"\n  [cyan]{tmi_path.name}[/cyan]")
    with rasterio.open(tmi_path) as src:
        tmi_raw = src.read(1).astype("float32")
        nodata  = src.nodata
        dx      = src.res[0]
    if nodata is not None: tmi_raw[tmi_raw == nodata] = np.nan
    console.print(f"    Grid: {tmi_raw.shape}  TMI range: {np.nanmin(tmi_raw):.1f}–{np.nanmax(tmi_raw):.1f} nT")

    rank = min(50, min(tmi_raw.shape) // 2)
    tmi_f = svd_filter(tmi_raw, rank=rank)
    save_raster(tmi_f, AEROMAG_OUT / "tmi_filtered.tif", tmi_path)

    tmi_reg, tmi_res = polynomial_trend(tmi_f, degree=3)
    save_raster(tmi_reg, AEROMAG_OUT / "tmi_regional.tif", tmi_path)
    save_raster(tmi_res, AEROMAG_OUT / "tmi_residual.tif", tmi_path)

    tmi_1vd = first_vertical_derivative(tmi_f, dx=dx)
    save_raster(tmi_1vd, AEROMAG_OUT / "tmi_1vd.tif", tmi_path)

    tmi_asa = analytical_signal_amplitude(tmi_f, dx=dx)
    save_raster(tmi_asa, AEROMAG_OUT / "tmi_asa.tif", tmi_path)

    tmi_tdr = tilt_derivative(tmi_f, dx=dx)
    save_raster(tmi_tdr, AEROMAG_OUT / "tmi_tdr.tif", tmi_path)

    for h in [500, 1000, 2000]:
        save_raster(upward_continuation(tmi_f, h, dx), AEROMAG_OUT / f"tmi_uc_{h}m.tif", tmi_path)

    norm_dir = AEROMAG_OUT / "normalized"
    for name, arr in [("tmi", tmi_f), ("regional", tmi_reg), ("residual", tmi_res),
                       ("1vd", tmi_1vd), ("asa", tmi_asa), ("tdr", tmi_tdr)]:
        save_raster(robust_normalize(arr), norm_dir / f"{name}_norm.tif", tmi_path)

    meta = {"source": str(tmi_path), "grid_shape": list(tmi_raw.shape),
            "resolution_m": dx, "svd_rank": rank}
    with open(AEROMAG_OUT / "processing_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    console.print("\n[bold cyan]═══ Step 04: Aeromagnetic Processing ═══[/bold cyan]\n")
    tmi_files = list(AEROMAG_AL.glob("*.tif"))
    if not tmi_files:
        console.print(f"  [yellow]⚠ No aligned TMI in {AEROMAG_AL}[/yellow]")
        raise SystemExit(0)
    for tmi_path in tmi_files:
        process_tmi(tmi_path)
    console.print("\n[bold]Next:[/bold] [cyan]python src/preprocessing/05_feature_engineering.py[/cyan]")
PYEOF

echo "  ✓ 04_aeromagnetic_processing.py"

# =============================================================================
# 05_feature_engineering.py
# =============================================================================
cat > $BASE/src/preprocessing/05_feature_engineering.py << 'PYEOF'
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
PYEOF

echo "  ✓ 05_feature_engineering.py"

# =============================================================================
# 06_generate_labels.py
# =============================================================================
cat > $BASE/src/preprocessing/06_generate_labels.py << 'PYEOF'
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
PYEOF

echo "  ✓ 06_generate_labels.py"

# =============================================================================
# 07_stack_channels.py
# =============================================================================
cat > $BASE/src/preprocessing/07_stack_channels.py << 'PYEOF'
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
PYEOF

echo "  ✓ 07_stack_channels.py"

# =============================================================================
# run_preprocessing.sh
# =============================================================================
cat > $BASE/run_preprocessing.sh << 'SHEOF'
#!/bin/bash
set -e
source ~/sharon/mineral_mapping/activate.sh
cd ~/sharon/mineral_mapping
echo "========================================================"
echo " Mineral Mapping — Full Preprocessing Pipeline"
echo "========================================================"
python src/preprocessing/01_unzip_inspect.py
python src/preprocessing/02_reproject_align.py
python src/preprocessing/03_rasterize_geology.py
python src/preprocessing/04_aeromagnetic_processing.py
python src/preprocessing/05_feature_engineering.py
python src/preprocessing/06_generate_labels.py
python src/preprocessing/07_stack_channels.py
echo "========================================================"
echo " ✅  Done! Output: data/processed/stacked/multiband_input.tif"
echo "========================================================"
SHEOF

chmod +x $BASE/run_preprocessing.sh
echo "  ✓ run_preprocessing.sh"

echo ""
echo "========================================================"
echo " ✅  All scripts created!"
echo ""
echo " Folder structure:"
echo "   src/preprocessing/01_unzip_inspect.py"
echo "   src/preprocessing/02_reproject_align.py"
echo "   src/preprocessing/03_rasterize_geology.py"
echo "   src/preprocessing/04_aeromagnetic_processing.py"
echo "   src/preprocessing/05_feature_engineering.py"
echo "   src/preprocessing/06_generate_labels.py"
echo "   src/preprocessing/07_stack_channels.py"
echo "   run_preprocessing.sh"
echo ""
echo " Now run:"
echo "   bash run_preprocessing.sh"
echo "========================================================"