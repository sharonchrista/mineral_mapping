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
