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
