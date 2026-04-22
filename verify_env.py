#!/usr/bin/env python3
"""
verify_env.py — Run after setup_env.sh to confirm everything installed correctly.
Usage: python verify_env.py
"""

import sys
import importlib
from rich.console import Console
from rich.table import Table

console = Console()

CHECKS = [
    # (display_name, import_name, extra_check_fn)
    # ── PyTorch ──────────────────────────────────────────────
    ("torch",               "torch",                None),
    ("torchvision",         "torchvision",          None),
    ("pytorch_lightning",   "lightning",            None),
    ("timm",                "timm",                 None),
    ("transformers",        "transformers",         None),
    ("einops",              "einops",               None),
    ("torchmetrics",        "torchmetrics",         None),
    # ── Geospatial ───────────────────────────────────────────
    ("gdal",                "osgeo.gdal",           None),
    ("rasterio",            "rasterio",             None),
    ("geopandas",           "geopandas",            None),
    ("shapely",             "shapely",              None),
    ("pyproj",              "pyproj",               None),
    ("rioxarray",           "rioxarray",            None),
    ("xarray",              "xarray",               None),
    ("earthpy",             "earthpy",              None),
    # ── Geophysics ───────────────────────────────────────────
    ("SimPEG",              "SimPEG",               None),
    ("discretize",          "discretize",           None),
    ("verde",               "verde",                None),
    ("harmonica",           "harmonica",            None),
    ("empymod",             "empymod",              None),
    # ── ML / Science ─────────────────────────────────────────
    ("scikit-learn",        "sklearn",              None),
    ("numpy",               "numpy",                None),
    ("pandas",              "pandas",               None),
    ("scipy",               "scipy",                None),
    ("imbalanced-learn",    "imblearn",             None),
    # ── XAI ──────────────────────────────────────────────────
    ("shap",                "shap",                 None),
    ("captum",              "captum",               None),
    # ── Visualization ────────────────────────────────────────
    ("matplotlib",          "matplotlib",           None),
    ("plotly",              "plotly",               None),
    ("seaborn",             "seaborn",              None),
    # ── Tracking ─────────────────────────────────────────────
    ("wandb",               "wandb",                None),
    ("mlflow",              "mlflow",               None),
    # ── Config ───────────────────────────────────────────────
    ("hydra",               "hydra",                None),
    ("omegaconf",           "omegaconf",            None),
    # ── I/O ──────────────────────────────────────────────────
    ("h5py",                "h5py",                 None),
    ("zarr",                "zarr",                 None),
    ("dask",                "dask",                 None),
    # ── Web ──────────────────────────────────────────────────
    ("streamlit",           "streamlit",            None),
    ("fastapi",             "fastapi",              None),
]

def check_import(import_name):
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "n/a")
        return True, version
    except Exception as e:
        return False, str(e)

def check_gpu():
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        if available:
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return True, f"{count}× GPU: {', '.join(names)}"
        else:
            return False, "CUDA not available — CPU only"
    except Exception as e:
        return False, str(e)

def main():
    console.print("\n[bold cyan]🔍 Mineral Mapping — Environment Verification[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Package",    style="cyan",  width=22)
    table.add_column("Status",     style="white", width=10)
    table.add_column("Version",    style="white", width=14)
    table.add_column("Notes",      style="dim",   width=40)

    passed = 0
    failed = 0

    for display_name, import_name, _ in CHECKS:
        ok, info = check_import(import_name)
        if ok:
            status = "[green]✓  OK[/green]"
            version = info
            notes = ""
            passed += 1
        else:
            status = "[red]✗ FAIL[/red]"
            version = ""
            notes = info[:60]
            failed += 1
        table.add_row(display_name, status, version, notes)

    console.print(table)

    # GPU check
    gpu_ok, gpu_info = check_gpu()
    gpu_status = "[green]✓  OK[/green]" if gpu_ok else "[yellow]⚠ WARN[/yellow]"
    console.print(f"\n[bold]GPU Status:[/bold] {gpu_status}  {gpu_info}")

    # Python info
    console.print(f"[bold]Python:[/bold]  {sys.version}")

    # Summary
    console.print(f"\n[bold]Summary:[/bold] [green]{passed} passed[/green]  |  [red]{failed} failed[/red]")

    if failed == 0:
        console.print("\n[bold green]✅  All checks passed! Environment is ready.[/bold green]")
    else:
        console.print(f"\n[bold red]❌  {failed} package(s) failed. Re-run setup_env.sh or install manually.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
