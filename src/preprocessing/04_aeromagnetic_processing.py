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
