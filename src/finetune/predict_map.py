"""
src/finetune/predict_map.py
Generate full prospectivity map GeoTIFFs with MC uncertainty.
Outputs per depth: prospectivity_mean_<depth>.tif, prospectivity_uncertainty_<depth>.tif
Plus: prospectivity_ensemble_mean.tif

Usage:
  python src/finetune/predict_map.py --fold 0
  python src/finetune/predict_map.py --fold 0 --mc_passes 50
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import rasterio
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.finetune.model_fusion import MineralProspectivityModel, DEPTH_INTERVALS
from src.finetune.dataset import FullRasterDataset

console = Console()


def save_map(arr, out_path, meta, desc=""):
    profile = {"driver":"GTiff","dtype":"float32","crs":meta["crs"],
               "transform":meta["transform"],"width":meta["width"],
               "height":meta["height"],"count":1,"compress":"lzw",
               "tiled":True,"nodata":-9999.0}
    arr_out = np.where(np.isnan(arr), -9999.0, arr).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_out, 1)
        if desc: dst.update_tags(description=desc)
    valid = arr[~np.isnan(arr)]
    console.print(f"  Saved: {out_path.name}  [min={valid.min():.3f} max={valid.max():.3f}]")


def predict_map(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"]) / f"fold{config['fold']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(config["ckpt_dir"]) / f"fold{config['fold']}" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_path}[/red]"); return

    console.print(f"Loading: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = ckpt["config"]
    model = MineralProspectivityModel(
        pretrain_ckpt=None, img_size=cfg.get("img_size",256),
        patch_size=cfg.get("patch_size",16),
        encoder_embed_dim=cfg.get("encoder_embed_dim",768),
        encoder_depth=cfg.get("encoder_depth",12),
        encoder_num_heads=cfg.get("encoder_num_heads",12),
        cca_dim=cfg.get("cca_dim",256), fusion_dim=cfg.get("fusion_dim",512),
        mc_dropout=cfg.get("mc_dropout",0.2),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    console.print(f"  Epoch {ckpt['epoch']}  val_auc_pr={ckpt.get('val_auc_pr','N/A')}")

    ds = FullRasterDataset(config["tif_path"], config["patch_size"], config["patch_size"]//2)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=True)

    with rasterio.open(config["tif_path"]) as src:
        meta = {"crs":src.crs,"transform":src.transform,"height":src.height,"width":src.width}
    H, W, half = meta["height"], meta["width"], config["patch_size"]//2

    mean_maps = {d: np.zeros((H,W), dtype="float64") for d in DEPTH_INTERVALS}
    std_maps  = {d: np.zeros((H,W), dtype="float64") for d in DEPTH_INTERVALS}
    count_map = np.zeros((H,W), dtype="float64")

    for batch in track(loader, description="Predicting..."):
        patches = batch["patch"].to(device)
        rows    = batch["row"].numpy()
        cols    = batch["col"].numpy()
        mc      = model.predict_proba(patches, mc_passes=config["mc_passes"])
        for i in range(len(rows)):
            r,c = int(rows[i]), int(cols[i])
            r0,r1,c0,c1 = r-half,r+half,c-half,c+half
            count_map[r0:r1,c0:c1] += 1.0
            for d in DEPTH_INTERVALS:
                mean_maps[d][r0:r1,c0:c1] += mc[d]["mean"][i].item()
                std_maps[d][r0:r1,c0:c1]  += mc[d]["std"][i].item()

    cnt = np.where(count_map>0, count_map, np.nan)
    for d in DEPTH_INTERVALS:
        mean_maps[d] /= cnt; std_maps[d] /= cnt

    console.rule("Saving maps")
    for d in DEPTH_INTERVALS:
        save_map(mean_maps[d].astype("float32"), out_dir/f"prospectivity_mean_{d}.tif",   meta, f"Prospectivity mean depth={d}")
        save_map(std_maps[d].astype("float32"),  out_dir/f"prospectivity_uncertainty_{d}.tif", meta, f"Uncertainty depth={d}")

    ensemble = np.nanmean(np.stack([mean_maps[d] for d in DEPTH_INTERVALS],0), 0)
    save_map(ensemble.astype("float32"), out_dir/"prospectivity_ensemble_mean.tif", meta, "Ensemble mean")
    console.print(f"\nAll maps -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--mc_passes",  type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()
    predict_map({
        "tif_path":   "data/processed/stacked/multiband_input.tif",
        "output_dir": "results/prospectivity_maps",
        "ckpt_dir":   "experiments/finetune",
        "patch_size": 256, "batch_size": args.batch_size,
        "num_workers": args.workers, "mc_passes": args.mc_passes,
        "fold": args.fold,
    })
