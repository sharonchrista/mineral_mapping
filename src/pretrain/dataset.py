"""
src/pretrain/dataset.py
Tiles the 15-channel multiband GeoTIFF into 256x256 patches for MIM pre-training.
"""

import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class GeoTileDataset(Dataset):
    """
    Tiles multiband_input.tif into spatial patches for MIM pre-training.

    Args:
        tif_path   : path to multiband_input.tif
        patch_size : tile size in pixels (default 256)
        stride     : step between tiles (default = patch_size)
        augment    : random rotation, flip, gaussian noise
        min_valid  : minimum fraction of non-zero pixels to keep a tile
        noise_sigma: std of gaussian noise (proposal 4.2.2: 0.01)
    """

    def __init__(self, tif_path, patch_size=256, stride=None,
                 augment=True, min_valid=0.5, noise_sigma=0.01):
        self.tif_path    = Path(tif_path)
        self.patch_size  = patch_size
        self.stride      = stride if stride is not None else patch_size
        self.augment     = augment
        self.min_valid   = min_valid
        self.noise_sigma = noise_sigma

        with rasterio.open(self.tif_path) as src:
            self.data = src.read().astype("float32")   # (C, H, W)
            self.n_channels = src.count
            self.H = src.height
            self.W = src.width

        self.data = np.where(self.data == -9999.0, 0.0, self.data)
        self.tiles = self._build_tile_index()
        print(f"GeoTileDataset: {len(self.tiles)} tiles | "
              f"patch={patch_size}px | stride={self.stride}px | "
              f"channels={self.n_channels}")

    def _build_tile_index(self):
        tiles = []
        P, S = self.patch_size, self.stride
        for r in range(0, self.H - P + 1, S):
            for c in range(0, self.W - P + 1, S):
                patch = self.data[:, r:r+P, c:c+P]
                if (patch.sum(axis=0) != 0).mean() >= self.min_valid:
                    tiles.append((r, c))
        return tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        r, c = self.tiles[idx]
        P = self.patch_size
        patch = torch.from_numpy(self.data[:, r:r+P, c:c+P].copy())
        if self.augment:
            patch = self._augment(patch)
        return patch

    def _augment(self, patch):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            patch = TF.rotate(patch, angle)
        if random.random() > 0.5:
            patch = TF.hflip(patch)
        if random.random() > 0.5:
            patch = TF.vflip(patch)
        if self.noise_sigma > 0:
            patch = torch.clamp(patch + torch.randn_like(patch) * self.noise_sigma, 0.0, 1.0)
        return patch


def build_dataloaders(tif_path, patch_size=256, stride=128, batch_size=32,
                      num_workers=8, val_fraction=0.1, noise_sigma=0.01, seed=42):
    from torch.utils.data import DataLoader, random_split

    full_ds = GeoTileDataset(tif_path=tif_path, patch_size=patch_size,
                              stride=stride, augment=True, noise_sigma=noise_sigma)
    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    val_ds.dataset.augment = False

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, full_ds.n_channels


if __name__ == "__main__":
    tif = Path("data/processed/stacked/multiband_input.tif")
    train_loader, val_loader, n_ch = build_dataloaders(
        tif, patch_size=256, stride=256, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}  dtype: {batch.dtype}  "
          f"min: {batch.min():.3f}  max: {batch.max():.3f}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print("Dataset OK")
