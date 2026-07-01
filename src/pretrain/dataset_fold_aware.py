"""
src/pretrain/dataset_fold_aware.py

Fold-aware MIM pre-training dataset.

Key fix (Reviewer #1, Comment 1):
    The original GeoTileDataset tiles the full 42,291 km² raster with no
    spatial fold awareness, meaning the ViT encoder sees test-region patches
    during MIM pre-training before fine-tuning.  This module adds a
    `exclude_mask` argument: any tile whose centre pixel is marked True in
    that mask is dropped from the pre-training pool.

    For fold k, pass:
        exclude_mask_path = "data/splits/fold{k}_test_mask.tif"

    The test mask already incorporates the 20 km exclusion buffer
    (FOLD_BUFFER_M = 20_000 in 06_generate_labels.py), so no additional
    buffering is needed here.

Usage
-----
    from src.pretrain.dataset_fold_aware import build_dataloaders_for_fold
    train_loader, val_loader, n_ch = build_dataloaders_for_fold(
        tif_path            = "data/processed/stacked/multiband_input.tif",
        exclude_mask_path   = "data/splits/fold0_test_mask.tif",
        fold_idx            = 0,
        patch_size          = 256,
        stride              = 128,
        batch_size          = 16,
        num_workers         = 4,
    )
"""

import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


class GeoTileDatasetFoldAware(Dataset):
    """
    Tiles multiband_input.tif into 256×256 patches for MIM pre-training,
    excluding any tile whose centre pixel falls inside `exclude_mask`.

    Args
    ----
    tif_path           : path to multiband_input.tif
    exclude_mask_path  : path to fold{k}_test_mask.tif  (uint8, 1 = exclude)
                         Pass None to disable masking (replicates original behaviour).
    patch_size         : tile size in pixels (default 256)
    stride             : step between tile centres (default = patch_size // 2)
    augment            : random rotation / flip / gaussian noise
    min_valid          : minimum fraction of non-zero pixels to keep a tile
    noise_sigma        : std of gaussian noise added during augmentation
    fold_idx           : fold number (used only for logging)
    """

    def __init__(self, tif_path, exclude_mask_path=None, patch_size=256,
                 stride=None, augment=True, min_valid=0.5,
                 noise_sigma=0.01, fold_idx=None):

        self.tif_path    = Path(tif_path)
        self.patch_size  = patch_size
        self.stride      = stride if stride is not None else patch_size
        self.augment     = augment
        self.min_valid   = min_valid
        self.noise_sigma = noise_sigma
        self.fold_idx    = fold_idx

        # ── Load multiband raster ──────────────────────────────────────────
        with rasterio.open(self.tif_path) as src:
            self.data       = src.read().astype("float32")   # (C, H, W)
            self.n_channels = src.count
            self.H          = src.height
            self.W          = src.width

        # Replace nodata with zero
        self.data = np.where(self.data == -9999.0, 0.0, self.data)

        # ── Load exclusion mask → derive column range ──────────────────────
        # The test mask marks only positive-label pixels within the test
        # strip, not the full strip.  We therefore derive the excluded
        # COLUMN RANGE from the mask's non-zero columns and use that to
        # reject any tile whose centre column falls inside the strip.
        self.exclude_col_min = None
        self.exclude_col_max = None
        if exclude_mask_path is not None:
            ep = Path(exclude_mask_path)
            if not ep.exists():
                raise FileNotFoundError(
                    f"Exclusion mask not found: {ep}\n"
                    f"Run src/preprocessing/06_generate_labels.py first."
                )
            with rasterio.open(ep) as src:
                mask_arr = src.read(1).astype("uint8")
            assert mask_arr.shape == (self.H, self.W), (
                f"Exclusion mask shape {mask_arr.shape} does not "
                f"match raster shape ({self.H}, {self.W})"
            )
            cols_with_mask = np.where(mask_arr.any(axis=0))[0]
            if len(cols_with_mask) == 0:
                print(f"  WARNING: exclusion mask has no non-zero columns — "
                      f"no tiles will be excluded")
            else:
                self.exclude_col_min = int(cols_with_mask.min())
                self.exclude_col_max = int(cols_with_mask.max())
                print(f"  Exclusion column range: {self.exclude_col_min} – "
                      f"{self.exclude_col_max} (from mask non-zero columns)")

        # ── Build tile index ───────────────────────────────────────────────
        self.tiles, n_excluded = self._build_tile_index()

        fold_str = f"fold {fold_idx}" if fold_idx is not None else "no fold masking"
        print(
            f"GeoTileDatasetFoldAware [{fold_str}]: "
            f"{len(self.tiles)} tiles kept | "
            f"{n_excluded} tiles excluded (test region) | "
            f"patch={patch_size}px | stride={self.stride}px | "
            f"channels={self.n_channels}"
        )

    # ──────────────────────────────────────────────────────────────────────
    def _build_tile_index(self):
        """
        Build list of (row, col) top-left corners of valid tiles.

        A tile is excluded if:
          (a) it has fewer than min_valid non-zero pixels, OR
          (b) its centre pixel is marked 1 in exclude_mask (test region).
        """
        tiles      = []
        n_excluded = 0
        P          = self.patch_size
        S          = self.stride

        for r in range(0, self.H - P + 1, S):
            for c in range(0, self.W - P + 1, S):

                # Spatial content check
                patch = self.data[:, r:r+P, c:c+P]
                if (patch.sum(axis=0) != 0).mean() < self.min_valid:
                    continue

                # Fold exclusion check — use centre column range
                if self.exclude_col_min is not None:
                    centre_c = c + P // 2
                    if self.exclude_col_min <= centre_c <= self.exclude_col_max:
                        n_excluded += 1
                        continue

                tiles.append((r, c))

        return tiles, n_excluded

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        r, c = self.tiles[idx]
        P    = self.patch_size
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
            patch = torch.clamp(
                patch + torch.randn_like(patch) * self.noise_sigma, 0.0, 1.0
            )
        return patch


# ──────────────────────────────────────────────────────────────────────────────
def build_dataloaders_for_fold(
    tif_path,
    exclude_mask_path,
    fold_idx,
    patch_size    = 256,
    stride        = 128,
    batch_size    = 16,
    num_workers   = 4,
    val_fraction  = 0.1,
    noise_sigma   = 0.01,
    seed          = 42,
):
    """
    Build train/val DataLoaders for MIM pre-training of fold `fold_idx`.

    The val split is drawn randomly from the *non-excluded* tiles only,
    so validation also never sees test-fold data.

    Returns
    -------
    train_loader, val_loader, n_channels
    """
    full_ds = GeoTileDatasetFoldAware(
        tif_path          = tif_path,
        exclude_mask_path = exclude_mask_path,
        patch_size        = patch_size,
        stride            = stride,
        augment           = True,
        noise_sigma       = noise_sigma,
        fold_idx          = fold_idx,
    )

    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    # Disable augmentation for validation
    val_ds.dataset.augment = False

    kw = dict(
        batch_size        = batch_size,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers = num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, full_ds.n_channels


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    tif  = Path("data/processed/stacked/multiband_input.tif")
    mask = Path("data/splits/fold0_test_mask.tif")

    if not tif.exists():
        print(f"ERROR: {tif} not found. Run from ~/sharon/mineral_mapping/")
        sys.exit(1)

    train_loader, val_loader, n_ch = build_dataloaders_for_fold(
        tif_path          = tif,
        exclude_mask_path = mask,
        fold_idx          = 0,
        patch_size        = 256,
        stride            = 256,   # wider stride for quick test
        batch_size        = 4,
        num_workers       = 0,
    )
    batch = next(iter(train_loader))
    print(f"Batch shape : {batch.shape}")
    print(f"dtype       : {batch.dtype}")
    print(f"min / max   : {batch.min():.3f} / {batch.max():.3f}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print("dataset_fold_aware.py OK")