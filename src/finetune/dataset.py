"""
src/finetune/dataset.py
PU-labeled dataset for prospectivity fine-tuning.
"""

import numpy as np
from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import random


class ProspectivityDataset(Dataset):
    def __init__(self, tif_path, label_path, patch_size=256, augment=True,
                 fold_mask_path=None, unlabeled_ratio=5.0, noise_sigma=0.01):
        self.patch_size      = patch_size
        self.augment         = augment
        self.unlabeled_ratio = unlabeled_ratio
        self.noise_sigma     = noise_sigma
        self.half            = patch_size // 2

        with rasterio.open(tif_path) as src:
            self.data = src.read().astype("float32")
            self.H, self.W = src.height, src.width
        self.data = np.where(self.data == -9999.0, 0.0, self.data)

        with rasterio.open(label_path) as src:
            self.labels = src.read(1).astype("uint8")

        self.fold_mask = None
        if fold_mask_path is not None and Path(fold_mask_path).exists():
            with rasterio.open(fold_mask_path) as src:
                self.fold_mask = src.read(1).astype("uint8")

        self.pos_pixels, self.unl_pixels = self._find_pixels()
        self.samples = self._build_sample_list()

        print(f"ProspectivityDataset:")
        print(f"  Positive : {len(self.pos_pixels)} px")
        print(f"  Unlabeled: {len(self.unl_pixels)} px")
        print(f"  Samples  : {len(self.samples)}")

    def _find_pixels(self):
        H, W, P = self.H, self.W, self.half
        valid = np.zeros((H,W), dtype=bool)
        valid[P:H-P, P:W-P] = True
        if self.fold_mask is not None:
            valid = valid & (self.fold_mask > 0)
        pos = (self.labels==1) & valid
        unl = (self.labels==0) & valid
        pr, pc = np.where(pos); ur, uc = np.where(unl)
        return list(zip(pr.tolist(),pc.tolist())), list(zip(ur.tolist(),uc.tolist()))

    def _build_sample_list(self, seed=42):
        n_unl = min(len(self.unl_pixels), int(len(self.pos_pixels)*self.unlabeled_ratio))
        rng = np.random.default_rng(seed)
        sampled = self.unl_pixels if len(self.unl_pixels)<=n_unl else \
                  [self.unl_pixels[i] for i in rng.choice(len(self.unl_pixels),n_unl,replace=False)]
        samples = [(r,c,1,1.0) for r,c in self.pos_pixels] + \
                  [(r,c,0,0.1) for r,c in sampled]
        rng.shuffle(samples)
        return samples

    def resample(self, seed=None):
        self.samples = self._build_sample_list(seed=seed if seed is not None else 42)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        r,c,label,weight = self.samples[idx]
        H2 = self.half
        patch = torch.from_numpy(self.data[:, r-H2:r+H2, c-H2:c+H2].copy())
        if self.augment: patch = self._augment(patch)
        return {"patch": patch, "label": torch.tensor(label,dtype=torch.float32),
                "weight": torch.tensor(weight,dtype=torch.float32),
                "row": torch.tensor(r,dtype=torch.long),
                "col": torch.tensor(c,dtype=torch.long)}

    def _augment(self, patch):
        a = random.choice([0,90,180,270])
        if a: patch = TF.rotate(patch, a)
        if random.random()>0.5: patch = TF.hflip(patch)
        if random.random()>0.5: patch = TF.vflip(patch)
        if self.noise_sigma>0:
            patch = torch.clamp(patch + torch.randn_like(patch)*self.noise_sigma, 0.0, 1.0)
        return patch

    def get_sample_weights(self):
        return torch.tensor([s[3] for s in self.samples], dtype=torch.float32)


class FullRasterDataset(Dataset):
    def __init__(self, tif_path, patch_size=256, stride=None, min_valid=0.3):
        self.patch_size = patch_size
        self.half       = patch_size // 2
        self.stride     = stride if stride is not None else patch_size // 2
        with rasterio.open(tif_path) as src:
            self.data = src.read().astype("float32")
            self.H, self.W = src.height, src.width
            self.transform = src.transform; self.crs = src.crs
        self.data  = np.where(self.data==-9999.0, 0.0, self.data)
        self.tiles = self._build_index(min_valid)
        print(f"FullRasterDataset: {len(self.tiles)} inference tiles")

    def _build_index(self, min_valid):
        tiles = []
        H2, S = self.half, self.stride
        for r in range(H2, self.H-H2, S):
            for c in range(H2, self.W-H2, S):
                if (self.data[:, r-H2:r+H2, c-H2:c+H2].sum(0)!=0).mean() >= min_valid:
                    tiles.append((r,c))
        return tiles

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        r,c = self.tiles[idx]; H2 = self.half
        return {"patch": torch.from_numpy(self.data[:,r-H2:r+H2,c-H2:c+H2].copy()),
                "row": torch.tensor(r,dtype=torch.long),
                "col": torch.tensor(c,dtype=torch.long)}


def build_finetune_loaders(tif_path, label_path, train_fold_mask, val_fold_mask,
                            patch_size=256, batch_size=32, num_workers=8, unlabeled_ratio=5.0):
    train_ds = ProspectivityDataset(tif_path, label_path, patch_size, True,
                                     train_fold_mask, unlabeled_ratio)
    val_ds   = ProspectivityDataset(tif_path, label_path, patch_size, False,
                                     val_fold_mask, unlabeled_ratio)
    weights  = train_ds.get_sample_weights()
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, persistent_workers=num_workers>0)
    return DataLoader(train_ds, sampler=sampler, **kw), DataLoader(val_ds, shuffle=False, **kw)
