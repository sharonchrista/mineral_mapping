# Mineral Prospectivity Mapping — Deep Multi-Modal Fusion

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper:** *Deep Multi-Modal Fusion for Depth-Stratified 2D Mineral Prospectivity Mapping: Self-Supervised Vision Transformers Integrating Multi-Scale Geological Maps with Aeromagnetic Data*
> **Author:** Sharon Christa, MIT Art Design and Technology University, Pune, India
> **Target Journal:** Natural Resources Research (Springer, Q1, IF 5.0)
> **Status:** Under Review

---

## Overview

This repository contains the full implementation of a self-supervised multi-modal fusion framework for mineral prospectivity mapping (MPM) applied to the **Dharwar Craton**, Karnataka and Andhra Pradesh, India (42,291 km², 50 m resolution).

### Key Contributions

| Contribution | Description |
|---|---|
| **MIM Pre-training** | ViT-Base pre-trained on full unlabeled 15-channel geological-geophysical raster via Masked Image Modeling — no labels required |
| **Multi-modal Fusion** | Dual-branch encoder (geo + aero) with cross-modal attention and deep CCA fusion |
| **Depth-stratified** | Three prospectivity heads: 0–500 m, 500–1,000 m, 1,000–2,000 m |
| **PU Learning** | Positive-Unlabeled framework with MC dropout uncertainty quantification (T=50) |
| **Ablation Study** | Systematic component analysis confirming contribution of each architectural element |

### Results Summary

| Method | AUC-PR | AUC-ROC | F1 |
|---|---|---|---|
| **Ours (ViT-MIM)** | **0.9958 ± 0.0013** | **0.9988 ± 0.0004** | **0.9934 ± 0.0019** |
| Random Forest | 0.8369 ± 0.1191 | 0.9385 ± 0.0335 | 0.7168 ± 0.1879 |
| ViT (no pre-training) | 0.7236 ± 0.0128 | 0.8966 ± 0.0120 | 0.4200 ± 0.0154 |
| Weights of Evidence | 0.4768 ± 0.0007 | 0.8776 ± 0.0001 | 0.3861 ± 0.0003 |

### Ablation Study (fold 0)

| Variant | AUC-PR | Drop |
|---|---|---|
| Full model | 0.9983 | — |
| w/o cross-modal attention | 0.9624 | −0.0359 |
| w/o CCA fusion | 0.9711 | −0.0272 |
| Geo branch only | 0.9945 | −0.0038 |
| Aero branch only | 0.9962 | −0.0021 |
| w/o MIM pre-training | 0.7236 | −0.2747 |

---

## Repository Structure

```
mineral_mapping/
├── src/
│   ├── pretrain/
│   │   ├── model.py              # ViT encoder + MAE decoder
│   │   └── train_pretrain.py     # MIM pre-training loop
│   ├── finetune/
│   │   ├── model_fusion.py       # Full fusion model
│   │   ├── dataset.py            # ProspectivityDataset
│   │   ├── train_finetune.py     # Fine-tuning loop
│   │   └── predict_map.py        # Full-area inference
│   └── baselines/
│       ├── woe.py                # Weights of Evidence
│       └── random_forest.py      # Random Forest baseline
├── experiments/
│   ├── pretrain_mim/             # Pre-training checkpoints
│   ├── finetune/                 # Fine-tuning checkpoints (fold 0-4)
│   ├── ablation_studies/
│   │   ├── A1_no_cross_attn/metrics.json
│   │   ├── A2_no_cca/metrics.json
│   │   ├── A3_geo_only/metrics.json
│   │   ├── A4_aero_only/metrics.json
│   │   └── summary.json
│   └── comparison_with_baselines/
├── results/
│   ├── prospectivity_maps/       # GeoTIFF output maps
│   └── figures/                  # Paper figures (PDF/PNG)
├── data/                         # Not tracked (see Data section)
│   ├── processed/stacked/        # 15-channel multiband_input.tif
│   └── labels/                   # pu_labels.tif
├── run_ablations.py              # Ablation study launcher
├── activate.sh                   # Environment activation
└── README.md
```

---

## Installation

### Requirements
- Python 3.10
- CUDA 11.8+
- 4× NVIDIA Tesla T4 (16 GB) or equivalent

```bash
# Clone repository
git clone https://github.com/sharonchrista/mineral_mapping.git
cd mineral_mapping

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy rasterio scikit-learn matplotlib scipy captum
```

---

## Data

The following input data are required but not included in this repository due to file size:

| File | Size | Description |
|---|---|---|
| `data/processed/stacked/multiband_input.tif` | 852 MB | 15-channel raster stack (3792×4461 px, 50 m) |
| `data/labels/pu_labels.tif` | ~50 MB | PU labels (417 occurrences, 500 m buffer) |

### Input Channels

| Ch | Name | Source | Type |
|---|---|---|---|
| 0 | Lithology 1:25K | GSI 25K | Categorical |
| 1 | Fault distance 25K | GSI 25K | EDT (m) |
| 2 | Fold distance 25K | GSI 25K | EDT (m) |
| 3 | Shear zone dist. 25K | GSI 25K | EDT (m) |
| 4 | Dyke distance 25K | GSI 25K | EDT (m) |
| 5 | Lithology 1:50K | GSI 50K | Categorical |
| 6 | Fault distance 50K | GSI 50K | EDT (m) |
| 7 | Fold distance 50K | GSI 50K | EDT (m) |
| 8 | TMI | Aeromagnetic | Normalized [0,1] |
| 9 | TMI 1VD | Aeromagnetic | Normalized [0,1] |
| 10 | TMI ASA | Aeromagnetic | Normalized [0,1] |
| 11 | TMI TDR | Aeromagnetic | Normalized [0,1] |
| 12 | Fault density (KDE) | Derived | KDE (km⁻²) |
| 13 | Structural complexity | Derived | SCI |
| 14 | Contact density (KDE) | Derived | KDE (km⁻²) |

---

## Usage

### Stage A — MIM Pre-training

```bash
source activate.sh

python src/pretrain/train_pretrain.py \
  --tif_path data/processed/stacked/multiband_input.tif \
  --output_dir experiments/pretrain_mim \
  --epochs 300 \
  --mask_ratio 0.75 \
  --batch_size 256 \
  --lr 1.5e-4 \
  --warmup_epochs 40

# Best val MSE: 0.0054 at epoch 288
```

### Stage B — Fine-tuning (5-fold spatial CV)

```bash
for fold in 0 1 2 3 4; do
  python src/finetune/train_finetune.py \
    --tif_path data/processed/stacked/multiband_input.tif \
    --label_path data/labels/pu_labels.tif \
    --pretrain_ckpt experiments/pretrain_mim/checkpoint_best.pt \
    --output_dir experiments/finetune/fold${fold} \
    --fold ${fold} \
    --epochs 100 \
    --freeze_epochs 10
done
```

### Stage C — Full-area Prospectivity Map

```bash
python src/finetune/predict_map.py \
  --tif_path data/processed/stacked/multiband_input.tif \
  --ckpt_dir experiments/finetune \
  --output_dir results/prospectivity_maps \
  --mc_passes 50
```

### Ablation Study

```bash
# Run all 4 ablations sequentially on single GPU
source activate.sh

CUDA_VISIBLE_DEVICES=0 python run_ablations.py \
  --variant A1_no_cross_attn --epochs 15 && \
CUDA_VISIBLE_DEVICES=0 python run_ablations.py \
  --variant A2_no_cca --epochs 15

# Results saved to experiments/ablation_studies/summary.json
```

---

## Model Architecture

```
Stage A — Self-Supervised MIM Pre-training
  Input: 15-channel raster (3792×4461 px)
  → Patch tiling (256×256 px, ~8,000 tiles)
  → Random masking (75%)
  → ViT-Base Encoder (12L, 768d, 12H, patch 16px, 86M params)
  → MIM Decoder (4L, 512d)
  → MSE loss on masked patches
  Best val MSE: 0.0054 (epoch 288)

Stage B — Multi-Modal Fusion Fine-tuning
  Geological Branch  : Ch 0-7  → ViT-Base (pretrained) → CLS 768d
  Geophysical Branch : Ch 8-11 → ViT-Base (pretrained) → CLS 768d
  Feature Branch     : Ch 12-14 → 2-layer MLP → 128d
  Cross-Modal Attention: geo ↔ aero (8 heads)
  Deep CCA Fusion    : [geo_proj(256d) ; aero_proj(256d)] → MLP → 512d
  Combined           : [fused(512d) ; features(128d)] → 640d

Stage C — Depth-Stratified Prediction
  Head 0-500m   : 2-layer MLP + MC Dropout (p=0.2)
  Head 500-1000m: 2-layer MLP + MC Dropout (p=0.2)
  Head 1000-2000m: 2-layer MLP + MC Dropout (p=0.2)
  Uncertainty: T=50 MC passes → mean + std maps
```

---

## Hardware

All experiments run on:
- **Server:** tyrone-hpc
- **GPUs:** 4× NVIDIA Tesla T4 (16 GB VRAM each)
- **Pre-training:** ~72 hours (300 epochs, 4 GPUs)
- **Fine-tuning:** ~10 hours per fold (5 folds)
- **Ablation:** ~15 hours per variant (single GPU)

---


---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Sharon Christa**
Department of Computer Science and Engineering
School of Computing, MIT Art Design and Technology University
Loni Kalbhor, Pune 412201, Maharashtra, India
✉ sharon.christa@mituniversity.edu.in