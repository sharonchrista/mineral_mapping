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
