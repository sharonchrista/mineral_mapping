#!/bin/bash
# =============================================================================
# Mineral Prospectivity Mapping — Setup (no sudo)
# Builds ALL missing C libs locally, then compiles Python 3.10 once cleanly
# Missing libs fixed: libffi (_ctypes), sqlite3 (_sqlite3), libbz2, liblzma
# Run: bash setup_env.sh
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_DIR/venv"
PYENV_DIR="$HOME/.pyenv"
LOCAL_DIR="$HOME/.local_libs"      # all locally-built C libs go here
BUILD_DIR="$HOME/.build_tmp"
PYTHON_VERSION="3.10.13"

echo "============================================================"
echo " Mineral Prospectivity Mapping — Setup (no sudo)"
echo " Local libs  : $LOCAL_DIR"
echo " Python      : $PYTHON_VERSION"
echo " Venv        : $ENV_DIR"
echo "============================================================"

mkdir -p "$LOCAL_DIR" "$BUILD_DIR"

# ── Export flags so every build below sees local libs ────────────────────────
export PATH="$LOCAL_DIR/bin:${PATH}"
export PKG_CONFIG_PATH="$LOCAL_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$LOCAL_DIR/lib:${LD_LIBRARY_PATH:-}"
export CFLAGS="-I$LOCAL_DIR/include"
export LDFLAGS="-L$LOCAL_DIR/lib -Wl,-rpath,$LOCAL_DIR/lib"
export CPPFLAGS="-I$LOCAL_DIR/include"

# ── Generic build helper ─────────────────────────────────────────────────────
build_lib() {
    local NAME="$1" URL="$2" DIR="$3" CHECK="$4"
    if [ -f "$LOCAL_DIR/$CHECK" ]; then
        echo "      $NAME already built — skipping"
        return
    fi
    echo "      Building $NAME ..."
    cd "$BUILD_DIR"
    wget -q "$URL"
    tar xf "$(basename "$URL")"
    cd "$DIR"
    ./configure --prefix="$LOCAL_DIR" --quiet --disable-shared 2>/dev/null \
        || ./configure --prefix="$LOCAL_DIR" --quiet
    make -j"$(nproc)" --quiet
    make install --quiet
    cd "$BUILD_DIR"
    echo "      ✓ $NAME installed"
}

# ------------------------------------------------------------
# STEP 1 — Build all missing C libraries
# ------------------------------------------------------------
echo ""
echo "[1/8] Building missing C libraries into $LOCAL_DIR ..."

# libffi  → fixes _ctypes
build_lib "libffi 3.4.4" \
    "https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz" \
    "libffi-3.4.4" \
    "lib/libffi.a"

# SQLite  → fixes _sqlite3
build_lib "SQLite 3.44.2" \
    "https://www.sqlite.org/2023/sqlite-autoconf-3440200.tar.gz" \
    "sqlite-autoconf-3440200" \
    "lib/libsqlite3.a"

# libbz2  → fixes _bz2 (needed by many packages)
cd "$BUILD_DIR"
if [ ! -f "$LOCAL_DIR/lib/libbz2.a" ]; then
    echo "      Building libbz2 1.0.8 ..."
    wget -q "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"
    tar xf bzip2-1.0.8.tar.gz
    cd bzip2-1.0.8
    make -j"$(nproc)" --quiet
    make install PREFIX="$LOCAL_DIR" --quiet
    cd "$BUILD_DIR"
    echo "      ✓ libbz2 installed"
else
    echo "      libbz2 already built — skipping"
fi

# liblzma → fixes _lzma (needed by pandas, etc.)
build_lib "xz/liblzma 5.4.5" \
    "https://github.com/tukaani-project/xz/releases/download/v5.4.5/xz-5.4.5.tar.gz" \
    "xz-5.4.5" \
    "lib/liblzma.a"

# libreadline → nicer REPL (optional but good)
build_lib "ncurses 6.4" \
    "https://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.4.tar.gz" \
    "ncurses-6.4" \
    "lib/libncurses.a"

build_lib "readline 8.2" \
    "https://ftp.gnu.org/gnu/readline/readline-8.2.tar.gz" \
    "readline-8.2" \
    "lib/libreadline.a"

echo "      ✓ All C libraries ready"

# ------------------------------------------------------------
# STEP 2 — Install / update pyenv
# ------------------------------------------------------------
echo ""
echo "[2/8] Setting up pyenv ..."

if [ ! -d "$PYENV_DIR" ]; then
    curl -fsSL https://pyenv.run | bash
fi

export PYENV_ROOT="$PYENV_DIR"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

if ! grep -q 'pyenv init' "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<'BASHRC'

# >>> pyenv >>>
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
# <<< pyenv <<<
BASHRC
fi
echo "      ✓ pyenv ready"

# ------------------------------------------------------------
# STEP 3 — Recompile Python 3.10 with ALL local libs
# ------------------------------------------------------------
echo ""
echo "[3/8] Compiling Python $PYTHON_VERSION (final — all libs present) ..."
echo "      This takes 5-10 min ..."

# Always remove previous broken build
if pyenv versions --bare 2>/dev/null | grep -q "^${PYTHON_VERSION}$"; then
    echo "      Removing previous broken build ..."
    pyenv uninstall -f "$PYTHON_VERSION"
fi

CFLAGS="-I$LOCAL_DIR/include" \
CPPFLAGS="-I$LOCAL_DIR/include" \
LDFLAGS="-L$LOCAL_DIR/lib -Wl,-rpath,$LOCAL_DIR/lib" \
PKG_CONFIG_PATH="$LOCAL_DIR/lib/pkgconfig" \
CONFIGURE_OPTS="--enable-optimizations --with-ensurepip=install" \
    pyenv install "$PYTHON_VERSION"

echo "      ✓ Python $PYTHON_VERSION compiled"

PYTHON_BIN="$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python3"

# Verify all critical modules
echo "      Verifying critical modules ..."
"$PYTHON_BIN" -c "
import ctypes;  print('      ✓ ctypes   OK')
import sqlite3; print('      ✓ sqlite3  OK')
import bz2;     print('      ✓ bz2      OK')
import lzma;    print('      ✓ lzma     OK')
import ssl;     print('      ✓ ssl      OK')
"

# ------------------------------------------------------------
# STEP 4 — Create venv
# ------------------------------------------------------------
echo ""
echo "[4/8] Creating virtual environment ..."

[ -d "$ENV_DIR" ] && rm -rf "$ENV_DIR"
"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
echo "      ✓ venv ready  ($(python --version))"

# ------------------------------------------------------------
# STEP 5 — PyTorch (CUDA 13.0 → cu121)
# ------------------------------------------------------------
echo ""
echo "[5/8] Installing PyTorch 2.1.2 (cu121 for CUDA 13.x) ..."

pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

python - <<'EOF'
import torch
print(f"      torch      : {torch.__version__}")
print(f"      CUDA avail : {torch.cuda.is_available()}")
print(f"      GPU count  : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"      GPU {i}      : {torch.cuda.get_device_name(i)}")
EOF

# ------------------------------------------------------------
# STEP 6 — Geospatial
# ------------------------------------------------------------
echo ""
echo "[6/8] Installing geospatial libraries ..."

GDAL_VERSION=$(gdal-config --version 2>/dev/null || echo "")
if [ -n "$GDAL_VERSION" ]; then
    echo "      System GDAL: $GDAL_VERSION"
    pip install "GDAL==${GDAL_VERSION}" --quiet
fi

pip install \
    rasterio==1.3.9 \
    geopandas==0.14.1 \
    shapely==2.0.2 \
    pyproj==3.6.1 \
    fiona==1.9.5 \
    rtree==1.1.0 \
    pyogrio==0.7.2 \
    xarray==2023.12.0 \
    rioxarray==0.15.0 \
    earthpy==0.9.4 \
    contextily==1.5.0 \
    --quiet

echo "      ✓ Geospatial stack installed"

# ------------------------------------------------------------
# STEP 7 — All remaining libraries
# ------------------------------------------------------------
echo ""
echo "[7/8] Installing DL, ML, geophysics, XAI & utilities ..."

pip install \
    pytorch-lightning==2.1.3 \
    timm==0.9.12 \
    transformers==4.36.2 \
    einops==0.7.0 \
    torchmetrics==1.2.1 \
    scikit-learn==1.3.2 \
    numpy==1.26.2 \
    pandas==2.1.4 \
    scipy==1.11.4 \
    imbalanced-learn==0.11.0 \
    albumentations==1.3.1 \
    wandb==0.16.1 \
    mlflow==2.9.2 \
    tensorboard==2.15.1 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    tqdm==4.66.1 \
    rich==13.7.0 \
    SimPEG==0.20.0 \
    discretize==0.10.0 \
    pymatsolver==0.2.0 \
    verde==1.8.0 \
    harmonica==0.6.0 \
    boule==0.4.1 \
    empymod==2.3.0 \
    shap==0.44.0 \
    captum==0.7.0 \
    lime==0.2.0.1 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.18.0 \
    folium==0.15.1 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    h5py==3.10.0 \
    netCDF4==1.6.5 \
    zarr==2.16.1 \
    "dask[complete]==2023.12.1" \
    joblib==1.3.2 \
    pyarrow==14.0.1 \
    streamlit==1.29.0 \
    fastapi==0.108.0 \
    uvicorn==0.25.0 \
    "pydantic==2.5.3" \
    pytest==7.4.3 \
    black==23.12.1 \
    isort==5.13.2 \
    flake8==7.0.0 \
    --quiet

echo "      ✓ All libraries installed"

# ------------------------------------------------------------
# STEP 8 — JupyterLab
# ------------------------------------------------------------
echo ""
echo "[8/8] Installing JupyterLab ..."

pip install jupyter==1.0.0 jupyterlab==4.0.9 ipywidgets==8.1.1 ipykernel --quiet
python -m ipykernel install --user --name "mineral_mapping" --display-name "Python (mineral_mapping)"
echo "      ✓ JupyterLab ready"

# ------------------------------------------------------------
# activate.sh
# ------------------------------------------------------------
cat > "$PROJECT_DIR/activate.sh" <<ACTIVATE
#!/bin/bash
# source activate.sh
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
export LD_LIBRARY_PATH="$LOCAL_DIR/lib:\${LD_LIBRARY_PATH:-}"
eval "\$(pyenv init -)"
source "$ENV_DIR/bin/activate"
echo "✓ mineral_mapping env active  (Python: \$(python --version))"
ACTIVATE
chmod +x "$PROJECT_DIR/activate.sh"

echo ""
echo "============================================================"
echo " ✅  Setup complete!"
echo ""
echo " Every new session:"
echo "   source ~/mineral_mapping/activate.sh"
echo ""
echo " Verify packages + GPU:"
echo "   python verify_env.py"
echo "============================================================"