#!/bin/bash
# source activate.sh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="/home/sharon/.local_libs/lib:${LD_LIBRARY_PATH:-}"
eval "$(pyenv init -)"
source "/home/sharon/sharon/mineral_mapping/venv/bin/activate"
echo "✓ mineral_mapping env active  (Python: $(python --version))"
