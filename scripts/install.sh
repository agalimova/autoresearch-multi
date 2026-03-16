#!/usr/bin/env bash
set -euo pipefail

REPO="agalimova/autoresearch-multi"
INSTALL_DIR="${AUTORESEARCH_DIR:-$HOME/autoresearch-multi}"

echo "autoresearch-multi installer"
echo "============================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "Error: Python 3.8+ required. Install from https://python.org"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
echo "Python: $($PYTHON --version)"

# Check pip
if ! $PYTHON -m pip --version &>/dev/null; then
    echo "Error: pip not found. Run: $PYTHON -m ensurepip"
    exit 1
fi

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --ff-only 2>/dev/null || echo "  (not a git repo, skipping update)"
else
    echo "Cloning to $INSTALL_DIR..."
    git clone "https://github.com/$REPO.git" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
$PYTHON -m pip install -r requirements.txt -q

# Verify
echo ""
echo "Verifying installation..."
$PYTHON -c "
import sklearn, pandas, numpy, torch
print(f'  sklearn {sklearn.__version__}')
print(f'  torch {torch.__version__}')
print(f'  numpy {numpy.__version__}')
try:
    import optuna
    print(f'  optuna {optuna.__version__}')
except ImportError:
    pass
"

# Detect hardware
echo ""
$PYTHON -c "
from engine.hardware import detect, suggest_config
hw = detect()
print(f'Hardware: {hw}')
cfg = suggest_config(hw)
print(f'Suggested: batch_size={cfg[\"batch_size\"]}, hidden={cfg[\"hidden_size\"]}')
"

echo ""
echo "Done. Run:"
echo ""
echo "  cd $INSTALL_DIR"
echo "  python demo.py --runs 4          # hyperparameter search (MNIST)"
echo "  python demo_slots.py             # combinatorial slot testing (sklearn)"
echo "  python demo_slots_nn.py          # neural net architecture search"
echo ""
