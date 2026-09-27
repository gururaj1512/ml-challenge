#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y build-essential git unzip tmux
if ! command -v conda >/dev/null && [ -x /opt/conda/bin/conda ]; then
  export PATH=/opt/conda/bin:$PATH
fi
if ! command -v conda >/dev/null; then
  echo 'Install/activate Miniforge as documented in README, then rerun.' >&2
  exit 1
fi
if conda run -n erfinal python -c 'import sys' >/dev/null 2>&1; then
  echo 'Using existing erfinal environment; validating native libraries below.'
else
  conda env create -f environment.yml
fi
# Use conda run so this script does not rely on interactive activation state.
conda run -n erfinal python -m pip install -r requirements.txt
conda run -n erfinal python -m pip install -e .
conda run -n erfinal python -m pip freeze --exclude-editable > environment.lock.txt
conda list -n erfinal --explicit > conda-explicit.txt
conda run -n erfinal python -m unittest discover -s tests -v
conda run -n erfinal python scripts/smoke_native.py
conda run -n erfinal python scripts/smoke_pipeline.py
conda run -n erfinal python -m erfast.cli preflight
