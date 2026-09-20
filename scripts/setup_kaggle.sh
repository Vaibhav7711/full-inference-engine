#!/usr/bin/env bash
set -euo pipefail

# Kaggle GPU notebooks ship a CUDA-matched PyTorch build, exactly like Colab.
# Reuse requirements-colab.txt so torch is never reinstalled/mismatched.
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
python -m pip install -e .
