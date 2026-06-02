#!/usr/bin/env bash
# Reproduce the KeySync conda env on this HPC cluster.
#
# Differences from the upstream README:
#   - dlib must come from conda-forge (the system gcc is too old to build
#     dlib 19.24.6 from source via pip).
#   - mamba is provided as a module (mamba/24.3.0) rather than installed
#     to $HOME.
#
# Usage:
#   bash slurm_scripts/setup_cluster.sh
#
# Re-running is safe: mamba create errors out if the env already exists,
# stopping the script before any reinstall.

set -euo pipefail

ENV_NAME="${ENV_NAME:-KeySync}"

module load mamba/24.3.0
# shellcheck disable=SC1091
source /hpc/software/mamba/24.3.0/etc/profile.d/conda.sh

mamba create -n "$ENV_NAME" python=3.11 conda-forge::ffmpeg -y
conda activate "$ENV_NAME"

# Prebuilt dlib (replaces the pip dlib==19.24.6 line in requirements.txt).
mamba install -c conda-forge "dlib=19.24" -y

# Install everything else from requirements.txt, skipping dlib.
REQ_TMP="$(mktemp --suffix=.txt)"
trap 'rm -f "$REQ_TMP"' EXIT
grep -vE '^dlib==' requirements.txt > "$REQ_TMP"
python -m pip install -r "$REQ_TMP" --no-deps

# PyTorch with CUDA 12.1 (matches the cuda/cuda-12.1.0 module on this cluster).
python -m pip install \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo
echo "Env ready. Activate in a new shell with:"
echo "  module load mamba/24.3.0"
echo "  source /hpc/software/mamba/24.3.0/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
