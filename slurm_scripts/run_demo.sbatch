#!/usr/bin/env bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=keysync-demo
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Runs the end-to-end KeySync pipeline on the three demo clips under
# assets/ (symlinked into data/{videos,audios}). Produces final clips
# at the original 1080x1920 resolution in $WORK_DIR/recomposed/.
#
# Submit with:
#   mkdir -p slurm_logs
#   sbatch slurm_scripts/run_demo.sbatch
#
# Override --partition / --account if needed before submitting.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/pretrained_models/checkpoints}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/demo_run}"

# Inputs (symlinks created when this was set up).
RAW_VIDEO_DIR="$REPO_ROOT/data/videos"
RAW_AUDIO_DIR="$REPO_ROOT/data/audios"

mkdir -p "$WORK_DIR"

# The longest demo audio is ~20s, so 20 covers all three.
COMPUTE_UNTIL="${COMPUTE_UNTIL:-20}"

echo "node:        $(hostname)"
echo "gpu:         ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "repo:        $REPO_ROOT"
echo "ckpt_dir:    $CKPT_DIR"
echo "work_dir:    $WORK_DIR"
echo

bash "$REPO_ROOT/slurm_scripts/run_full_pipeline.sh" \
    "$RAW_VIDEO_DIR" "$RAW_AUDIO_DIR" "$WORK_DIR" \
    "$CKPT_DIR/keyframe_dub.pt" \
    "$CKPT_DIR/interpolation_dub.pt" \
    "$COMPUTE_UNTIL"
