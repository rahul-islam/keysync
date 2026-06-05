#!/usr/bin/env bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=keysync-recompose
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Composite the 512x512 KeySync inference outputs back into the original
# full-resolution frames. CPU-only but memory-hungry (loads full-res 1080x1920
# frame stacks), so it gets killed on the login node — run it here with 120G.
#
# Submit from the repo root:
#   sbatch slurm_scripts/run_recompose.sbatch <work_dir> <inference_dir>
# Defaults target the demo_run.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

WORK_DIR="${1:-$REPO_ROOT/outputs/demo_run}"
INFERENCE_DIR="${2:-$REPO_ROOT/outputs/gpfs/home/qhu3808/keysync/outputs/demo_run/inference}"

echo "node:          $(hostname)"
echo "work_dir:      $WORK_DIR"
echo "inference_dir: $INFERENCE_DIR"
echo

bash "$REPO_ROOT/slurm_scripts/recompose.sh" "$WORK_DIR" "$INFERENCE_DIR"
