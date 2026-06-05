#!/usr/bin/env bash
# Composite KeySync 512x512 inference outputs back into the original
# 25fps frames so the final clip keeps the source resolution.
#
# Inputs (matching the layout produced by preprocess_crop.sh):
#   <work_dir>/videos_25fps/   : the 25 fps clips that landmarks were extracted from
#   <work_dir>/landmarks/      : .npy files, one per video (same basename)
#   <inference_dir>            : KeySync inference output mp4s. File naming
#                                follows scripts/sampling/dubbing_pipeline_raw.py:
#                                "<video_basename>_<audio_basename>.mp4"
#
# Output:
#   <work_dir>/recomposed/<inference_basename>.mp4 (original resolution)
#
# Usage:
#   bash slurm_scripts/recompose.sh <work_dir> <inference_dir>

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <work_dir> <inference_dir>" >&2
    exit 1
fi

WORK_DIR="$1"
INFERENCE_DIR="$2"

ENV_NAME="${ENV_NAME:-KeySync}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO_25FPS_DIR="$WORK_DIR/videos_25fps"
LANDMARKS_DIR="$WORK_DIR/landmarks"
OUT_DIR="$WORK_DIR/recomposed"
mkdir -p "$OUT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    module load mamba/24.3.0
    # shellcheck disable=SC1091
    source /hpc/software/mamba/24.3.0/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
fi

# ~/.bashrc sets HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed,
# which breaks HuggingFace downloads. Force the standard downloader.
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$REPO_ROOT"

shopt -s nullglob
matched=0
for inf in "$INFERENCE_DIR"/*.mp4; do
    base="$(basename "$inf" .mp4)"
    # Match the longest video basename that prefixes the inference filename
    # (dubbing_pipeline_raw saves as "<video>_<audio>.mp4").
    video=""
    for cand in "$VIDEO_25FPS_DIR"/*.mp4; do
        cand_base="$(basename "$cand" .mp4)"
        if [[ "$base" == "${cand_base}"_* ]]; then
            if [[ -z "$video" || ${#cand_base} -gt ${#video_base} ]]; then
                video="$cand"
                video_base="$cand_base"
            fi
        fi
    done
    if [[ -z "$video" ]]; then
        echo "skip: $inf  (no matching video in $VIDEO_25FPS_DIR)"
        continue
    fi
    landmarks="$LANDMARKS_DIR/${video_base}.npy"
    if [[ ! -f "$landmarks" ]]; then
        echo "skip: $inf  (missing landmarks $landmarks)"
        continue
    fi

    out="$OUT_DIR/${base}.mp4"
    if [[ -f "$out" ]]; then
        echo "exists: $out  (delete to regenerate)"
        continue
    fi

    echo "recompose: $base  (orig=$video)"
    python scripts/util/uncrop_video.py \
        --video "$video" \
        --landmarks "$landmarks" \
        --inference "$inf" \
        --output "$out"
    matched=$((matched + 1))
done

if (( matched == 0 )); then
    echo "Nothing to do." >&2
    exit 1
fi
echo "Recomposed $matched clip(s) into $OUT_DIR"
