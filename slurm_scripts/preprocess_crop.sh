#!/usr/bin/env bash
# Preprocess raw videos into face-cropped 512x512 25fps clips for KeySync.
#
# Pipeline (uses the helpers under scripts/util/):
#   1. ffmpeg_converter.py  : raw videos -> 25 fps; raw audios -> 16 kHz
#   2. gen_landmarks.py     : extract 68-pt facial landmarks (needs a GPU)
#   3. crop_video.py        : crop and resize around the face using landmarks
#
# Usage:
#   bash slurm_scripts/preprocess_crop.sh <raw_video_dir> <raw_audio_dir> <work_dir>
#
# work_dir layout produced:
#   work_dir/videos_25fps/   <- standardised inputs
#   work_dir/audios_16k/
#   work_dir/landmarks/
#   work_dir/videos/         <- final cropped 512x512 25fps videos (feed these to inference)
#   work_dir/landmarks_cropped/
#
# To run as a SLURM job, wrap with sbatch — example header (replace partition):
#   #SBATCH --partition=<gpu-partition>
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=8
#   #SBATCH --mem=32G
#   #SBATCH --time=02:00:00
#   #SBATCH --output=preprocess_%j.log

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <raw_video_dir> <raw_audio_dir> <work_dir>" >&2
    exit 1
fi

RAW_VIDEO_DIR="$1"
RAW_AUDIO_DIR="$2"
WORK_DIR="$3"

ENV_NAME="${ENV_NAME:-KeySync}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO_25FPS_DIR="$WORK_DIR/videos_25fps"
AUDIO_16K_DIR="$WORK_DIR/audios_16k"
LANDMARKS_DIR="$WORK_DIR/landmarks"
CROPPED_VIDEO_DIR="$WORK_DIR/videos"
CROPPED_LANDMARKS_DIR="$WORK_DIR/landmarks_cropped"

mkdir -p "$VIDEO_25FPS_DIR" "$AUDIO_16K_DIR" "$LANDMARKS_DIR" \
         "$CROPPED_VIDEO_DIR" "$CROPPED_LANDMARKS_DIR"

# Activate the conda env only if not already active. A redundant
# `module load mamba` inside a sub-shell prepends mamba/bin ahead of
# the env's bin and `conda activate` is then a no-op, so `python`
# silently resolves to base mamba.
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

echo "[1/3] ffmpeg: standardising video to 25fps and audio to 16kHz"
python scripts/util/ffmpeg_converter.py \
    --video_dir "$RAW_VIDEO_DIR" \
    --video_dir_25fps "$VIDEO_25FPS_DIR" \
    --audio_dir "$RAW_AUDIO_DIR" \
    --audio_dir_16k "$AUDIO_16K_DIR"

echo "[2/3] gen_landmarks: extracting 68-point facial landmarks (GPU)"
# gen_landmarks.py imports landmarks_extractor as a sibling module,
# so it must be invoked with its own directory on sys.path.
( cd scripts/util && python gen_landmarks.py "$VIDEO_25FPS_DIR" \
        --output_dir "$LANDMARKS_DIR" )

echo "[3/3] crop_video: face-cropping to 512x512"
python scripts/util/crop_video.py \
    --video_dir "$VIDEO_25FPS_DIR" \
    --video_dir_cropped "$CROPPED_VIDEO_DIR" \
    --landmarks_dir "$LANDMARKS_DIR" \
    --landmarks_dir_cropped "$CROPPED_LANDMARKS_DIR"

echo
echo "Done. Cropped clips: $CROPPED_VIDEO_DIR"
echo "Pass to inference with --file_list $CROPPED_VIDEO_DIR --file_list_audio $AUDIO_16K_DIR"
