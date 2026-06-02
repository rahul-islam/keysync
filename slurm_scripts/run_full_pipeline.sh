#!/usr/bin/env bash
# End-to-end KeySync pipeline:
#   raw videos + raw audios
#       -> ffmpeg standardise (25fps / 16kHz)
#       -> landmark extraction
#       -> face crop (512x512)
#       -> KeySync inference (keyframe + interpolation)
#       -> recompose back to original frame resolution
#
# Usage:
#   bash slurm_scripts/run_full_pipeline.sh \
#       <raw_video_dir> <raw_audio_dir> <work_dir> \
#       <keyframes_ckpt> <interpolation_ckpt> [compute_until]
#
# Produces:
#   <work_dir>/videos_25fps/      standardised inputs
#   <work_dir>/audios_16k/
#   <work_dir>/landmarks/
#   <work_dir>/videos/            face-cropped 512x512 (inference input)
#   <work_dir>/inference/         raw KeySync output (512x512)
#   <work_dir>/recomposed/        FINAL output at original resolution
#
# Needs a GPU (landmark extraction + inference).

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <raw_video_dir> <raw_audio_dir> <work_dir> <keyframes_ckpt> <interpolation_ckpt> [compute_until]" >&2
    exit 1
fi

RAW_VIDEO_DIR="$1"
RAW_AUDIO_DIR="$2"
WORK_DIR="$3"
KEYFRAMES_CKPT="$4"
INTERPOLATION_CKPT="$5"
COMPUTE_UNTIL="${6:-45}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFERENCE_DIR="$WORK_DIR/inference"

# 1 + 2 + 3: ffmpeg, landmarks, crop
bash "$REPO_ROOT/slurm_scripts/preprocess_crop.sh" \
    "$RAW_VIDEO_DIR" "$RAW_AUDIO_DIR" "$WORK_DIR"

# 4: inference. infer_raw.sh internally re-runs landmark + audio embedding
# extraction, but it does that on whatever you give it as --file_list, so
# pointing it at the already-cropped videos is what we want.
mkdir -p "$INFERENCE_DIR"
bash "$REPO_ROOT/scripts/infer_raw.sh" \
    --file_list "$WORK_DIR/videos" \
    --file_list_audio "$WORK_DIR/audios_16k" \
    --output_folder "$INFERENCE_DIR" \
    --keyframes_ckpt "$KEYFRAMES_CKPT" \
    --interpolation_ckpt "$INTERPOLATION_CKPT" \
    --compute_until "$COMPUTE_UNTIL"

# 5: paste 512x512 inference results back into the original frames.
bash "$REPO_ROOT/slurm_scripts/recompose.sh" "$WORK_DIR" "$INFERENCE_DIR"

echo
echo "Pipeline complete."
echo "Final clips (original resolution): $WORK_DIR/recomposed/"
