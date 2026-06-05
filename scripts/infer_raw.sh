#!/bin/bash

# Get command line arguments with defaults
# Default values
DEFAULT_FILE_LIST="filelist_val.txt"
DEFAULT_KEYFRAMES_CKPT="None"
DEFAULT_INTERPOLATION_CKPT="None"
DEFAULT_COMPUTE_UNTIL="45"
DEFAULT_FILE_LIST_AUDIO="None"
DEFAULT_FIX_OCCLUSION="False"
DEFAULT_POSITION="None"
DEFAULT_START_FRAME="0"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_folder)
            output_folder="$2"
            shift 2
            ;;
        --file_list)
            file_list="${2:-$DEFAULT_FILE_LIST}"
            shift 2
            ;;
        --keyframes_ckpt)
            keyframes_ckpt="${2:-$DEFAULT_KEYFRAMES_CKPT}"
            shift 2
            ;;
        --interpolation_ckpt)
            interpolation_ckpt="${2:-$DEFAULT_INTERPOLATION_CKPT}"
            shift 2
            ;;
        --compute_until)
            compute_until="${2:-$DEFAULT_COMPUTE_UNTIL}"
            shift 2
            ;;
        --file_list_audio)
            file_list_audio="${2:-$DEFAULT_FILE_LIST_AUDIO}"
            shift 2
            ;;
        --fix_occlusion)
            fix_occlusion="${2:-$DEFAULT_FIX_OCCLUSION}"
            shift 2
            ;;
        --position)
            position="${2:-$DEFAULT_POSITION}"
            shift 2
            ;;
        --start_frame)
            start_frame="${2:-$DEFAULT_START_FRAME}"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set defaults if not provided
file_list=${file_list:-$DEFAULT_FILE_LIST}
keyframes_ckpt=${keyframes_ckpt:-$DEFAULT_KEYFRAMES_CKPT}
interpolation_ckpt=${interpolation_ckpt:-$DEFAULT_INTERPOLATION_CKPT}
compute_until=${compute_until:-$DEFAULT_COMPUTE_UNTIL}
file_list_audio=${file_list_audio:-$DEFAULT_FILE_LIST_AUDIO}
fix_occlusion=${fix_occlusion:-$DEFAULT_FIX_OCCLUSION}
position=${position:-$DEFAULT_POSITION}
start_frame=${start_frame:-$DEFAULT_START_FRAME}

# Check if output_folder is provided
if [ -z "$output_folder" ]; then
    echo "Error: --output_folder is required"
    exit 1
fi

# Upstream prepends "outputs/" assuming a relative folder name. Our pipeline
# passes an absolute path (outputs/demo_run/inference), which would produce a
# broken "outputs//gpfs/..." path. Only add the prefix for relative paths.
if [[ "$output_folder" = /* ]]; then
    output_arg="$output_folder"
else
    output_arg="outputs/$output_folder"
fi

# Run the Python script with the appropriate arguments
python scripts/sampling/dubbing_pipeline_raw.py \
    --filelist=${file_list} \
    --filelist_audio=${file_list_audio} \
    --decoding_t 1 \
    --cond_aug 0. \
    --resize_size=512 \
    --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
    --latent_folder=videos \
    --video_folder=videos \
    --model_config=scripts/sampling/configs/interpolation.yaml \
    --model_keyframes_config=scripts/sampling/configs/keyframe.yaml \
    --chunk_size=2 \
    --landmark_folder=landmarks \
    --audio_folder=audios \
    --audio_emb_folder=audios \
    --output_folder=${output_arg} \
    --keyframes_ckpt=${keyframes_ckpt} \
    --interpolation_ckpt=${interpolation_ckpt} \
    --add_zero_flag=True \
    --extra_audio=None \
    --compute_until=${compute_until} \
    --audio_emb_type=hubert \
    --recompute=True \
    --fix_occlusion=${fix_occlusion} \
    --position=${position} \
    --start_frame=${start_frame}
