import math
import os
import sys
from typing import Optional, Tuple, List, Union, Dict, Any
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io import read_video
from tqdm import tqdm

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from transformers import HubertModel  # noqa
from scripts.util.audio.WavLM import WavLM_wrapper  # noqa
from scripts.util.vae_wrapper import VaeWrapper  # noqa
from scripts.util.video_to_latent import encode_video_chunk  # noqa
from scripts.util.landmarks_extractor import LandmarksExtractor  # noqa

from sgm.util import (  # noqa
    default,
    instantiate_from_config,
    get_raw_audio,
    save_audio_video,
    calculate_splits,
    ensure_landmarks_shape,
)
from sgm.data.data_utils import (  # noqa
    create_masks_from_landmarks_full_size,
    create_face_mask_from_landmarks,
    create_masks_from_landmarks_box,
    create_masks_from_landmarks_mouth,
    draw_kps_image,
    scale_landmarks,
)
from sgm.data.mask import face_mask_cheeks_batch  # noqa

try:
    from sam2.build_sam import build_sam2_video_predictor  # noqa
except ImportError:
    print("SAM2 is not installed, occlusion handling won't work")


def get_segmentation_mask_arms(
    video_path: str,
    ann_frame_idx: int,
    position: List[float],
    video_len: int,
    video_size: Tuple[int, int],
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Generate a segmentation mask for arms using SAM2.

    Args:
        video_path: Path to the video file
        ann_frame_idx: Frame index for annotation
        position: Position to place the annotation point
        video_len: Length of the video in frames
        video_size: Size of the video frames (height, width)
        target_size: Target size for the output mask

    Returns:
        Segmentation mask for arms region
    """
    sam2_checkpoint = "pretrained_models/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
    inference_state = predictor.init_state(
        video_path=video_path, offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    ann_obj_id = (
        1  # give a unique id to each object we interact with (it can be any integers)
    )
    points = np.array([position], dtype=np.float32)
    print(points)
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    video_segments: Dict[
        int, Dict[int, np.ndarray]
    ] = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    mask = np.zeros((video_len, video_size[0], video_size[1]))
    for out_frame_idx in range(ann_frame_idx, video_len):
        mask[out_frame_idx, ...] = video_segments[out_frame_idx][ann_obj_id]

    # Interpolate mask to target size
    mask = torch.from_numpy(mask).float()
    mask = F.interpolate(mask.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)
    mask = mask.numpy()

    return mask


def load_landmarks(
    landmarks: np.ndarray,
    original_size: Tuple[int, int],
    index: int,
    target_size: Tuple[int, int] = (64, 64),
    is_dub: bool = True,
    what_mask: str = "box",
    nose_index: int = 28,
) -> torch.Tensor:
    """
    Load and process facial landmarks to create masks.

    Args:
        landmarks: Facial landmarks array
        original_size: Original size of the video frames
        index: Index for non-dub mode
        target_size: Target size for the output mask
        is_dub: Whether this is for dubbing mode
        what_mask: Type of mask to create ("full", "box", "heart", "mouth")
        nose_index: Index of the nose landmark

    Returns:
        Processed landmarks mask
    """
    if is_dub:
        expand_box = 0.0
        if len(landmarks.shape) == 2:
            landmarks = landmarks[None, ...]
        if what_mask == "full":
            mask = create_masks_from_landmarks_full_size(
                landmarks,
                original_size[0],
                original_size[1],
                offset=expand_box,
                nose_index=nose_index,
            )
        elif what_mask == "box":
            mask = create_masks_from_landmarks_box(
                landmarks,
                (original_size[0], original_size[1]),
                box_expand=expand_box,
                nose_index=nose_index,
            )
        elif what_mask == "heart":
            mask = face_mask_cheeks_batch(
                original_size, landmarks, box_expand=expand_box, show_nose=True
            )
        elif what_mask == "mouth":
            mask = create_masks_from_landmarks_mouth(
                landmarks,
                (original_size[0], original_size[1]),
                box_expand=0.01,
                nose_index=nose_index,
            )
        else:
            mask = create_face_mask_from_landmarks(
                landmarks, original_size[0], original_size[1], mask_expand=0.05
            )
        mask = F.interpolate(
            mask.unsqueeze(1).float(), size=target_size, mode="nearest"
        )
        return mask
    else:
        landmarks = landmarks[index]
        land_image = draw_kps_image(
            target_size, original_size, landmarks, rgb=True, pts_width=1
        )
        return torch.from_numpy(land_image).float() / 255.0


def merge_overlapping_segments(segments: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Merges overlapping segments by averaging overlapping frames.
    Segments have shape (b, t, ...), where 'b' is the number of segments,
    't' is frames per segment, and '...' are other dimensions.

    Args:
        segments: Tensor of shape (b, t, ...)
        overlap: Integer, number of frames that overlap between consecutive segments

    Returns:
        Tensor of the merged video
    """
    # Get the shape details
    b, t, *other_dims = segments.shape
    num_frames = (b - 1) * (
        t - overlap
    ) + t  # Calculate the total number of frames in the merged video

    # Initialize the output tensor and a count tensor to keep track of contributions for averaging
    output_shape = [num_frames] + other_dims
    output = torch.zeros(output_shape, dtype=segments.dtype, device=segments.device)
    count = torch.zeros(output_shape, dtype=torch.float32, device=segments.device)

    current_index = 0
    for i in range(b):
        end_index = current_index + t
        # Add the segment to the output tensor
        output[current_index:end_index] += rearrange(segments[i], "... -> ...")
        # Increment the count tensor for each frame that's added
        count[current_index:end_index] += 1
        # Update the starting index for the next segment
        current_index += t - overlap

    # Avoid division by zero
    count[count == 0] = 1
    # Average the frames where there's overlap
    output /= count

    return output


def get_audio_indexes(main_index: int, n_audio_frames: int, max_len: int) -> List[int]:
    """
    Get indexes for audio frames around a main index.

    Args:
        main_index: Central index
        n_audio_frames: Number of audio frames to include
        max_len: Maximum length of the sequence

    Returns:
        List of audio indexes
    """
    # Get indexes for audio from both sides of the main index
    audio_ids = []
    # get audio embs from both sides of the GT frame
    audio_ids += [0] * max(n_audio_frames - main_index, 0)
    for i in range(
        max(main_index - n_audio_frames, 0),
        min(main_index + n_audio_frames + 1, max_len),
    ):
        audio_ids += [i]
    audio_ids += [max_len - 1] * max(main_index + n_audio_frames - max_len + 1, 0)
    return audio_ids


def create_pipeline_inputs(
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_interpolation: torch.Tensor,
    num_frames: int,
    video_emb: torch.Tensor,
    landmarks: np.ndarray,
    overlap: int = 1,
    add_zero_flag: bool = False,
    what_mask: str = "box",
    mask_arms: Optional[np.ndarray] = None,
    nose_index: int = 28,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    int,
    List[bool],
    List[List[int]],
    List[Optional[int]],
]:
    """
    Create inputs for the keyframe generation and interpolation pipeline.

    Args:
        video: Input video tensor
        audio: Audio embeddings for keyframe generation
        audio_interpolation: Audio embeddings for interpolation
        num_frames: Number of frames per segment
        video_emb: Optional video embeddings
        landmarks: Facial landmarks for mask generation
        overlap: Number of frames to overlap between segments
        add_zero_flag: Whether to add zero flag every num_frames
        what_mask: Type of mask to generate ("box" or other options)
        mask_arms: Optional mask for arms region
        nose_index: Index of the nose landmark point

    Returns:
        Tuple containing all necessary inputs for the pipeline
    """
    audio_interpolation_chunks = []
    audio_image_preds = []
    gt_chunks = []
    gt_keyframes_chunks = []
    # Adjustment for overlap to ensure segments are created properly
    step = num_frames - overlap

    # Ensure there's at least one step forward on each iteration
    if step < 1:
        step = 1

    audio_image_preds_idx = []
    audio_interp_preds_idx = []
    masks_chunks = []
    masks_interpolation_chunks = []
    for i in range(0, audio.shape[0] - num_frames + 1, step):
        try:
            audio[i + num_frames - 1]
        except IndexError:
            break  # Last chunk is smaller than num_frames
        segment_end = i + num_frames
        gt_chunks.append(video_emb[i:segment_end])
        masks = load_landmarks(
            landmarks[i:segment_end],
            video.shape[-2:],
            i,
            target_size=(64, 64),
            is_dub=True,
            what_mask=what_mask,
            nose_index=nose_index,
        )
        if mask_arms is not None:
            masks = np.logical_and(
                masks, np.logical_not(mask_arms[i:segment_end, None, ...])
            )
        masks_interpolation_chunks.append(masks)

        if i not in audio_image_preds_idx:
            audio_image_preds.append(audio[i])
            masks_chunks.append(masks[0])
            gt_keyframes_chunks.append(video_emb[i])
            audio_image_preds_idx.append(i)

        if segment_end - 1 not in audio_image_preds_idx:
            audio_image_preds_idx.append(segment_end - 1)

            audio_image_preds.append(audio[segment_end - 1])
            masks_chunks.append(masks[-1])
            gt_keyframes_chunks.append(video_emb[segment_end - 1])

        audio_interpolation_chunks.append(audio_interpolation[i:segment_end])
        audio_interp_preds_idx.append([i, segment_end - 1])

    # If the flag is on, add element 0 every 14 audio elements
    if add_zero_flag:
        first_element = audio_image_preds[0]

        len_audio_image_preds = (
            len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        )
        for i in range(0, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, first_element)
            audio_image_preds_idx.insert(i, None)
            masks_chunks.insert(i, masks_chunks[0])
            gt_keyframes_chunks.insert(i, gt_keyframes_chunks[0])

    to_remove = [idx is None for idx in audio_image_preds_idx]
    audio_image_preds_idx_clone = [idx for idx in audio_image_preds_idx]
    if add_zero_flag:
        # Remove the added elements from the list
        audio_image_preds_idx = [
            sample for i, sample in zip(to_remove, audio_image_preds_idx) if not i
        ]

    interpolation_cond_list = []
    for i in range(0, len(audio_image_preds_idx) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append(
            [audio_image_preds_idx[i], audio_image_preds_idx[i + 1]]
        )

    # Since we generate num_frames at a time, we need to ensure that the last chunk is of size num_frames
    # Calculate the number of frames needed to make audio_image_preds a multiple of num_frames
    frames_needed = (num_frames - (len(audio_image_preds) % num_frames)) % num_frames

    # Extend from the start of audio_image_preds
    audio_image_preds = audio_image_preds + [audio_image_preds[-1]] * frames_needed
    masks_chunks = masks_chunks + [masks_chunks[-1]] * frames_needed
    gt_keyframes_chunks = (
        gt_keyframes_chunks + [gt_keyframes_chunks[-1]] * frames_needed
    )

    to_remove = to_remove + [True] * frames_needed
    audio_image_preds_idx_clone = (
        audio_image_preds_idx_clone + [audio_image_preds_idx_clone[-1]] * frames_needed
    )

    print(
        f"Added {frames_needed} frames from the start to make audio_image_preds a multiple of {num_frames}"
    )

    # random_cond_idx = np.random.randint(0, len(video_emb))
    random_cond_idx = 0

    assert len(to_remove) == len(audio_image_preds), (
        "to_remove and audio_image_preds must have the same length"
    )

    return (
        gt_chunks,
        gt_keyframes_chunks,
        audio_interpolation_chunks,
        audio_image_preds,
        video_emb[random_cond_idx],
        video[random_cond_idx],
        masks_chunks,
        masks_interpolation_chunks,
        to_remove,
        audio_interp_preds_idx,
        audio_image_preds_idx_clone,
    )


@torch.inference_mode()
def compute_hubert_embeddings(
    raw_audio: torch.Tensor,
    hubert_model: HubertModel,
) -> torch.Tensor:
    print("Computing hubert embeddings")
    raw_audio = rearrange(raw_audio, "f s -> (f s)")
    audio = (
        (raw_audio - raw_audio.mean()) / torch.sqrt(raw_audio.var() + 1e-7)
    ).unsqueeze(0)
    chunks = 16000 * 20

    # Get audio embeddings
    audio_embeddings = []
    splits = list(calculate_splits(audio, chunks))

    for i, chunk in enumerate(tqdm(splits, desc="Computing hubert embeddings")):
        hidden_states = hubert_model(chunk.cuda())[0]
        audio_embeddings.append(hidden_states)
    audio_embeddings = torch.cat(audio_embeddings, dim=1)

    # audio_embeddings = self.model.wav2vec2(rearrange(audio_frames, "f s -> () (f s)"))[0]
    if audio_embeddings.shape[1] % 2 != 0:
        audio_embeddings = torch.cat(
            [audio_embeddings, torch.zeros_like(audio_embeddings[:, :1])], dim=1
        )
    audio_embeddings = rearrange(audio_embeddings, "() (f d) c -> f d c", d=2)
    torch.cuda.empty_cache()

    return audio_embeddings


@torch.inference_mode()
def compute_wavlm_embeddings(
    raw_audio: torch.Tensor,
    wavlm_model: WavLM_wrapper,
) -> torch.Tensor:
    print("Computing wavlm embeddings")
    # audio = rearrange(raw_audio, "(f s) -> f s", s=640)
    audio = raw_audio
    if audio.shape[0] % 2 != 0:
        audio = torch.cat([audio, torch.zeros(1, 640)], dim=0)
    chunks = 500

    # Get audio embeddings
    audio_embeddings = []
    splits = list(calculate_splits(audio, chunks, dim=0))

    for i, chunk in enumerate(tqdm(splits, desc="Computing wavlm embeddings")):
        wavlm_hidden_states = wavlm_model(chunk.unsqueeze(0).cuda()).squeeze(0)
        audio_embeddings.append(wavlm_hidden_states)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    torch.cuda.empty_cache()

    return audio_embeddings


def get_audio_embeddings(
    audio_path: str,
    audio_rate: int = 16000,
    hubert_model: Optional[HubertModel] = None,
    wavlm_model: Optional[WavLM_wrapper] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load audio embeddings from file or generate them from raw audio.

    Args:
        audio_path: Path to audio file or embeddings
        audio_rate: Audio sample rate
        fps: Frames per second
        audio_emb_type: Type of audio embeddings
        audio_folder: Folder containing raw audio files
        audio_emb_folder: Folder containing audio embedding files
        extra_audio: Whether to include extra audio embeddings
        max_frames: Maximum number of frames to process

    Returns:
        Tuple of (audio embeddings, interpolation audio embeddings, raw audio)
    """
    # Process audio
    audio = None
    raw_audio = None
    audio_interpolation = None

    if os.path.exists(audio_path):
        raw_audio = get_raw_audio(audio_path, audio_rate)
    else:
        raise ValueError(f"Could not find raw audio file at {audio_path}.")

    audio = compute_hubert_embeddings(raw_audio, hubert_model)
    # audio = compute_wavlm_embeddings(raw_audio, wavlm_model)
    audio_interpolation = compute_wavlm_embeddings(raw_audio, wavlm_model)

    return audio, audio_interpolation, raw_audio


@torch.inference_mode()
def sample_keyframes(
    model_keyframes: Any,
    audio_list: torch.Tensor,
    gt_list: torch.Tensor,
    masks_list: torch.Tensor,
    condition: torch.Tensor,
    num_frames: int,
    fps_id: int,
    cond_aug: float,
    device: str,
    embbedings: Optional[torch.Tensor],
    force_uc_zero_embeddings: List[str],
    n_batch_keyframes: int,
    added_frames: int,
    strength: float,
    scale: Optional[Union[float, List[float]]],
    gt_as_cond: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample keyframes using the keyframe generation model.

    Args:
        model_keyframes: The keyframe generation model
        audio_list: List of audio embeddings
        gt_list: List of ground truth frames
        masks_list: List of masks
        condition: Conditioning tensor
        num_frames: Number of frames to generate
        fps_id: FPS ID
        cond_aug: Conditioning augmentation factor
        device: Device to use for computation
        embbedings: Optional embeddings
        force_uc_zero_embeddings: List of embeddings to force to zero in unconditional case
        n_batch_keyframes: Batch size for keyframe generation
        added_frames: Number of additional frames
        strength: Strength parameter for sampling
        scale: Scale parameter for guidance
        gt_as_cond: Whether to use ground truth as conditioning

    Returns:
        Tuple of (latent samples, decoded samples)
    """
    if scale is not None:
        model_keyframes.sampler.guider.set_scale(scale)
    samples_list = []
    samples_z_list = []
    samples_x_list = []

    for i in range(audio_list.shape[0]):
        H, W = condition.shape[-2:]
        assert condition.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        audio_cond = audio_list[i].unsqueeze(0)

        value_dict: Dict[str, Any] = {}
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = condition
        if embbedings is not None:
            value_dict["cond_frames"] = embbedings + cond_aug * torch.randn_like(
                embbedings
            )
        else:
            value_dict["cond_frames"] = condition + cond_aug * torch.randn_like(
                condition
            )
        gt = rearrange(gt_list[i].unsqueeze(0), "b t c h w -> b c t h w").to(device)

        if gt_as_cond:
            value_dict["cond_frames"] = gt[:, :, 0]

        value_dict["cond_aug"] = cond_aug
        value_dict["audio_emb"] = audio_cond

        value_dict["gt"] = gt
        value_dict["masks"] = masks_list[i].unsqueeze(0).transpose(1, 2).to(device)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(
                        model_keyframes.conditioner
                    ),
                    value_dict,
                    [1, 1],
                    T=num_frames,
                    device=device,
                )

                c, uc = model_keyframes.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in ["crossattn"]:
                    if c[k].shape[1] != num_frames:
                        uc[k] = repeat(
                            uc[k],
                            "b ... -> b t ...",
                            t=num_frames,
                        )
                        uc[k] = rearrange(
                            uc[k],
                            "b t ... -> (b t) ...",
                            t=num_frames,
                        )
                        c[k] = repeat(
                            c[k],
                            "b ... -> b t ...",
                            t=num_frames,
                        )
                        c[k] = rearrange(
                            c[k],
                            "b t ... -> (b t) ...",
                            t=num_frames,
                        )

                video = torch.randn(shape, device=device)

                additional_model_inputs: Dict[str, torch.Tensor] = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    n_batch_keyframes, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(
                    input: torch.Tensor, sigma: torch.Tensor, c: Dict[str, torch.Tensor]
                ) -> torch.Tensor:
                    return model_keyframes.denoiser(
                        model_keyframes.model,
                        input,
                        sigma,
                        c,
                        **additional_model_inputs,
                    )

                samples_z = model_keyframes.sampler(
                    denoiser, video, cond=c, uc=uc, strength=strength
                )
                samples_z_list.append(samples_z)

                samples_x = model_keyframes.decode_first_stage(samples_z)
                samples_x_list.append(samples_x)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                samples_list.append(samples)

                video = None

    samples = (
        torch.concat(samples_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_list)
    )
    samples_z = (
        torch.concat(samples_z_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_z_list)
    )
    samples_x = (
        torch.concat(samples_x_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_x_list)
    )

    return samples_z, samples_x


@torch.inference_mode()
def sample_interpolation(
    model: Any,
    samples_z: torch.Tensor,
    samples_x: torch.Tensor,
    audio_interpolation_list: List[torch.Tensor],
    gt_chunks: List[torch.Tensor],
    masks_chunks: List[torch.Tensor],
    condition: torch.Tensor,
    num_frames: int,
    device: str,
    overlap: int,
    fps_id: int,
    cond_aug: float,
    force_uc_zero_embeddings: List[str],
    n_batch: int,
    chunk_size: Optional[int],
    strength: float,
    scale: Optional[float] = None,
    cut_audio: bool = False,
    to_remove: List[bool] = [],
) -> np.ndarray:
    """
    Sample interpolation frames between keyframes.

    Args:
        model: The interpolation model
        samples_z: Latent samples from keyframe generation
        samples_x: Decoded samples from keyframe generation
        audio_interpolation_list: List of audio embeddings for interpolation
        gt_chunks: Ground truth video chunks
        masks_chunks: Mask chunks for conditional generation
        condition: Visual conditioning
        num_frames: Number of frames to generate
        device: Device to run inference on
        overlap: Number of frames to overlap between segments
        fps_id: FPS ID for conditioning
        motion_bucket_id: Motion bucket ID for conditioning
        cond_aug: Conditioning augmentation strength
        force_uc_zero_embeddings: Keys to zero out in unconditional embeddings
        n_batch: Batch size for generation
        chunk_size: Size of chunks for processing (to manage memory)
        strength: Strength of the conditioning
        scale: Optional scale for classifier-free guidance
        cut_audio: Whether to cut audio embeddings
        to_remove: List of flags indicating which frames to remove

    Returns:
        Generated video frames as numpy array
    """
    if scale is not None:
        model.sampler.guider.set_scale(scale)

    # Creating condition for interpolation model. We need to create a list of inputs, each input is  [first, last]
    # The first and last are the first and last frames of the interpolation
    interpolation_cond_list = []
    interpolation_cond_list_emb = []

    samples_x = [sample for i, sample in zip(to_remove, samples_x) if not i]
    samples_z = [sample for i, sample in zip(to_remove, samples_z) if not i]

    for i in range(0, len(samples_z) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append(
            torch.stack([samples_x[i], samples_x[i + 1]], dim=1)
        )
        interpolation_cond_list_emb.append(
            torch.stack([samples_z[i], samples_z[i + 1]], dim=1)
        )

    condition = torch.stack(interpolation_cond_list).to(device)
    audio_cond = torch.stack(audio_interpolation_list).to(device)
    embbedings = torch.stack(interpolation_cond_list_emb).to(device)

    gt_chunks = torch.stack(gt_chunks).to(device)
    masks_chunks = torch.stack(masks_chunks).to(device)

    H, W = condition.shape[-2:]
    F = 8
    C = 4
    shape = (num_frames * audio_cond.shape[0], C, H // F, W // F)

    value_dict: Dict[str, Any] = {}
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = condition

    value_dict["cond_frames"] = embbedings
    value_dict["cond_aug"] = cond_aug
    if cut_audio:
        value_dict["audio_emb"] = audio_cond[:, :, :, :768]
    else:
        value_dict["audio_emb"] = audio_cond

    value_dict["gt"] = rearrange(gt_chunks, "b t c h w -> b c t h w").to(device)
    value_dict["masks"] = masks_chunks.transpose(1, 2).to(device)

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch_overlap(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in ["crossattn"]:
                if c[k].shape[1] != num_frames:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            video = torch.randn(shape, device=device)

            additional_model_inputs: Dict[str, torch.Tensor] = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                n_batch, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            # Debug information
            print(
                f"Shapes - Condition: {condition.shape}, Embeddings: {embbedings.shape}, "
                f"Audio: {audio_cond.shape}, Video: {shape}, Additional inputs: {additional_model_inputs}"
            )

            if chunk_size is not None:
                chunk_size = chunk_size * num_frames

            def denoiser(
                input: torch.Tensor, sigma: torch.Tensor, c: Dict[str, torch.Tensor]
            ) -> torch.Tensor:
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                    num_overlap_frames=overlap,
                    num_frames=num_frames,
                    n_skips=n_batch,
                    chunk_size=chunk_size,
                    **additional_model_inputs,
                )

            samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
            samples_z = rearrange(samples_z, "(b t) c h w -> b t c h w", t=num_frames)
            samples_z[:, 0] = embbedings[:, :, 0]
            samples_z[:, -1] = embbedings[:, :, 1]
            samples_z = rearrange(samples_z, "b t c h w -> (b t) c h w")

            samples_x = model.decode_first_stage(samples_z)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # Free up memory
            video = None

    samples = rearrange(samples, "(b t) c h w -> b t c h w", t=num_frames)
    samples = merge_overlapping_segments(samples, overlap)

    vid = (
        (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
    )

    return vid


@torch.no_grad()
def compute_video_embedding(video_frames, vae_model):
    """Compute embeddings from video"""
    print("Encoding video frames")

    encoded = []
    chunk_size = 16
    resolution = 512

    for i, start_idx in enumerate(
        tqdm(range(0, video_frames.shape[0], chunk_size), desc="Encoding video frames")
    ):
        end_idx = min(start_idx + chunk_size, video_frames.shape[0])
        video_chunk = video_frames[start_idx:end_idx]
        encoded.append(encode_video_chunk(vae_model, video_chunk, resolution))

    encoded = torch.cat(encoded, dim=0)
    torch.cuda.empty_cache()
    return encoded


@torch.no_grad()
def extract_video_landmarks(video_frames, landmarks_model):
    """Extract landmarks from video frames"""
    print("Extracting landmarks from video frames")
    landmarks = []
    batch_size = 10

    for i in tqdm(range(0, len(video_frames), batch_size), desc="Extracting landmarks"):
        batch = video_frames[i : i + batch_size]
        batch_landmarks = landmarks_model.extract_landmarks(batch)
        batch_landmarks = ensure_landmarks_shape(batch_landmarks, landmarks)
        landmarks.extend(batch_landmarks)

    torch.cuda.empty_cache()

    print(np.array(landmarks).shape)

    return np.array(landmarks)


def sample(
    model: Any,
    model_keyframes: Any,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    landmark_folder: Optional[str] = None,
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
    version: str = "svd",
    fps_id: int = 24,
    cond_aug: float = 0.0,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    strength: float = 1.0,
    min_seconds: Optional[int] = None,
    force_uc_zero_embeddings: List[str] = [
        "cond_frames",
        "cond_frames_without_noise",
    ],
    chunk_size: Optional[int] = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    add_zero_flag: bool = False,
    n_batch: int = 1,
    n_batch_keyframes: int = 1,
    compute_until: Union[float, str] = "end",
    extra_audio: Union[bool, str] = False,
    audio_emb_type: str = "wav2vec2",
    extra_naming: str = "",
    what_mask: str = "box",
    fix_occlusion: bool = False,
    position: Optional[List[float]] = None,
    start_frame: int = 0,
    gt_as_cond: bool = False,
    nose_index: int = 28,
    save_occlusion_mask: bool = False,
    recompute: bool = False,
    hubert_model: Optional[HubertModel] = None,
    wavlm_model: Optional[WavLM_wrapper] = None,
    vae_model: Optional[VaeWrapper] = None,
    landmarks_model: Optional[LandmarksExtractor] = None,
) -> None:
    """
    Simple script to generate a single sample conditioned on an image `video_path` or multiple images, one for each
    image file in folder `video_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        output_folder = default(output_folder, "outputs/full_pipeline/svd/")
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        output_folder = default(output_folder, "outputs/full_pipeline/svd_xt/")
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        output_folder = default(
            output_folder, "outputs/full_pipeline/svd_image_decoder/"
        )
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        output_folder = default(
            output_folder, "outputs/full_pipeline/svd_xt_image_decoder/"
        )
    else:
        raise ValueError(f"Version {version} does not exist.")

    os.makedirs(output_folder, exist_ok=True)

    if extra_naming != "":
        video_out_name = (
            os.path.basename(video_path).replace(".mp4", "")
            + "_"
            + extra_naming
            + ".mp4"
        )
    else:
        video_out_name = os.path.basename(video_path)

    out_video_path = os.path.join(output_folder, video_out_name)

    if os.path.exists(out_video_path) and not recompute:
        print(f"Video already exists at {out_video_path}. Skipping.")
        return

    torch.manual_seed(seed)

    video = read_video(video_path, output_format="TCHW")[0]

    h, w = video.shape[2:]
    original_len = video.shape[0]
    video = torch.nn.functional.interpolate(video, (512, 512), mode="bilinear")

    video_emb = compute_video_embedding(video.permute(0, 2, 3, 1), vae_model)
    # video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
    # if video_folder is not None and latent_folder is not None:
    #     video_embedding_path = video_embedding_path.replace(video_folder, latent_folder)
    # video_emb = load_safetensors(video_embedding_path)["latents"].cpu()

    if compute_until == "end":
        compute_until = int((video.shape[0] * 10) / 25)

    if compute_until is not None:
        max_frames = compute_until * (fps_id + 1)
        # Calculate the ceiling to the closest multiple of 14
        remainder = (13 - ((max_frames - 14) % 13)) % 13
        if remainder > 0:
            max_frames = max_frames + remainder
        print(f"Adjusted max_frames to {max_frames} to be a multiple of 14")

    audio, audio_interpolation, raw_audio = get_audio_embeddings(
        audio_path, 16000, hubert_model=hubert_model, wavlm_model=wavlm_model
    )
    landmarks = extract_video_landmarks(video, landmarks_model)

    video = (video / 255.0) * 2.0 - 1.0

    landmarks = scale_landmarks(landmarks[:, :, :2], (h, w), (512, 512))
    if len(landmarks) < len(audio):
        # Repeat last landmark
        landmarks = np.concatenate(
            [
                landmarks,
                landmarks[-1:].repeat(len(audio) - len(landmarks), axis=0),
            ]
        )

    if compute_until is not None:
        if video.shape[0] > max_frames:
            video = video[:max_frames]
            audio = audio[:max_frames]
            landmarks = landmarks[:max_frames]
            audio_interpolation = audio_interpolation[:max_frames]
            video_emb = video_emb[:max_frames] if video_emb is not None else None
            raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
    if min_seconds is not None:
        min_frames = min_seconds * (fps_id + 1)
        video = video[min_frames:]
        audio = audio[min_frames:]
        landmarks = landmarks[min_frames:]
        audio_interpolation = audio_interpolation[min_frames:]
        video_emb = video_emb[min_frames:] if video_emb is not None else None
        raw_audio = raw_audio[min_frames:] if raw_audio is not None else None
    audio = audio

    print(
        "Video has ",
        video.shape[0],
        "frames",
        "and",
        video.shape[0] / 25,
        "seconds",
        "and audio has",
        audio.shape[0],
        "frames",
    )
    print("audio", audio.shape)

    min_len = min(video.shape[0], audio.shape[0])
    video = video[:min_len]
    audio = audio[:min_len]
    landmarks = landmarks[:min_len]
    audio_interpolation = audio_interpolation[:min_len]
    video_emb = video_emb[:min_len] if video_emb is not None else None
    raw_audio = raw_audio[:min_len] if raw_audio is not None else None

    h, w = video.shape[2:]

    model_input = video
    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        if resize_size is not None:
            width, height = (
                (resize_size, resize_size)
                if isinstance(resize_size, int)
                else resize_size
            )
        else:
            width = min(width, 1024)
            height = min(height, 576)
        model_input = torch.nn.functional.interpolate(
            model_input, (height, width), mode="bilinear"
        ).squeeze(0)
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )
    if len(model_input) < len(audio):
        model_input = torch.cat(
            [
                model_input,
                model_input[-1:].repeat(len(audio) - len(model_input)),
            ]
        )

    mask_arms = None
    if fix_occlusion:
        mask_arms = get_segmentation_mask_arms(
            video_path,
            start_frame,
            position,
            original_len,
            (h, w),
            target_size=(64, 64),
        )

        if save_occlusion_mask:
            video_name = os.path.basename(video_path).replace(".mp4", "")
            output_path = f"/vol/paramonos2/projects/antoni/code/Personal/keyface/outputs/{video_name}_mask_arms.npy"
            np.save(output_path, mask_arms)

        mask_arms = mask_arms[:max_frames]

    (
        gt_interpolation,
        gt_keyframes,
        audio_interpolation_list,
        audio_list,
        emb,
        cond,
        masks_keyframes,
        masks_interpolation,
        to_remove,
        test_interpolation_list,
        test_keyframes_list,
    ) = create_pipeline_inputs(
        model_input,
        audio,
        audio_interpolation,
        num_frames,
        video_emb,
        landmarks,
        overlap=overlap,
        add_zero_flag=add_zero_flag,
        what_mask=what_mask,
        mask_arms=mask_arms,
        nose_index=nose_index,
    )

    model_keyframes.en_and_decode_n_samples_a_time = decoding_t
    model.en_and_decode_n_samples_a_time = decoding_t

    additional_audio_frames = (
        model_keyframes.model.diffusion_model.additional_audio_frames
    )
    print(f"Additional audio frames: {additional_audio_frames}")

    audio_list = torch.stack(audio_list).to(device)
    gt_keyframes = torch.stack(gt_keyframes).to(device)
    masks_keyframes = torch.stack(masks_keyframes).to(device)

    audio_list = rearrange(audio_list, "(b t) c d  -> b t c d", t=num_frames)
    gt_keyframes = rearrange(gt_keyframes, "(b t) c h w -> b t c h w", t=num_frames)
    masks_keyframes = rearrange(
        masks_keyframes, "(b t) c h w -> b t c h w", t=num_frames
    )

    # Convert to_remove into chunks of num_frames
    to_remove_chunks = [
        to_remove[i : i + num_frames] for i in range(0, len(to_remove), num_frames)
    ]
    test_keyframes_list = [
        test_keyframes_list[i : i + num_frames]
        for i in range(0, len(test_keyframes_list), num_frames)
    ]

    condition = cond

    audio_cond = audio_list
    condition = condition.unsqueeze(0).to(device)
    embbedings = emb.unsqueeze(0).to(device) if emb is not None else None

    # One batch of keframes is approximately 7 seconds
    chunk_size = 2
    complete_video = []
    complete_audio = []
    start_idx = 0
    last_frame_z = None
    last_frame_x = None
    last_keyframe_idx = None
    last_to_remove = None
    assert len(audio_interpolation_list) == len(gt_interpolation)
    for chunk_start in range(0, len(audio_cond), chunk_size):
        # Clear GPU cache between chunks
        torch.cuda.empty_cache()

        chunk_end = min(chunk_start + chunk_size, len(audio_cond))

        chunk_audio_cond = audio_cond[chunk_start:chunk_end].cuda()

        chunk_gt_keyframes = gt_keyframes[chunk_start:chunk_end].cuda()
        chunk_masks = masks_keyframes[chunk_start:chunk_end].cuda()

        test_keyframes_list_unwrapped = [
            elem
            for sublist in test_keyframes_list[chunk_start:chunk_end]
            for elem in sublist
        ]
        to_remove_chunks_unwrapped = [
            elem
            for sublist in to_remove_chunks[chunk_start:chunk_end]
            for elem in sublist
        ]

        if last_keyframe_idx is not None:
            test_keyframes_list_unwrapped = [
                last_keyframe_idx
            ] + test_keyframes_list_unwrapped
            to_remove_chunks_unwrapped = [last_to_remove] + to_remove_chunks_unwrapped

        last_keyframe_idx = test_keyframes_list_unwrapped[-1]
        last_to_remove = to_remove_chunks_unwrapped[-1]
        # Find the first non-None keyframe in the chunk
        first_keyframe = next(
            (kf for kf in test_keyframes_list_unwrapped if kf is not None), None
        )

        # Find the last non-None keyframe in the chunk
        last_keyframe = next(
            (kf for kf in reversed(test_keyframes_list_unwrapped) if kf is not None),
            None,
        )

        start_idx = next(
            (
                idx
                for idx, comb in enumerate(test_interpolation_list)
                if comb[0] == first_keyframe
            ),
            None,
        )
        end_idx = next(
            (
                idx
                for idx, comb in enumerate(reversed(test_interpolation_list))
                if comb[1] == last_keyframe
            ),
            None,
        )

        if start_idx is not None and end_idx is not None:
            end_idx = (
                len(test_interpolation_list) - 1 - end_idx
            )  # Adjust for reversed enumeration
        end_idx += 1
        if start_idx is None:
            break
        if end_idx < start_idx:
            end_idx = len(audio_interpolation_list)

        audio_interpolation_list_chunk = audio_interpolation_list[start_idx:end_idx]
        chunk_masks_interpolation = masks_interpolation[start_idx:end_idx]
        gt_interpolation_chunks = gt_interpolation[start_idx:end_idx]

        samples_z, samples_x = sample_keyframes(
            model_keyframes,
            chunk_audio_cond,
            chunk_gt_keyframes,
            chunk_masks,
            condition.cuda(),
            num_frames,
            fps_id,
            cond_aug,
            device,
            embbedings.cuda(),
            force_uc_zero_embeddings,
            n_batch_keyframes,
            0,
            strength,
            None,
            gt_as_cond=gt_as_cond,
        )

        if last_frame_x is not None:
            samples_x = torch.cat([last_frame_x.unsqueeze(0), samples_x], axis=0)
            samples_z = torch.cat([last_frame_z.unsqueeze(0), samples_z], axis=0)

        last_frame_x = samples_x[-1]
        last_frame_z = samples_z[-1]

        vid = sample_interpolation(
            model,
            samples_z,
            samples_x,
            audio_interpolation_list_chunk,
            gt_interpolation_chunks,
            chunk_masks_interpolation,
            condition.cuda(),
            num_frames,
            device,
            overlap,
            fps_id,
            cond_aug,
            force_uc_zero_embeddings,
            n_batch,
            chunk_size,
            strength,
            None,
            cut_audio=extra_audio not in ["both", "interp"],
            to_remove=to_remove_chunks_unwrapped,
        )

        if chunk_start == 0:
            complete_video = vid
        else:
            complete_video = np.concatenate([complete_video[:-1], vid], axis=0)

    if raw_audio is not None:
        complete_audio = rearrange(
            raw_audio[: complete_video.shape[0]], "f s -> () (f s)"
        )

    save_audio_video(
        complete_video,
        audio=complete_audio,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=out_video_path,
        keep_intermediate=False,
    )

    print(f"Saved video to {out_video_path}")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_batch_overlap(
    keys: List[str],
    value_dict: Dict[str, Any],
    N: Tuple[int, ...],
    T: Optional[int],
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create a batch dictionary with overlapping frames for model input.

    Args:
        keys: List of keys to include in the batch
        value_dict: Dictionary containing values for each key
        N: Batch dimensions
        T: Number of frames (optional)
        device: Device to place tensors on

    Returns:
        Tuple of (batch dictionary, unconditional batch dictionary)
    """
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "b ... -> (b t) ...", t=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "b ... -> (b t) ...", t=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    input_key: str,
    ckpt: Optional[str] = None,
) -> Tuple[Any, Any, int]:
    """
    Load a model from configuration.

    Args:
        config: Path to model configuration file
        device: Device to load the model on
        num_frames: Number of frames to process
        input_key: Input key for the model
        ckpt: Optional checkpoint path

    Returns:
        Tuple of (model, filter, batch size)
    """
    config = OmegaConf.load(config)

    config["model"]["params"]["input_key"] = input_key

    if ckpt is not None:
        config.model.params.ckpt_path = ckpt

    if "num_frames" in config.model.params.sampler_config.params.guider_config.params:
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )

    if (
        "IdentityGuider"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 1
    elif (
        "MultipleCondVanilla"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 3
    elif (
        "AudioRefMultiCondGuider"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 3
    elif "VanillaSTG" in config.model.params.sampler_config.params.guider_config.target:
        n_batch = 3
    else:
        n_batch = 2  # Conditional and unconditional
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    return model, filter, n_batch


def main(
    filelist: str = "",
    filelist_audio: str = "",
    num_frames: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    landmark_folder: Optional[str] = None,
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
    version: str = "svd",
    fps_id: int = 24,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    strength: float = 1.0,
    model_config: Optional[str] = None,
    model_keyframes_config: Optional[str] = None,
    min_seconds: Optional[int] = None,
    force_uc_zero_embeddings: List[str] = [
        "cond_frames",
        "cond_frames_without_noise",
    ],
    chunk_size: Optional[int] = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    keyframes_ckpt: Optional[str] = None,
    interpolation_ckpt: Optional[str] = None,
    add_zero_flag: bool = False,
    extra_audio: Optional[str] = None,
    compute_until: str = "end",
    starting_index: int = 0,
    audio_emb_type: str = "wav2vec2",
    scale: Optional[List[float]] = None,
    mix_audio: bool = False,
    what_mask: str = "box",
    fix_occlusion: bool = False,
    position: Optional[List[float]] = None,
    start_frame: int = 0,
    gt_as_cond: bool = False,
    nose_index: int = 28,
    save_occlusion_mask: bool = False,
    recompute: bool = False,
) -> None:
    """
    Main function to run the dubbing pipeline.

    Args:
        filelist: Path to a text file with video paths or a single video path
        filelist_audio: Path to a text file with audio paths or a single audio path
        num_frames: Number of frames to process at once
        resize_size: Size to resize frames to
        video_folder: Folder containing video files
        latent_folder: Folder for latent representations
        landmark_folder: Folder containing facial landmarks
        emotion_folder: Folder containing emotion data
        audio_folder: Folder containing audio files
        audio_emb_folder: Folder containing audio embeddings
        version: Model version
        fps_id: Frames per second ID
        motion_bucket_id: Motion bucket ID for conditioning
        cond_aug: Conditioning augmentation strength
        seed: Random seed
        decoding_t: Number of frames to decode at once
        device: Device to run on
        output_folder: Folder to save outputs
        strength: Strength of conditioning
        model_config: Path to model configuration
        model_keyframes_config: Path to keyframe model configuration
        min_seconds: Minimum seconds to process
        lora_path_interp: Path to LoRA weights for interpolation model
        lora_path_keyframes: Path to LoRA weights for keyframe model
        force_uc_zero_embeddings: Keys to zero out in unconditional embeddings
        chunk_size: Size of chunks for processing
        overlap: Number of frames to overlap
        keyframes_ckpt: Path to keyframes model checkpoint
        interpolation_ckpt: Path to interpolation model checkpoint
        add_zero_flag: Whether to add zero fla
        recurse: Whether to recurse through directories
        extra_audio: Extra audio configuration
        compute_until: When to stop computation
        starting_index: Index to start processing from
        audio_emb_type: Type of audio embeddings
        is_image_model: Whether using an image model
        scale: Scale for classifier-free guidance
        mix_audio: Whether to mix audio
        what_mask: Type of mask to use
        fix_occlusion: Whether to fix occlusions
        position: Position for segmentation
        start_frame: Frame to start from
        gt_as_cond: Whether to use ground truth as conditioning
        nose_index: Index of nose landmark
        save_occlusion_mask: Whether to save occlusion mask
    """
    print("Scale: ", scale)
    num_frames = default(num_frames, 14)
    model, filter, n_batch = load_model(
        model_config,
        device,
        num_frames,
        "latents",
        interpolation_ckpt,
    )

    model_keyframes, filter, n_batch_keyframes = load_model(
        model_keyframes_config,
        device,
        num_frames,
        "latents",
        keyframes_ckpt,
    )

    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm_model = WavLM_wrapper(
        model_size="Base+",
        feed_as_frames=False,
        merge_type="None",
        model_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "pretrained_models", "checkpoints", "WavLM-Base+.pt",
        ),
    ).cuda()
    vae_model = VaeWrapper("video")
    landmarks_model = LandmarksExtractor()

    if scale is not None:
        if len(scale) == 1:
            scale = scale[0]
        model_keyframes.sampler.guider.set_scale(scale)

    # Open the filelist and read the video paths
    if filelist.endswith(".txt"):
        with open(filelist, "r") as f:
            video_paths = f.readlines()

        # Remove the newline character from each path
        video_paths = [path.strip() for path in video_paths]
    elif filelist.endswith(".mp4"):
        video_paths = [filelist]
    else:
        video_paths = os.listdir(filelist)
        video_paths = [
            os.path.join(filelist, path)
            for path in video_paths
            if path.endswith(".mp4")
        ]

    if filelist_audio:
        if filelist_audio.endswith(".txt"):
            with open(filelist_audio, "r") as f:
                audio_paths = f.readlines()

            # Remove the newline character from each path
            audio_paths = [path.strip() for path in audio_paths]
        elif filelist_audio.endswith(".wav") or filelist_audio.endswith(".mp4"):
            audio_paths = [filelist_audio]
        else:
            audio_paths = os.listdir(filelist_audio)
            audio_paths = [
                os.path.join(filelist_audio, path)
                for path in audio_paths
                if path.endswith(".wav") or path.endswith(".mp4")
            ]

        if ".mp4" in audio_paths[0]:
            audio_paths = [
                path.strip().replace(f"/{video_folder}", f"/{audio_folder}")
                for path in audio_paths
            ]
        else:
            audio_paths = [path.strip() for path in audio_paths]
    else:
        audio_paths = [
            video_path.replace(f"/{video_folder}", f"/{audio_folder}").replace(
                ".mp4", ".wav"
            )
            for video_path in video_paths
        ]

    if mix_audio:
        # Randomly shuffle audio paths with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(audio_paths)
        random.shuffle(video_paths)

    if starting_index:
        video_paths = video_paths[starting_index:]
        audio_paths = audio_paths[starting_index:]

    for video_path, audio_path in zip(video_paths, audio_paths):
        try:
            sample(
                model,
                model_keyframes,
                video_path=video_path,
                audio_path=audio_path,
                num_frames=num_frames,
                resize_size=resize_size,
                video_folder=video_folder,
                latent_folder=latent_folder,
                landmark_folder=landmark_folder,
                audio_folder=audio_folder,
                audio_emb_folder=audio_emb_folder,
                version=version,
                fps_id=fps_id,
                cond_aug=cond_aug,
                seed=seed,
                decoding_t=decoding_t,
                device=device,
                output_folder=output_folder,
                strength=strength,
                min_seconds=min_seconds,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
                chunk_size=chunk_size,
                overlap=overlap,
                add_zero_flag=add_zero_flag,
                n_batch=n_batch,
                n_batch_keyframes=n_batch_keyframes,
                extra_audio=extra_audio,
                compute_until=compute_until,
                audio_emb_type=audio_emb_type,
                extra_naming=os.path.basename(audio_path).split(".")[0]
                if filelist_audio
                else "",
                what_mask=what_mask,
                fix_occlusion=fix_occlusion,
                position=position,
                start_frame=start_frame,
                gt_as_cond=gt_as_cond,
                nose_index=nose_index,
                save_occlusion_mask=save_occlusion_mask,
                recompute=recompute,
                hubert_model=hubert_model,
                wavlm_model=wavlm_model,
                vae_model=vae_model,
                landmarks_model=landmarks_model,
            )
        except Exception as e:
            raise e
            print(f"Error processing {video_path} and {audio_path}: {e}")
            continue


if __name__ == "__main__":
    Fire(main)