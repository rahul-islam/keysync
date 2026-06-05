"""
Composite a KeySync inference output (512x512) back into the original
25fps frames so the result keeps the original resolution / aspect ratio.

The crop bounding boxes are re-derived from the saved landmarks via the
same VideoPreProcessor that crop_video.py used, so we do not need crop
metadata persisted at crop time as long as the same crop parameters
(crop_scale_factor=2, crop_type="per_frame") are used in both places.

Audio is copied from the inference output (the dubbed audio).

Usage:
    python scripts/util/uncrop_video.py \\
        --video        path/to/videos_25fps/clip.mp4 \\
        --landmarks    path/to/landmarks/clip.npy \\
        --inference    path/to/keysync_out/clip_audio.mp4 \\
        --output       path/to/recomposed/clip_audio.mp4
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video

# Make sgm / scripts importable when run as a standalone script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.util.video_processor import VideoPreProcessor  # noqa: E402
from sgm.util import save_audio_video  # noqa: E402


def derive_crop_data(landmarks: np.ndarray, video_thwc_shape):
    """Re-run the VideoPreProcessor crop-data pipeline (without doing the
    actual crop) to recover the per-frame bbox used at crop time."""
    t, h, w, _ = video_thwc_shape
    # _extract_crop_data wants (T, C, H, W) for the .shape[2]/.shape[1] checks.
    dummy = torch.zeros((t, 3, h, w), dtype=torch.uint8)
    pp = VideoPreProcessor()
    cd = pp._extract_crop_data(landmarks, dummy)
    cd = pp._smooth_crop_data(cd)
    cd = pp._refine_crop_data(cd, h, w)
    return cd


_FEATHER_CACHE = {}


def _feather_alpha(h: int, w: int, feather_px: int) -> torch.Tensor:
    """Per-pixel blend weight (H, W, 1) in [0, 1] for the pasted patch:
    1.0 in the interior, ramping linearly to 0.0 over the outer `feather_px`
    pixels on every side so the patch fades into the original frame instead
    of leaving a hard rectangular seam. Corners feather too (min of the two
    edge distances). Masks are cached per (h, w, feather_px)."""
    key = (h, w, feather_px)
    cached = _FEATHER_CACHE.get(key)
    if cached is not None:
        return cached
    if feather_px <= 0:
        alpha = torch.ones((h, w, 1), dtype=torch.float32)
    else:
        ys = torch.minimum(torch.arange(h), torch.arange(h).flip(0)).float()
        xs = torch.minimum(torch.arange(w), torch.arange(w).flip(0)).float()
        dist = torch.minimum(ys.unsqueeze(1), xs.unsqueeze(0))  # (H, W)
        alpha = (dist / float(feather_px)).clamp(0.0, 1.0).unsqueeze(-1)
    _FEATHER_CACHE[key] = alpha
    return alpha


def composite(original_thwc: torch.Tensor,
              inference_thwc: torch.Tensor,
              crop_data,
              feather: float = 0.08) -> torch.Tensor:
    """Paste each inference frame back into the corresponding original
    frame at its crop bbox. Only the first min(T_inf, T_orig) frames are
    composited; tail original frames are preserved as-is.

    `feather` is the fraction of the (shorter) box side over which the patch
    is alpha-blended into the original frame at the border, softening the
    seam. 0 disables feathering (hard paste, the previous behaviour)."""
    out = original_thwc.clone()
    t = min(inference_thwc.shape[0], original_thwc.shape[0], len(crop_data))
    for i in range(t):
        bbox = crop_data[i]
        y0, y1 = int(bbox.y_start), int(bbox.y_end)
        x0, x1 = int(bbox.x_start), int(bbox.x_end)
        h_box, w_box = y1 - y0, x1 - x0
        if h_box <= 0 or w_box <= 0:
            continue
        # (H, W, C) -> (1, C, H, W) for interpolate
        patch = inference_thwc[i].permute(2, 0, 1).unsqueeze(0).float()
        patch = F.interpolate(patch, size=(h_box, w_box),
                              mode="bilinear", align_corners=False)
        patch = patch.clamp(0, 255).squeeze(0).permute(1, 2, 0)  # (H, W, C) float
        if feather and feather > 0:
            feather_px = max(1, int(round(feather * min(h_box, w_box))))
            alpha = _feather_alpha(h_box, w_box, feather_px)  # (H, W, 1) float
            region = out[i, y0:y1, x0:x1].float()
            blended = alpha * patch + (1.0 - alpha) * region
            out[i, y0:y1, x0:x1] = blended.clamp(0, 255).to(torch.uint8)
        else:
            out[i, y0:y1, x0:x1] = patch.to(torch.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="Original 25fps video (the one landmarks were extracted from)")
    ap.add_argument("--landmarks", required=True,
                    help=".npy of shape (T, N, 2)")
    ap.add_argument("--inference", required=True,
                    help="KeySync 512x512 output mp4 (with audio)")
    ap.add_argument("--output", required=True,
                    help="Output mp4 at original resolution")
    ap.add_argument("--feather", type=float, default=0.08,
                    help="Fraction of the (shorter) crop-box side over which "
                         "to alpha-blend the patch into the original frame at "
                         "the border, softening the seam. 0 = hard paste.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Read original (T, H, W, C) uint8.
    orig, _, _ = read_video(args.video, output_format="THWC")

    landmarks = np.load(args.landmarks)
    # gen_landmarks.py truncates landmarks to per-batch outputs; trim to the
    # shorter of the two so derive_crop_data can pair frames 1:1.
    t = min(orig.shape[0], landmarks.shape[0])
    orig = orig[:t]
    landmarks = landmarks[:t]

    crop_data = derive_crop_data(landmarks, orig.shape)

    # Read inference output frames + audio.
    inf, audio, info = read_video(args.inference, output_format="THWC")
    audio_fps = int(info.get("audio_fps", 16000)) if audio is not None and audio.numel() else 16000

    recomposed = composite(orig, inf, crop_data, feather=args.feather)  # (T, H, W, C) uint8

    # save_audio_video wants (T, C, H, W) and accepts ndarray.
    recomposed_tchw = recomposed.permute(0, 3, 1, 2).numpy()

    audio_arg = audio if (audio is not None and audio.numel() > 0) else None
    save_audio_video(
        recomposed_tchw,
        audio=audio_arg,
        frame_rate=25,
        sample_rate=audio_fps,
        save_path=args.output,
    )
    print(f"Wrote {args.output}  ({recomposed.shape[0]} frames, "
          f"{recomposed.shape[2]}x{recomposed.shape[1]})")


if __name__ == "__main__":
    main()
