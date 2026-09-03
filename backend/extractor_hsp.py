# -*- coding: utf-8 -*-
"""
HSP-BiMamba skeleton extraction — aligned with training (vsl_hsp_bimamba_k4).

Critical: extract_video_stable picks T video frames FIRST, then runs MediaPipe
on exactly those frames. Webcam must do the same (buffer wide, pick T, then MP).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_HSP_ROOT = _REPO / "vsl_hsp_bimamba_k4"
if _HSP_ROOT.is_dir() and str(_HSP_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSP_ROOT))

from extract_skeletons_holistic_stable import (  # type: ignore
    extract_from_frames_stable,
    extract_video_stable,
)
from hsp_bimamba.skeleton_layout import convert_holistic143_to_mp75  # type: ignore

DEFAULT_NUM_FRAMES = 30
DEFAULT_BUFFER_FRAMES = 50


def _sample_frames(frames_bgr: list, target: int) -> list:
    t = len(frames_bgr)
    if t == 0:
        return []
    if t < target:
        return frames_bgr + [frames_bgr[-1]] * (target - t)
    if t > target:
        idx = np.linspace(0, t - 1, target, dtype=int)
        return [frames_bgr[i] for i in idx]
    return frames_bgr


def holistic143_to_mp75(sk143: np.ndarray) -> np.ndarray:
    return convert_holistic143_to_mp75(sk143.astype(np.float32))


def extract_hsp_from_frames(
    frames_bgr: list,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
) -> np.ndarray | None:
    """
    Webcam frames -> MP75 [num_frames, 75, 3].

    1. Normalize capture length to buffer_frames (pad / subsample).
    2. Linspace-pick num_frames images (same as extract_video_stable).
    3. MediaPipe stable on exactly those num_frames images.
    """
    if not frames_bgr:
        return None
    buf = _sample_frames(frames_bgr, buffer_frames)
    picked = _sample_frames(buf, num_frames)
    sk143 = extract_from_frames_stable(picked)
    return holistic143_to_mp75(sk143)


def extract_hsp_from_video(
    video_path: str | Path,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> np.ndarray | None:
    sk143 = extract_video_stable(Path(video_path), num_frames=num_frames)
    if sk143 is None:
        return None
    return holistic143_to_mp75(sk143)
