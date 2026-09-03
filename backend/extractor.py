# -*- coding: utf-8 -*-
"""
MediaPipe Holistic skeleton extractor → numpy array [T, V, 3].

Supported formats:
- V=42  (hands only): left hand 21 + right hand 21
- V=110 (hand+face): 42 hands + face mesh 68 selected landmarks
- V=143 (WB-DPNet layout): hand 42 + face 68 + pose 33
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

NUM_FRAMES = 16

# face landmark indices that give a reasonable 68-point subset
FACE_68_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 10, 151, 9, 8, 168, 6, 197, 195, 5, 4,
    1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18,
    61, 185, 40, 39, 38, 37, 0, 267, 269, 270,
][:68]


class SkeletonExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

    def close(self):
        self.holistic.close()

    def extract_from_video_path(
        self,
        video_path: str | Path,
        num_joints: int = 110,
        num_frames: int | None = None,
    ) -> np.ndarray | None:
        """Read video → extract skeleton → return [T, V, 3] or None if failed."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frames_landmarks = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            lm = self._process_frame(frame, num_joints=num_joints)
            frames_landmarks.append(lm)
        cap.release()
        if not frames_landmarks:
            return None
        seq = np.stack(frames_landmarks, axis=0)  # [T, V, 3]
        return self._align(seq, num_frames=num_frames)

    def extract_from_frames(
        self,
        frames: list[np.ndarray],
        num_joints: int = 110,
        num_frames: int | None = None,
    ) -> np.ndarray | None:
        """Extract from list of BGR frames."""
        if not frames:
            return None
        landmarks = [self._process_frame(f, num_joints=num_joints) for f in frames]
        seq = np.stack(landmarks, axis=0)
        return self._align(seq, num_frames=num_frames)

    def _process_frame(self, frame: np.ndarray, num_joints: int) -> np.ndarray:
        """Returns [V, 3] (x,y,z) for one frame; zeros if not detected."""
        if num_joints not in (42, 110, 143):
            raise ValueError(f"Unsupported num_joints={num_joints}. Use 42, 110, or 143.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)

        def lm_to_arr(lm_list, n: int) -> np.ndarray:
            arr = np.zeros((n, 3), dtype=np.float32)
            if lm_list:
                for i, lm in enumerate(lm_list.landmark[:n]):
                    arr[i] = [lm.x, lm.y, lm.z]
            return arr

        left_hand = lm_to_arr(result.left_hand_landmarks, 21)
        right_hand = lm_to_arr(result.right_hand_landmarks, 21)

        if num_joints == 42:
            return np.concatenate([left_hand, right_hand], axis=0)  # [42, 3]

        face_arr = np.zeros((68, 3), dtype=np.float32)
        if result.face_landmarks:
            all_face = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks.landmark], dtype=np.float32)
            for j, fi in enumerate(FACE_68_IDX):
                face_arr[j] = all_face[fi]

        if num_joints == 110:
            return np.concatenate([left_hand, right_hand, face_arr], axis=0)  # [110, 3]

        # num_joints == 143: hand(42) + face(68) + pose(33)
        pose_arr = np.zeros((33, 3), dtype=np.float32)
        if result.pose_landmarks:
            for j, lm in enumerate(result.pose_landmarks.landmark[:33]):
                pose_arr[j] = [lm.x, lm.y, lm.z]

        return np.concatenate([left_hand, right_hand, face_arr, pose_arr], axis=0)  # [143, 3]

    def _align(self, seq: np.ndarray, num_frames: int | None = None) -> np.ndarray:
        """Temporal sampling/padding (default NUM_FRAMES=16 for legacy CurriVSL models)."""
        if num_frames is not None and int(num_frames) <= 0:
            return seq.astype(np.float32)
        target = NUM_FRAMES if num_frames is None else int(num_frames)
        T = seq.shape[0]
        if T < target:
            pad = np.repeat(seq[-1:], target - T, axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        elif T > target:
            idx = np.linspace(0, T - 1, target, dtype=int)
            seq = seq[idx]
        return seq.astype(np.float32)
