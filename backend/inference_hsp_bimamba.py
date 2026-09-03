# -*- coding: utf-8 -*-
"""HSP-BiMamba inference wrapper for sign_translate (MP75, T=30)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
_HSP_ROOT = _REPO / "vsl_hsp_bimamba_k4"
if _HSP_ROOT.is_dir() and str(_HSP_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSP_ROOT))

from hsp_bimamba.infer import load_predictor  # type: ignore
from hsp_bimamba.skeleton_layout import convert_holistic143_to_mp75  # type: ignore
from hsp_bimamba.skeleton_preprocess import temporal_sample  # type: ignore


class HSPBiMambaModel:
    def __init__(
        self,
        ckpt_path: str | Path,
        num_joints: int = 75,
        num_frames: int = 30,
        top_k: int = 10,
        protocol: str = "skeleton_sota_mp75",
        use_tta: bool = False,
    ):
        if int(num_joints) != 75:
            raise ValueError(f"HSP-BiMamba MP75 expects num_joints=75, got {num_joints}")
        self.num_joints = 75
        self.num_frames = int(num_frames)
        self.top_k = int(top_k)
        self.use_tta = bool(use_tta)
        self.predictor = load_predictor(ckpt_path, protocol=protocol)
        self.id2label = self.predictor.id2label

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> list[dict]:
        sk = skeleton.astype(np.float32)
        if sk.shape[1] >= 143:
            if sk.shape[1] >= 143:
                sk = convert_holistic143_to_mp75(sk)
            sk = temporal_sample(
                sk, self.num_frames,
                train=False, random_temporal_crop=False, eval_temporal_center=True,
            )
        elif sk.shape[1] == 75 and sk.shape[0] != self.num_frames:
            sk = temporal_sample(
                sk, self.num_frames,
                train=False, random_temporal_crop=False, eval_temporal_center=True,
            )

        x = torch.from_numpy(sk).unsqueeze(0).to(self.predictor.device)
        out = self.predictor.model(x)
        logits = out.logits
        if self.predictor.prototype_logit_weight > 0.0 and out.proto_logits is not None:
            logits = logits + self.predictor.prototype_logit_weight * out.proto_logits
        logits = logits[0]
        probs = F.softmax(logits, dim=-1)
        k = min(self.top_k, logits.numel())
        top_probs, top_ids = probs.topk(k)

        results: list[dict] = []
        for rank, (prob, idx_t) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
            gloss = self.id2label.get(int(idx_t), f"id_{idx_t}")
            results.append({
                "rank": rank,
                "gloss": gloss,
                "score": round(float(logits[int(idx_t)].item()), 4),
                "confidence_pct": round(float(prob) * 100.0, 2),
            })
        return results
