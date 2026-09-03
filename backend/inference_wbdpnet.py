# -*- coding: utf-8 -*-
"""
WBDPNet inference wrapper for sign_translate.

The bundled WB-DPNet checkpoint expects skeleton layout:
  (B, T, V, C) with V=143 = hand(42) + face(68) + pose(33), C=3.
Our API provides skeleton as numpy array of shape (T, V, C).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --- import bundled WBDPNet code (extracted zip) ------------------------------
_BUNDLE_WBDP_PARENT = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "wbdpnet_vsl_v2_be_bundle"
    / "wbdpnet_vsl_v2_be"
)

if _BUNDLE_WBDP_PARENT.is_dir():
    sys.path.insert(0, str(_BUNDLE_WBDP_PARENT))
    from wbdpnet import WBDPNet  # type: ignore
else:
    WBDPNet = None  # type: ignore


class WBDPNetModel:
    def __init__(
        self,
        ckpt_path: str | Path,
        num_joints: int = 143,
        top_k: int = 10,
        proto_alpha: float = 0.5,
    ):
        """
        Parameters
        - ckpt_path: checkpoint .pt (contains ckpt["model"] and label maps)
        - num_joints: must be 143 for this model
        - proto_alpha: logits := logits + alpha * proto_logits (matches eval_vsl.py)
        """
        if WBDPNet is None:
            raise RuntimeError(
                f"WBDPNet code not found under {_BUNDLE_WBDP_PARENT}. "
                "Make sure you extracted wbdpnet_vsl_v2_be_bundle.zip into sign_translate/models."
            )
        if int(num_joints) != 143:
            raise ValueError(f"WBDPNet expects num_joints=143, got num_joints={num_joints}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_joints = 143
        self.top_k = int(top_k)
        self.proto_alpha = float(proto_alpha)

        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        label_to_idx = ckpt.get("label_to_idx")
        if label_to_idx is None:
            # Fallback to label_to_idx.json next to checkpoint
            guess = ckpt_path.parent / "label_to_idx.json"
            with open(guess, "r", encoding="utf-8") as f:
                label_to_idx = json.load(f)

        # label_to_idx is {label_str: idx}
        self.id2label: dict[int, str] = {int(v): str(k) for k, v in label_to_idx.items()}
        num_classes = len(self.id2label)

        # README/eval_vsl loads WBDPNet with num_signers=1 for VSL-style inference
        self.model = WBDPNet(num_classes=num_classes, num_signers=1).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> list[dict]:
        """
        skeleton: numpy float32 [T, V, C] where T=16, V=143, C=3
        """
        sk = torch.from_numpy(skeleton).unsqueeze(0).to(self.device).float()  # [1, T, V, C]
        if sk.ndim != 4 or sk.size(2) != 143:
            raise ValueError(f"Expected skeleton [T,143,3], got shape {tuple(sk.shape)}")

        lengths = torch.tensor([sk.size(1)], dtype=torch.long, device=self.device)
        signer_ids = torch.zeros((1,), dtype=torch.long, device=self.device)

        out = self.model(sk, signer_ids=signer_ids, lengths=lengths)
        logits = out["logits"] + self.proto_alpha * out.get("proto_logits", 0.0)
        probs = F.softmax(logits, dim=-1)[0]  # [num_classes]

        k = min(self.top_k, logits.size(-1))
        top_probs, top_ids = probs.topk(k)

        results: list[dict] = []
        for rank, (prob, idx_t) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
            gloss = self.id2label.get(int(idx_t), f"id_{idx_t}")
            results.append(
                {
                    "rank": rank,
                    "gloss": gloss,
                    "score": round(float(logits[0, int(idx_t)].item()), 4),
                    "confidence_pct": round(float(prob) * 100.0, 2),
                }
            )
        return results

