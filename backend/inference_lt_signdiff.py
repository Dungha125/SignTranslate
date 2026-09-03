# -*- coding: utf-8 -*-
"""LT-SignDiff Top-200 inference for sign_translate (110 joints, T=16)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
_LT_ROOT = _REPO / "LT_SignDiff"
if _LT_ROOT.is_dir() and str(_LT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LT_ROOT))

from gallery import append_embedding, gallery_stats, load_gallery

from lt_signdiff.models.hybrid import LTSignDiff  # type: ignore

def _align_skeleton(sk: np.ndarray, num_frames: int = 16, num_joints: int = 110) -> np.ndarray:
    sk = np.asarray(sk, dtype=np.float32)
    if sk.ndim != 3:
        raise ValueError(f"Expected [T,V,C], got {sk.shape}")
    T, V, C = sk.shape
    if T < num_frames:
        pad = np.repeat(sk[-1:], num_frames - T, axis=0)
        sk = np.concatenate([sk, pad], axis=0)
    elif T > num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
        sk = sk[idx]
    if V > num_joints:
        sk = sk[:, :num_joints]
    elif V < num_joints:
        pad = np.zeros((num_frames, num_joints - V, C), dtype=np.float32)
        sk = np.concatenate([sk, pad], axis=1)
    if sk.shape[2] > 3:
        sk = sk[:, :, :3]
    elif sk.shape[2] < 3:
        pad = np.zeros((num_frames, num_joints, 3 - sk.shape[2]), dtype=np.float32)
        sk = np.concatenate([sk, pad], axis=2)
    return sk.astype(np.float32)


class LTSignDiffModel:
    """Closed-set LT-SignDiff recognizer (default: Top-200 VSL glosses)."""

    def __init__(
        self,
        ckpt_path: str | Path,
        num_joints: int = 110,
        num_frames: int = 16,
        top_k: int = 10,
        use_tta: bool = True,
        alpha: float | None = None,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_joints = int(num_joints)
        self.num_frames = int(num_frames)
        self.top_k = int(top_k)
        self.use_tta = bool(use_tta)

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        meta: dict[str, Any] = dict(ckpt.get("meta") or {})
        label2id = ckpt.get("label_to_idx") or ckpt.get("label2id")
        if label2id is None:
            # sidecar JSON next to checkpoint
            side = Path(ckpt_path).with_name("label_to_idx.json")
            if side.is_file():
                import json

                label2id = json.loads(side.read_text(encoding="utf-8"))
            else:
                raise KeyError("Checkpoint missing label_to_idx / label2id")

        # normalize keys
        self.label2id = {str(k): int(v) for k, v in label2id.items()}
        id2 = ckpt.get("id2label")
        if isinstance(id2, dict) and id2:
            self.id2label = {int(k): str(v) for k, v in id2.items()}
        else:
            self.id2label = {int(v): str(k) for k, v in self.label2id.items()}

        num_classes = len(self.label2id)
        self.num_frames = int(meta.get("num_frames", self.num_frames))
        self.num_joints = int(meta.get("num_joints", self.num_joints))
        use_alpha = float(alpha if alpha is not None else meta.get("alpha", 0.65))

        self.model = LTSignDiff(
            num_classes=num_classes,
            num_joints=self.num_joints,
            in_dim=3,
            conv_channels=(3, 64, 128),
            lstm_hidden=256,
            lstm_layers=2,
            dropout=0.0,
            classifier_scale=float(meta.get("classifier_scale", 12.0)),
            alpha=use_alpha,
            knn_k=5,
            diffusion_hidden=512,
            diffusion_steps=50,
        ).to(self.device)

        state = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[LT-SignDiff] missing keys: {len(missing)}")
        if unexpected:
            print(f"[LT-SignDiff] unexpected keys: {len(unexpected)}")

        # Mark bank ready if M was saved with content
        if hasattr(self.model, "M") and self.model.M.numel() > 0 and float(self.model.M.abs().sum()) > 0:
            self.model._bank_built = True
        self.use_hybrid = bool(meta.get("use_hybrid", True)) and self.model._bank_built

        self.model.eval()

        # kNN gallery (train refs + user enrollments)
        self.gallery_z = None
        self.gallery_y = None
        self.gallery_meta: dict[str, Any] = {}
        self._load_gallery()

        print(
            f"[LT-SignDiff] Loaded {num_classes} classes from {ckpt_path} "
            f"on {self.device} hybrid={self.use_hybrid} tta={self.use_tta} "
            f"gallery={0 if self.gallery_z is None else self.gallery_z.size(0)}"
        )

    def _load_gallery(self) -> None:
        z, y, meta = load_gallery(self.ckpt_path, self.device)
        self.gallery_z, self.gallery_y = z, y
        self.gallery_meta = meta or {}

    @torch.no_grad()
    def embed(self, skeleton: np.ndarray) -> torch.Tensor:
        sk = _align_skeleton(skeleton, self.num_frames, self.num_joints)
        x = torch.from_numpy(sk).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode(x), dim=-1)[0]

    def enroll(self, skeleton: np.ndarray, gloss: str) -> dict:
        gloss = gloss.strip()
        if gloss not in self.label2id:
            raise ValueError(f"Gloss '{gloss}' không thuộc vocab model ({self.model.num_classes} lớp)")
        cid = int(self.label2id[gloss])
        z = self.embed(skeleton)
        info = append_embedding(self.ckpt_path, z, cid, self.device, source="enroll")
        self._load_gallery()
        return {"gloss": gloss, **info}

    def gallery_info(self) -> dict:
        if self.gallery_z is None:
            return {"total_refs": 0, "num_classes_covered": 0, "by_gloss": {}}
        return gallery_stats(self.gallery_z, self.gallery_y, self.id2label)

    def _logits(self, sk: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(sk).unsqueeze(0).to(self.device)
        return self.model(x, use_hybrid=self.use_hybrid)[0]

    def _knn_probs(self, sk: np.ndarray, k: int = 11) -> torch.Tensor | None:
        if self.gallery_z is None or self.gallery_y is None:
            return None
        z = self.embed(sk)
        sims = z @ self.gallery_z.t()
        topv, topi = sims.topk(min(k, sims.numel()))
        # weighted vote → class probabilities
        scores = torch.zeros(self.model.num_classes, device=self.device)
        for s, idx in zip(topv, topi):
            scores[int(self.gallery_y[idx])] += float(torch.exp((s - topv[0]) * 12.0))
        return F.softmax(scores, dim=-1)

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> list[dict]:
        sk = _align_skeleton(skeleton, self.num_frames, self.num_joints)
        logits = self._logits(sk)

        if self.use_tta:
            sk_flip = sk.copy()
            sk_flip[..., 0] = 1.0 - sk_flip[..., 0]
            logits = 0.5 * (logits + self._logits(sk_flip))
            sk_shift = np.roll(sk, 1, axis=0)
            sk_shift[0] = sk[0]
            logits = (2.0 * logits + self._logits(sk_shift)) / 3.0

        ce_probs = F.softmax(logits, dim=-1)
        knn_probs = self._knn_probs(sk)
        if knn_probs is not None:
            n_refs = self.gallery_z.size(0)
            enroll_counts = self.gallery_meta.get("enroll_counts") or {}
            n_enroll = sum(int(v) for v in enroll_counts.values()) if enroll_counts else 0
            # Offline train/val gallery alone is weaker than hybrid CE; only lean on
            # kNN heavily after the user enrolls personal refs.
            if n_enroll > 0:
                knn_w = min(0.82, 0.45 + 0.02 * min(n_enroll, 20) + 0.0002 * n_refs)
            else:
                knn_w = min(0.28, 0.12 + 0.00025 * n_refs)
            probs = (1.0 - knn_w) * ce_probs + knn_w * knn_probs
        else:
            probs = ce_probs

        k = min(self.top_k, probs.numel())
        top_probs, top_ids = probs.topk(k)
        results: list[dict] = []
        for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
            results.append(
                {
                    "rank": rank,
                    "gloss": self.id2label.get(int(idx), f"id_{idx}"),
                    "score": round(float(prob), 4),
                    "confidence_pct": round(float(prob) * 100.0, 2),
                }
            )
        return results
