# -*- coding: utf-8 -*-
"""Suy luận LT-SignDiff v2 cho sign_translate.

Khác v1 ở ba điểm quan trọng:
  * đầu vào là skeleton **đầy đủ frame, 143 khớp** (tay + mặt + pose) rồi mới
    dựng đặc trưng v2 (T=32, 74 khớp, 10 kênh) — không dùng cache 16 frame nữa;
  * điểm số cuối là tổ hợp có trọng số của classifier, prototype và kNN gallery,
    trọng số được chốt trên tập val lúc huấn luyện và lưu trong checkpoint;
  * TTA gồm bản gốc, bản lật ngang và hai lát thời gian.
"""
from __future__ import annotations

import os
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

from lt_signdiff.data.features_v2 import (  # type: ignore
    NUM_CHANNELS_V2,
    NUM_JOINTS_V2,
    build_features,
    mask_from_zeros,
    mirror_features,
)
from lt_signdiff.models.lt_signdiff_v2 import LTSignDiffV2  # type: ignore


class LTSignDiffV2Model:
    """Nhận dạng closed-set với encoder v2 + hợp nhất classifier/prototype/kNN."""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        num_frames: int = 32,
        top_k: int = 10,
        use_tta: bool = True,
        **_ignored,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = bool(use_tta)
        self.top_k = int(top_k)

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        cfg: dict[str, Any] = dict(ckpt.get("config") or {})
        meta: dict[str, Any] = dict(ckpt.get("meta") or {})
        cfg.update({k: v for k, v in meta.items() if k in cfg or k in
                    ("num_frames", "num_joints", "in_channels", "fusion_weights")})

        label2id = ckpt.get("label_to_idx")
        if label2id is None:
            side = self.ckpt_path.with_name("label_to_idx.json")
            if not side.is_file():
                raise KeyError("Checkpoint thiếu label_to_idx")
            import json

            label2id = json.loads(side.read_text(encoding="utf-8"))
        self.label2id = {str(k): int(v) for k, v in label2id.items()}
        id2 = ckpt.get("id2label")
        self.id2label = (
            {int(k): str(v) for k, v in id2.items()}
            if isinstance(id2, dict) and id2
            else {v: k for k, v in self.label2id.items()}
        )

        self.num_frames = int(cfg.get("num_frames", num_frames))
        self.num_joints = int(cfg.get("num_joints", NUM_JOINTS_V2))
        self.in_channels = int(cfg.get("in_channels", NUM_CHANNELS_V2))
        # Ba kênh chênh nhau chỉ vài mẫu trên 200 nên đáng để chỉnh được ngoài
        # runtime: LT_SIGNDIFF_V2_FUSION="cls,proto,knn", ví dụ "0,0,1" là chỉ
        # dùng kNN thuần.
        w = cfg.get("fusion_weights") or [0.2, 0.2, 0.6]
        override = os.environ.get("LT_SIGNDIFF_V2_FUSION", "").strip()
        if override:
            try:
                parts = [float(x) for x in override.split(",")]
                if len(parts) == 3 and sum(parts) > 0:
                    w = parts
                    print(f"[LT-SignDiff v2] dùng trọng số hợp nhất từ env: {w}")
                else:
                    print(f"[LT-SignDiff v2] bỏ qua LT_SIGNDIFF_V2_FUSION không hợp lệ: {override!r}")
            except ValueError:
                print(f"[LT-SignDiff v2] bỏ qua LT_SIGNDIFF_V2_FUSION không hợp lệ: {override!r}")
        self.w_cls, self.w_proto, self.w_knn = (float(x) for x in w)

        self.model = LTSignDiffV2(
            len(self.label2id),
            num_joints=self.num_joints,
            in_channels=self.in_channels,
            num_frames=self.num_frames,
            temporal_dim=int(cfg.get("temporal_dim", 320)),
            temporal_layers=int(cfg.get("temporal_layers", 3)),
            dropout=0.0,
            drop_path=0.0,
        ).to(self.device)
        missing, unexpected = self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if missing:
            print(f"[LT-SignDiff v2] thiếu {len(missing)} tensor")
        if unexpected:
            print(f"[LT-SignDiff v2] thừa {len(unexpected)} tensor")
        self.model.eval()

        self.gallery_z = None
        self.gallery_y = None
        self.gallery_meta: dict[str, Any] = {}
        self._load_gallery()

        print(
            f"[LT-SignDiff v2] {len(self.label2id)} lớp · T={self.num_frames} · "
            f"w=({self.w_cls},{self.w_proto},{self.w_knn}) · tta={self.use_tta} · "
            f"gallery={0 if self.gallery_z is None else self.gallery_z.size(0)} · {self.device}"
        )

    # ── gallery ────────────────────────────────────────────────────────────
    def _load_gallery(self) -> None:
        z, y, meta = load_gallery(self.ckpt_path, self.device)
        self.gallery_z, self.gallery_y = z, y
        self.gallery_meta = meta or {}
        if self.gallery_z is not None and not bool(self.model.M_ready.item()):
            # Dựng prototype từ chính gallery để kênh proto có tác dụng ngay.
            try:
                self.model.build_prototypes(self.gallery_z, self.gallery_y)
            except Exception as exc:  # noqa: BLE001
                print(f"[LT-SignDiff v2] không dựng được prototype: {exc}")

    def gallery_info(self) -> dict:
        if self.gallery_z is None:
            return {"total_refs": 0, "num_classes_covered": 0, "by_gloss": {}}
        return gallery_stats(self.gallery_z, self.gallery_y, self.id2label)

    # ── đặc trưng ──────────────────────────────────────────────────────────
    def _views(self, sk: np.ndarray) -> list[np.ndarray]:
        """Bản gốc + (nếu bật TTA) lật ngang và hai lát thời gian."""
        sk = np.asarray(sk, dtype=np.float32)
        if sk.ndim != 3:
            raise ValueError(f"Cần [T,V,3], nhận {sk.shape}")
        if sk.shape[1] < 143:
            pad = np.zeros((sk.shape[0], 143 - sk.shape[1], 3), dtype=np.float32)
            sk = np.concatenate([sk, pad], axis=1)
        sk = sk[:, :143]
        mask = mask_from_zeros(sk)

        base = build_features(sk, mask, num_frames=self.num_frames)
        if not self.use_tta:
            return [base]
        return [
            base,
            mirror_features(base),
            build_features(sk, mask, num_frames=self.num_frames, crop=(0.0, 0.88)),
            build_features(sk, mask, num_frames=self.num_frames, crop=(0.12, 1.0)),
        ]

    @torch.no_grad()
    def embed(self, skeleton: np.ndarray) -> torch.Tensor:
        views = self._views(skeleton)
        x = torch.from_numpy(np.stack(views)).to(self.device)
        z = self.model.encode(x)
        return F.normalize(z.mean(dim=0), dim=-1)

    def enroll(self, skeleton: np.ndarray, gloss: str) -> dict:
        gloss = gloss.strip()
        if gloss not in self.label2id:
            raise ValueError(f"Từ '{gloss}' không thuộc bộ {len(self.label2id)} lớp của model")
        cid = int(self.label2id[gloss])
        info = append_embedding(self.ckpt_path, self.embed(skeleton), cid, self.device, source="enroll")
        self._load_gallery()
        return {"gloss": gloss, **info}

    # ── dự đoán ────────────────────────────────────────────────────────────
    def _knn_probs(self, z: torch.Tensor, k: int = 7, temp: float = 12.0) -> torch.Tensor | None:
        if self.gallery_z is None or self.gallery_y is None or self.gallery_z.numel() == 0:
            return None
        sims = z @ self.gallery_z.t()
        k = min(k, sims.numel())
        topv, topi = sims.topk(k)
        scores = torch.zeros(self.model.num_classes, device=self.device)
        weights = torch.exp((topv - topv[0]) * temp)
        scores.index_add_(0, self.gallery_y[topi], weights)
        return scores / scores.sum().clamp(min=1e-8)

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> list[dict]:
        z = self.embed(skeleton)
        scale = self.model.head.scale

        p_cls = F.softmax(scale * self.model.head.cosine(z.unsqueeze(0))[0], dim=-1)
        if bool(self.model.M_ready.item()):
            p_proto = F.softmax(scale * (z @ F.normalize(self.model.M, dim=-1).t()), dim=-1)
        else:
            p_proto = p_cls

        p_knn = self._knn_probs(z)
        w_cls, w_proto, w_knn = self.w_cls, self.w_proto, self.w_knn
        if p_knn is None:
            p_knn, w_knn = torch.zeros_like(p_cls), 0.0
        else:
            # Người dùng enroll càng nhiều mẫu riêng thì càng nên tin kNN.
            n_enroll = sum(int(v) for v in (self.gallery_meta.get("enroll_counts") or {}).values())
            if n_enroll:
                w_knn = min(1.6 * w_knn + 0.02 * min(n_enroll, 25), 1.2)

        total = max(w_cls + w_proto + w_knn, 1e-6)
        probs = (w_cls * p_cls + w_proto * p_proto + w_knn * p_knn) / total

        k = min(self.top_k, probs.numel())
        top_p, top_i = probs.topk(k)
        return [
            {
                "rank": r,
                "gloss": self.id2label.get(int(i), f"id_{i}"),
                "score": round(float(p), 4),
                "confidence_pct": round(float(p) * 100.0, 2),
            }
            for r, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()), start=1)
        ]
