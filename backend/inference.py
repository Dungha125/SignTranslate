# -*- coding: utf-8 -*-
"""CurriVSL 110 inference: skeleton [T,110,3] → top-K gloss predictions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── model (self-contained copy, no import from Model_full) ───────────────────

class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(in_dim, out_dim, (kernel_size, 1), padding=(p, 0))
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(self.act(self.bn(self.conv(x))))
        return x.permute(0, 2, 3, 1)


class SkeletonBackbone(nn.Module):
    def __init__(self, num_joints: int = 110, in_dim: int = 3, hidden_dim: int = 256,
                 lstm_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_joints, self.in_dim = num_joints, in_dim
        self.temporal1 = TemporalConvBlock(in_dim, hidden_dim // 2, 5, dropout)
        self.temporal2 = TemporalConvBlock(hidden_dim // 2, hidden_dim // 2, 3, dropout)
        din = (hidden_dim // 2) * num_joints
        self.input_ln = nn.LayerNorm(din)
        self.lstm = nn.LSTM(din, hidden_dim, lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2) * 0.02)
        self.out_ln = nn.LayerNorm(hidden_dim * 2)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        B, T, V, C = skeleton.shape
        if V > self.num_joints:
            skeleton = skeleton[:, :, :self.num_joints, :self.in_dim]
        elif V < self.num_joints:
            pad = torch.zeros(B, T, self.num_joints - V, C, device=skeleton.device)
            skeleton = torch.cat([skeleton, pad], dim=2)
        x = self.temporal2(self.temporal1(skeleton))
        B, T, V, C1 = x.shape
        x = self.input_ln(x.reshape(B, T, V * C1))
        x = self.out_ln(self.dropout(self.lstm(x)[0]))
        q = self.attn_query.expand(B, -1, -1)
        attn = F.softmax(torch.matmul(q, x.transpose(1, 2)) / math.sqrt(x.size(-1)), dim=-1)
        return torch.matmul(attn, x).squeeze(1)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.matmul(F.normalize(x, dim=-1),
                                         F.normalize(self.weight, dim=-1).t())


# ─── loader & inference ───────────────────────────────────────────────────────

class CurriVSLModel:
    def __init__(self, ckpt_path: str | Path, num_joints: int = 110, top_k: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_joints = num_joints
        self.top_k = top_k

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        label2id: dict[str, int] = ckpt["label2id"]
        self.id2label: dict[int, str] = {v: k for k, v in label2id.items()}

        self.backbone = SkeletonBackbone(num_joints=num_joints, in_dim=3,
                                          hidden_dim=256, lstm_layers=2, dropout=0.0).to(self.device)
        self.head = CosineClassifier(512, len(label2id), scale=10.0).to(self.device)

        bb_state = ckpt.get("backbone", ckpt)
        self.backbone.load_state_dict(bb_state, strict=True)
        self.head.load_state_dict(ckpt["head"], strict=True)
        self.backbone.eval()
        self.head.eval()

        print(f"[CurriVSL] Loaded {len(label2id)} classes from {ckpt_path} on {self.device}")

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> list[dict]:
        """
        skeleton: [T, V, C] numpy float32
        Returns list of {rank, gloss, score, confidence_pct}
        """
        sk = torch.from_numpy(skeleton).unsqueeze(0).to(self.device)  # [1, T, V, C]
        emb = self.backbone(sk)
        logits = self.head(emb)[0]                                      # [K]
        probs = F.softmax(logits, dim=-1)
        k = min(self.top_k, logits.size(-1))
        top_probs, top_ids = probs.topk(k)
        results = []
        for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
            results.append({
                "rank": rank,
                "gloss": self.id2label.get(idx, f"id_{idx}"),
                "score": round(logits[idx].item(), 4),
                "confidence_pct": round(prob * 100, 2),
            })
        return results
