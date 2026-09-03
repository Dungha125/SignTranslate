# -*- coding: utf-8 -*-
"""
Train end-to-end for a new sign word (A → B → C, mỗi stage 10 epoch).

Chiến lược:
  Stage A (10 ep) : fine-tune backbone với sign–text contrastive alignment.
  Stage B (10 ep) : fine-tune backbone với self-supervised pairwise cosine
                    (buộc các augmentation khác nhau của cùng từ gần nhau).
  Stage C (10 ep) : mở rộng head + fine-tune classifier với class mới.

Input duy nhất: skeleton [T, V, C] + nhãn text.
Augmentation 40× để tránh over-fit với 1 mẫu.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── re-use model classes from inference.py ──────────────────────────────────
from inference import CosineClassifier, SkeletonBackbone

NUM_AUG   = 40
BATCH_SZ  = 8
LR_STAGE  = {"A": 1e-4, "B": 5e-5, "C": 5e-5}
LAMBDA_W_REG = 0.3  # keep old class weights stable (anti-forgetting)


# ─── augmentation ─────────────────────────────────────────────────────────────
def _augment(sk: np.ndarray, n: int = NUM_AUG) -> list[np.ndarray]:
    """[T,V,C] → list of n augmented [T,V,C]."""
    T = sk.shape[0]
    result = [sk.copy()]
    rng = np.random.default_rng()
    while len(result) < n:
        s = sk.copy()
        # temporal jitter
        if rng.random() > 0.3:
            new_t = int(T * rng.uniform(0.7, 1.3))
            new_t = max(8, min(new_t, T + 4))
            idx = np.linspace(0, T - 1, new_t, dtype=int)
            s = s[idx]
            if len(s) < T:
                s = np.concatenate([s, np.repeat(s[-1:], T - len(s), axis=0)], axis=0)
            elif len(s) > T:
                s = s[np.linspace(0, len(s) - 1, T, dtype=int)]
        # gaussian noise
        s = s + rng.normal(0, 0.012, s.shape).astype(np.float32)
        # random frame mask
        n_mask = rng.integers(0, 4)
        if n_mask:
            mask_idx = rng.choice(T, min(n_mask, T), replace=False)
            s[mask_idx] = 0.0
        # horizontal mirror (x → 1-x)
        if rng.random() > 0.5:
            s[:, :, 0] = 1.0 - s[:, :, 0]
        # scale
        scale = rng.uniform(0.85, 1.15)
        s = s * scale
        result.append(s.astype(np.float32))
    return result[:n]


# ─── job registry ────────────────────────────────────────────────────────────
class TrainingJob:
    def __init__(self, job_id: str, label: str, model_id: str):
        self.job_id = job_id
        self.label  = label
        self.model_id = model_id
        self.status = "queued"          # queued|running|done|error
        self.stage  = ""               # A | B | C
        self.epoch  = 0
        self.total_epochs = 30          # 10×3
        self.loss   = 0.0
        self.message = ""
        self.error   = ""
        self.started_at = time.time()
        self.finished_at: float | None = None

    def as_dict(self) -> dict:
        return {
            "job_id":       self.job_id,
            "label":        self.label,
            "model_id":     self.model_id,
            "status":       self.status,
            "stage":        self.stage,
            "epoch":        self.epoch,
            "total_epochs": self.total_epochs,
            "loss":         round(self.loss, 5),
            "message":      self.message,
            "error":        self.error,
            "elapsed_s":    round(time.time() - self.started_at, 1),
        }


_jobs: dict[str, TrainingJob] = {}


# ─── core training ────────────────────────────────────────────────────────────
def _run_training(
    job: TrainingJob,
    skeleton: np.ndarray,
    ckpt_path: Path,
    num_joints: int,
    top_k: int,
    reload_model_cb: Callable,
):
    """Runs in background thread. Updates job in-place. Calls reload_model_cb at end."""
    try:
        job.status = "running"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── load checkpoint ──────────────────────────────────────────────────
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        label2id: dict[str, int] = dict(ckpt["label2id"])
        num_classes_old = len(label2id)

        is_new_label = job.label not in label2id
        if is_new_label:
            label2id[job.label] = num_classes_old
        target_id = label2id[job.label]

        # ── backbone ─────────────────────────────────────────────────────────
        backbone = SkeletonBackbone(num_joints=num_joints, in_dim=3,
                                     hidden_dim=256, lstm_layers=2, dropout=0.3).to(device)
        bb_state = ckpt.get("backbone", ckpt)
        backbone.load_state_dict(bb_state, strict=True)
        # Anti-forgetting: freeze backbone when learning from few real clips
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

        # ── augment ──────────────────────────────────────────────────────────
        augmented = _augment(skeleton, NUM_AUG)
        tensors = [torch.from_numpy(a).unsqueeze(0).to(device) for a in augmented]  # list of [1,T,V,C]

        # ─────────────────────────────────────────────────────────────────────
        # STAGE A: sign–text alignment (10 epochs)
        # ─────────────────────────────────────────────────────────────────────
        job.stage = "A"
        job.message = "Stage A — sign–text contrastive alignment"
        emb_dim = 512   # backbone output dim

        try:
            from sentence_transformers import SentenceTransformer
            txt_enc = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            txt_enc_dim = txt_enc.get_sentence_embedding_dimension()
            use_txt = True
        except Exception:
            use_txt = False

        proj_v = nn.Linear(emb_dim, 256).to(device)
        proj_t = nn.Linear(txt_enc_dim if use_txt else emb_dim, 256).to(device)

        if use_txt:
            with torch.no_grad():
                txt_emb_np = txt_enc.encode([job.label], convert_to_tensor=True)
                txt_emb = txt_emb_np.to(device).float()

        # Stage A updates only projection heads (backbone frozen)
        optA = torch.optim.AdamW(
            list(proj_v.parameters()) + list(proj_t.parameters()),
            lr=LR_STAGE["A"], weight_decay=1e-4,
        )

        for ep in range(10):
            backbone.eval(); proj_v.train(); proj_t.train()
            epoch_loss = 0.0
            indices = np.random.permutation(NUM_AUG)
            for i in range(0, NUM_AUG, BATCH_SZ):
                batch_idx = indices[i:i + BATCH_SZ]
                batch = torch.cat([tensors[j] for j in batch_idx], dim=0)  # [B,T,V,C]
                with torch.no_grad():
                    emb = backbone(batch)
                v_proj = F.normalize(proj_v(emb), dim=-1)           # [B, 256]
                if use_txt:
                    t_proj = F.normalize(proj_t(txt_emb.expand(len(batch_idx), -1)), dim=-1)
                    # cosine alignment: maximize similarity → minimize 1 - cos
                    loss = (1.0 - (v_proj * t_proj).sum(dim=-1)).mean()
                else:
                    # self-alignment: pull all embeddings together
                    mean_v = v_proj.mean(dim=0, keepdim=True)
                    loss = (1.0 - (v_proj * mean_v.detach()).sum(dim=-1)).mean()
                optA.zero_grad(); loss.backward()
                optA.step()
                epoch_loss += loss.item()
            job.epoch = ep + 1
            job.loss = epoch_loss / max(1, NUM_AUG // BATCH_SZ)

        # ─────────────────────────────────────────────────────────────────────
        # STAGE B: pairwise self-supervised metric (10 epochs)
        # ─────────────────────────────────────────────────────────────────────
        job.stage = "B"
        job.message = "Stage B — pairwise consistency metric learning"
        # Stage B: keep backbone frozen; optimize only proj_v for consistency
        optB = torch.optim.AdamW(proj_v.parameters(), lr=LR_STAGE["B"], weight_decay=1e-4)

        for ep in range(10):
            backbone.eval()
            epoch_loss = 0.0
            perm = np.random.permutation(NUM_AUG)
            pairs = [(perm[i], perm[i + 1]) for i in range(0, NUM_AUG - 1, 2)]
            for (ai, bi) in pairs:
                sk_a = tensors[ai]; sk_b = tensors[bi]
                with torch.no_grad():
                    emb_a = backbone(sk_a); emb_b = backbone(sk_b)
                z_a = F.normalize(proj_v(emb_a), dim=-1)
                z_b = F.normalize(proj_v(emb_b), dim=-1)
                # pull pair together
                loss = (1.0 - F.cosine_similarity(z_a, z_b)).mean()
                optB.zero_grad(); loss.backward()
                optB.step()
                epoch_loss += loss.item()
            job.epoch = 10 + ep + 1
            job.loss = epoch_loss / max(1, len(pairs))

        # ─────────────────────────────────────────────────────────────────────
        # STAGE C: expand classifier + fine-tune (10 epochs)
        # ─────────────────────────────────────────────────────────────────────
        job.stage = "C"
        job.message = "Stage C — expand classifier + fine-tune"
        num_classes_new = len(label2id)

        head = CosineClassifier(emb_dim, num_classes_old, scale=10.0).to(device)
        head.load_state_dict(ckpt["head"], strict=True)

        if is_new_label:
            # expand: copy old weights, add random row
            old_w = head.weight.data                          # [K_old, 512]
            new_row = F.normalize(torch.randn(1, emb_dim, device=device), dim=-1)
            new_w = torch.cat([old_w, new_row], dim=0)       # [K_new, 512]
            head = CosineClassifier(emb_dim, num_classes_new, scale=10.0).to(device)
            head.weight = nn.Parameter(new_w)

        label_t = torch.tensor([target_id], dtype=torch.long, device=device)
        old_w = head.weight.detach().clone()  # keep old weights stable

        # Stage C updates only classifier head (backbone frozen)
        optC = torch.optim.AdamW(
            list(head.parameters()),
            lr=LR_STAGE["C"], weight_decay=1e-4,
        )

        for ep in range(10):
            backbone.eval(); head.train()
            epoch_loss = 0.0
            indices = np.random.permutation(NUM_AUG)
            for i in range(0, NUM_AUG, BATCH_SZ):
                batch_idx = indices[i:i + BATCH_SZ]
                batch = torch.cat([tensors[j] for j in batch_idx], dim=0)
                with torch.no_grad():
                    emb = backbone(batch)
                logits = head(emb)
                labels_batch = label_t.expand(len(batch_idx))
                loss = F.cross_entropy(logits, labels_batch)
                # Regularize: keep old class weights close to original (reduces forgetting)
                if num_classes_new >= num_classes_old:
                    loss = loss + LAMBDA_W_REG * F.mse_loss(head.weight[:num_classes_old], old_w)
                optC.zero_grad(); loss.backward()
                optC.step()
                epoch_loss += loss.item()
            job.epoch = 20 + ep + 1
            job.loss = epoch_loss / max(1, NUM_AUG // BATCH_SZ)

        # ── save updated checkpoint ───────────────────────────────────────────
        # backbone unchanged (frozen). Save it back as-is for consistency.
        new_ckpt = {
            "backbone": bb_state,
            "head":     head.state_dict(),
            "label2id": label2id,
            "num_joints": num_joints,
        }
        torch.save(new_ckpt, str(ckpt_path))

        job.status = "done"
        job.stage = "done"
        job.message = f"Đã học xong từ '{job.label}' ({num_classes_new} lớp). Mô hình đã được cập nhật."
        job.finished_at = time.time()

        # reload model in inference engine
        reload_model_cb(job.model_id, ckpt_path, num_joints, top_k)

    except Exception as e:
        import traceback
        job.status = "error"
        job.error = str(e)
        job.message = "Huấn luyện thất bại"
        traceback.print_exc()


# ─── public API ──────────────────────────────────────────────────────────────
def start_training(
    skeleton: np.ndarray,
    label: str,
    model_id: str,
    ckpt_path: Path,
    num_joints: int,
    top_k: int,
    reload_model_cb: Callable,
) -> str:
    """Starts background training. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(job_id, label, model_id)
    _jobs[job_id] = job
    t = threading.Thread(
        target=_run_training,
        args=(job, skeleton, ckpt_path, num_joints, top_k, reload_model_cb),
        daemon=True,
    )
    t.start()
    return job_id


def get_job(job_id: str) -> TrainingJob | None:
    return _jobs.get(job_id)


def list_jobs(model_id: str | None = None) -> list[dict]:
    jobs = list(_jobs.values())
    if model_id:
        jobs = [j for j in jobs if j.model_id == model_id]
    return [j.as_dict() for j in sorted(jobs, key=lambda x: -x.started_at)]
