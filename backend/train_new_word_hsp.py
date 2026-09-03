# -*- coding: utf-8 -*-
"""
Continual learning for HSP-BiMamba (new gloss or new signer, anti-forgetting).

Stages (same UX as CurriVSL learn tab):
  A — intra-class embedding consistency (new signer / environment)
  B — pairwise metric learning on video embeddings
  C — classifier fine-tune + weight reg + teacher KL distillation
"""

from __future__ import annotations

import copy
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
_HSP_ROOT = _REPO / "vsl_hsp_bimamba_k4"
if _HSP_ROOT.is_dir() and str(_HSP_ROOT) not in sys.path:
    sys.path.insert(0, str(_HSP_ROOT))

from hsp_bimamba.build import HSPBiMambaConfig, build_hsp_bimamba  # type: ignore

from train_new_word import TrainingJob, _jobs  # shared job registry

NUM_AUG = 40
BATCH_SZ = 8
EPOCHS_STAGE = 10
LR_STAGE = {"A": 5e-5, "B": 3e-5, "C": 8e-5}
LAMBDA_W_REG = 0.4
LAMBDA_KD = 1.5
KD_TEMP = 2.0


def _resolve_label(label: str, label2id: dict[str, int]) -> tuple[str, int, bool]:
    raw = label.strip()
    if not raw:
        raise ValueError("Nhãn trống")
    if raw in label2id:
        return raw, label2id[raw], False
    low = raw.lower()
    for k, v in label2id.items():
        if k.strip().lower() == low:
            return k, v, False
    new_id = len(label2id)
    label2id[raw] = new_id
    return raw, new_id, True


def _augment_hsp(sk: np.ndarray, n: int = NUM_AUG) -> list[np.ndarray]:
    """[T,75,3] -> n augmented samples."""
    T = sk.shape[0]
    out = [sk.astype(np.float32).copy()]
    rng = np.random.default_rng()
    while len(out) < n:
        s = sk.copy()
        if rng.random() > 0.25:
            new_t = max(8, min(int(T * rng.uniform(0.85, 1.15)), T + 6))
            idx = np.linspace(0, T - 1, new_t, dtype=int)
            s = s[idx]
            if len(s) < T:
                s = np.concatenate([s, np.repeat(s[-1:], T - len(s), axis=0)], axis=0)
            elif len(s) > T:
                s = s[np.linspace(0, len(s) - 1, T, dtype=int)]
        s = s + rng.normal(0, 0.008, s.shape).astype(np.float32)
        if rng.random() > 0.5:
            n_mask = int(rng.integers(1, 4))
            mask_idx = rng.choice(T, min(n_mask, T), replace=False)
            s[mask_idx] = 0.0
        s = s * float(rng.uniform(0.9, 1.1))
        out.append(s.astype(np.float32))
    return out[:n]


def _load_hsp_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    label2id: dict[str, int] = dict(ckpt["label2id"])
    cfg: HSPBiMambaConfig = ckpt.get("cfg") or HSPBiMambaConfig(
        num_classes=len(label2id),
        ablation_level=ckpt.get("ablation", "B5"),
        num_frames=ckpt.get("num_frames", 30),
        temporal_depth=ckpt.get("temporal_depth", 4),
        temporal_loop_steps=ckpt.get("temporal_loop_steps", 4),
        skeleton_layout=ckpt.get("skeleton_layout", "mp75"),
    )
    model = build_hsp_bimamba(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    meta = {
        "ablation": ckpt.get("ablation", cfg.ablation_level),
        "protocol": ckpt.get("protocol", ""),
        "num_frames": ckpt.get("num_frames", 30),
        "hand_rect": ckpt.get("hand_rect", False),
        "skeleton_layout": ckpt.get("skeleton_layout", "mp75"),
        "num_joints": ckpt.get("num_joints", 75),
        "temporal_depth": ckpt.get("temporal_depth", 4),
        "temporal_loop_steps": ckpt.get("temporal_loop_steps", 4),
        "epoch": ckpt.get("epoch", 0),
    }
    return model, label2id, cfg, meta, ckpt


def _expand_cls_head(model: nn.Module, num_classes_new: int, init_row: torch.Tensor | None):
    old = model.cls_head
    d = old.in_features
    new_head = nn.Linear(d, num_classes_new, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new_head.weight[: old.out_features] = old.weight
        new_head.bias[: old.out_features] = old.bias
        if init_row is not None:
            new_head.weight[-1] = init_row
            new_head.bias[-1] = 0.0
        else:
            nn.init.normal_(new_head.weight[-1:], std=0.02)
            nn.init.zeros_(new_head.bias[-1:])
    model.cls_head = new_head


def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _run_training(
    job: TrainingJob,
    skeleton: np.ndarray,
    ckpt_path: Path,
    num_joints: int,
    top_k: int,
    reload_model_cb: Callable,
):
    try:
        job.status = "running"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, label2id, cfg, meta, ckpt = _load_hsp_model(ckpt_path, device)
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        num_classes_old = len(label2id)
        canonical_label, target_id, is_new = _resolve_label(job.label, label2id)
        job.label = canonical_label

        if skeleton.shape[0] != 30 or skeleton.shape[1] != 75:
            raise ValueError(f"Cần skeleton [30,75,3], nhận {skeleton.shape}")

        augmented = _augment_hsp(skeleton, NUM_AUG)
        tensors = [torch.from_numpy(a).unsqueeze(0).to(device) for a in augmented]

        d_fused = model.cls_head.in_features
        proj_v = nn.Linear(d_fused, 256).to(device)

        # ── STAGE A: embedding consistency ───────────────────────────────────
        job.stage = "A"
        job.message = "Stage A — đồng nhất embedding (người ký / môi trường mới)"
        _freeze_all(model)
        for p in proj_v.parameters():
            p.requires_grad = True
        optA = torch.optim.AdamW(proj_v.parameters(), lr=LR_STAGE["A"], weight_decay=1e-4)

        for ep in range(EPOCHS_STAGE):
            model.eval()
            proj_v.train()
            epoch_loss = 0.0
            indices = np.random.permutation(NUM_AUG)
            n_batches = 0
            for i in range(0, NUM_AUG, BATCH_SZ):
                batch_idx = indices[i : i + BATCH_SZ]
                batch = torch.cat([tensors[j] for j in batch_idx], dim=0)
                with torch.no_grad():
                    emb = model(batch).video_emb
                z = F.normalize(proj_v(emb), dim=-1)
                mean_z = F.normalize(z.mean(dim=0, keepdim=True), dim=-1)
                loss = (1.0 - (z * mean_z).sum(dim=-1)).mean()
                optA.zero_grad()
                loss.backward()
                optA.step()
                epoch_loss += loss.item()
                n_batches += 1
            job.epoch = ep + 1
            job.loss = epoch_loss / max(1, n_batches)

        # ── STAGE B: pairwise metric ─────────────────────────────────────────
        job.stage = "B"
        job.message = "Stage B — metric learning cặp augmentation"
        optB = torch.optim.AdamW(proj_v.parameters(), lr=LR_STAGE["B"], weight_decay=1e-4)

        for ep in range(EPOCHS_STAGE):
            model.eval()
            epoch_loss = 0.0
            perm = np.random.permutation(NUM_AUG)
            pairs = [(perm[i], perm[i + 1]) for i in range(0, NUM_AUG - 1, 2)]
            for ai, bi in pairs:
                with torch.no_grad():
                    emb_a = model(tensors[ai]).video_emb
                    emb_b = model(tensors[bi]).video_emb
                za = F.normalize(proj_v(emb_a), dim=-1)
                zb = F.normalize(proj_v(emb_b), dim=-1)
                loss = (1.0 - F.cosine_similarity(za, zb)).mean()
                optB.zero_grad()
                loss.backward()
                optB.step()
                epoch_loss += loss.item()
            job.epoch = EPOCHS_STAGE + ep + 1
            job.loss = epoch_loss / max(1, len(pairs))

        # ── STAGE C: classifier + anti-forgetting ───────────────────────────
        job.stage = "C"
        mode = "từ mới" if is_new else "học lại (người ký mới)"
        job.message = f"Stage C — fine-tune classifier ({mode})"

        if is_new:
            with torch.no_grad():
                emb0 = model(tensors[0]).video_emb[0]
                init_row = F.normalize(emb0, dim=0)
            _expand_cls_head(model, len(label2id), init_row)

        _freeze_all(model)
        for p in model.cls_head.parameters():
            p.requires_grad = True

        old_w = model.cls_head.weight.detach().clone()
        old_b = model.cls_head.bias.detach().clone()
        label_t = torch.tensor([target_id], dtype=torch.long, device=device)
        optC = torch.optim.AdamW(model.cls_head.parameters(), lr=LR_STAGE["C"], weight_decay=1e-4)

        for ep in range(EPOCHS_STAGE):
            model.cls_head.train()
            epoch_loss = 0.0
            indices = np.random.permutation(NUM_AUG)
            n_batches = 0
            for i in range(0, NUM_AUG, BATCH_SZ):
                batch_idx = indices[i : i + BATCH_SZ]
                batch = torch.cat([tensors[j] for j in batch_idx], dim=0)
                logits = model(batch).logits
                labels_batch = label_t.expand(len(batch_idx))
                loss_ce = F.cross_entropy(logits, labels_batch)

                with torch.no_grad():
                    t_logits = teacher(batch).logits[:, :num_classes_old]
                s_logits = logits[:, :num_classes_old]
                loss_kd = F.kl_div(
                    F.log_softmax(s_logits / KD_TEMP, dim=-1),
                    F.softmax(t_logits / KD_TEMP, dim=-1),
                    reduction="batchmean",
                ) * (KD_TEMP ** 2)

                loss_w = F.mse_loss(model.cls_head.weight[:num_classes_old], old_w)
                loss_b = F.mse_loss(model.cls_head.bias[:num_classes_old], old_b)
                loss = loss_ce + LAMBDA_KD * loss_kd + LAMBDA_W_REG * (loss_w + loss_b)

                optC.zero_grad()
                loss.backward()
                optC.step()
                epoch_loss += loss.item()
                n_batches += 1

            job.epoch = 2 * EPOCHS_STAGE + ep + 1
            job.loss = epoch_loss / max(1, n_batches)

        cfg.num_classes = len(label2id)
        new_ckpt = {
            "model": model.state_dict(),
            "label2id": label2id,
            "cfg": cfg,
            **meta,
            "best_val_top1": ckpt.get("best_val_top1", 0.0),
        }
        if ckpt_path.is_file():
            bak = ckpt_path.with_suffix(".pt.bak")
            shutil.copy2(ckpt_path, bak)
        torch.save(new_ckpt, str(ckpt_path))

        job.status = "done"
        job.stage = "done"
        n_cls = len(label2id)
        if is_new:
            job.message = f"Đã thêm từ '{canonical_label}' ({n_cls} lớp). Từ cũ được giữ bằng distillation."
        else:
            job.message = f"Đã cập nhật '{canonical_label}' với người ký/môi trường mới ({n_cls} lớp)."
        job.finished_at = time.time()
        reload_model_cb(job.model_id, ckpt_path, num_joints, top_k)

    except Exception as e:
        import traceback

        job.status = "error"
        job.error = str(e)
        job.message = "Huấn luyện thất bại"
        traceback.print_exc()


def start_training(
    skeleton: np.ndarray,
    label: str,
    model_id: str,
    ckpt_path: Path,
    num_joints: int,
    top_k: int,
    reload_model_cb: Callable,
) -> str:
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
