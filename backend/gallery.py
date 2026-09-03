# -*- coding: utf-8 -*-
"""Persistent embedding gallery for LT-SignDiff kNN inference."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


def gallery_paths(ckpt_path: str | Path) -> tuple[Path, Path]:
    ckpt_path = Path(ckpt_path)
    return ckpt_path.with_name("gallery.pt"), ckpt_path.with_name("gallery_meta.json")


def load_gallery(ckpt_path: str | Path, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
    gal_path, meta_path = gallery_paths(ckpt_path)
    if not gal_path.is_file():
        return None, None, {}
    data = torch.load(str(gal_path), map_location=device, weights_only=False)
    z = F.normalize(data["z"].to(device).float(), dim=-1)
    y = data["y"].to(device).long()
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return z, y, meta


def save_gallery(
    ckpt_path: str | Path,
    z: torch.Tensor,
    y: torch.Tensor,
    meta: dict | None = None,
) -> Path:
    gal_path, meta_path = gallery_paths(ckpt_path)
    gal_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "y": y.cpu()}, gal_path)
    if meta is not None:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return gal_path


def gallery_stats(z: torch.Tensor, y: torch.Tensor, id2label: dict[int, str]) -> dict:
    counts = Counter(int(i) for i in y.tolist())
    by_gloss = {id2label.get(cid, str(cid)): int(n) for cid, n in counts.items()}
    return {
        "total_refs": int(y.numel()),
        "num_classes_covered": len(counts),
        "by_gloss": dict(sorted(by_gloss.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def append_embedding(
    ckpt_path: str | Path,
    z_new: torch.Tensor,
    class_id: int,
    device: torch.device,
    *,
    source: str = "enroll",
    max_per_class: int = 12,
) -> dict:
    """Add one embedding; cap per-class refs to avoid dominance."""
    z_new = F.normalize(z_new.view(-1).to(device).float(), dim=0)
    gal_path, meta_path = gallery_paths(ckpt_path)
    z, y, meta = load_gallery(ckpt_path, device)
    if z is None:
        z = z_new.unsqueeze(0)
        y = torch.tensor([class_id], device=device, dtype=torch.long)
    else:
        mask = y == class_id
        if mask.sum() >= max_per_class:
            # drop oldest ref for this class
            idx = (y == class_id).nonzero(as_tuple=False)[0, 0].item()
            keep = torch.ones(y.size(0), dtype=torch.bool, device=device)
            keep[idx] = False
            z, y = z[keep], y[keep]
        z = torch.cat([z, z_new.unsqueeze(0)], dim=0)
        y = torch.cat([y, torch.tensor([class_id], device=device, dtype=torch.long)], dim=0)

    enroll = meta.get("enroll_counts", {})
    enroll[str(class_id)] = int(enroll.get(str(class_id), 0)) + 1
    meta["enroll_counts"] = enroll
    meta["last_source"] = source
    save_gallery(ckpt_path, z, y, meta)
    return {"total_refs": int(y.numel()), "class_id": class_id}
