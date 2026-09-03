# -*- coding: utf-8 -*-
"""Kho dữ liệu VSL: metadata trong Redis, video trong MinIO.

Mô hình dữ liệu
---------------
st:ds:clip:<clip_id>       JSON  — một bản ghi clip
st:ds:gloss:<gloss>        SET   — clip_id thuộc gloss đó
st:ds:glosses              SET   — toàn bộ gloss có trong kho
st:ds:index                LIST  — clip_id mới nhất trước (để duyệt nhanh)
st:ds:stats                JSON  — số liệu tổng hợp (cache 60s)

Object key trong bucket `vsl-dataset`:  clips/<gloss_slug>/<clip_id>.mp4
Thumbnail trong bucket `vsl-thumbs`:    thumbs/<clip_id>.jpg

Metadata luôn ghi kèm một bản sao JSONL trên đĩa (`_localstore/dataset_index.jsonl`)
để kho không biến mất khi Redis chạy chế độ in-memory hoặc bị flush.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable

import storage
from storage import kv, objects

INDEX_KEY = "ds:index"
GLOSSES_KEY = "ds:glosses"
STATS_KEY = "ds:stats"
CLIP_TTL = None          # metadata clip: không hết hạn
INDEX_CAP = 20000

_JOURNAL = storage.LOCAL_FALLBACK_DIR / "dataset_index.jsonl"

SPLITS = ("train", "val", "test", "unassigned")
SOURCES = ("upload", "webcam", "corpus", "enroll")

# Số clip tối thiểu nên có cho mỗi từ. Đo được từ tập Top-200: độ chính xác
# nhảy bậc từ ~69 % (≤3 clip) lên 97 % (≥4 clip).
MIN_CLIPS_PER_GLOSS = 4


# ─── helpers ────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    """`chào bạn` → `chao-ban` (an toàn cho object key S3)."""
    norm = unicodedata.normalize("NFD", str(text))
    ascii_ = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    ascii_ = ascii_.replace("đ", "d").replace("Đ", "D")
    ascii_ = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_).strip("-").lower()
    return ascii_ or "unknown"


def _clip_key(clip_id: str) -> str:
    return f"ds:clip:{clip_id}"


def _gloss_key(gloss: str) -> str:
    return f"ds:gloss:{gloss}"


def _journal_append(record: dict) -> None:
    try:
        _JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with open(_JOURNAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - journal chỉ là lưới an toàn
        pass


def restore_from_journal() -> int:
    """Nạp lại metadata từ JSONL khi Redis trống (chạy lúc startup)."""
    if not _JOURNAL.is_file():
        return 0
    if kv.get_json(INDEX_KEY + ":restored"):
        return 0
    latest: dict[str, dict] = {}
    try:
        with open(_JOURNAL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if rec.get("deleted"):
                    latest.pop(rec.get("clip_id"), None)
                elif rec.get("clip_id"):
                    latest[rec["clip_id"]] = rec
    except Exception:  # noqa: BLE001
        return 0

    n = 0
    for clip_id, rec in sorted(latest.items(), key=lambda kv_: kv_[1].get("created_at", 0)):
        if kv.get_json(_clip_key(clip_id)):
            continue
        kv.set_json(_clip_key(clip_id), rec, ttl=CLIP_TTL)
        kv.sadd(_gloss_key(rec.get("gloss", "")), clip_id)
        kv.sadd(GLOSSES_KEY, rec.get("gloss", ""))
        kv.push_history(INDEX_KEY, clip_id, cap=INDEX_CAP)
        n += 1
    kv.set_json(INDEX_KEY + ":restored", {"n": n, "at": time.time()}, ttl=None)
    return n


# ─── ghi ────────────────────────────────────────────────────────────────────
def add_clip(
    *,
    gloss: str,
    data: bytes,
    filename: str = "",
    content_type: str = "video/mp4",
    split: str = "unassigned",
    source: str = "upload",
    uploaded_by: str = "anonymous",
    thumbnail: bytes | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Đẩy video vào MinIO + ghi metadata. Trả về bản ghi clip."""
    gloss = str(gloss).strip()
    if not gloss:
        raise ValueError("Thiếu gloss")
    if split not in SPLITS:
        split = "unassigned"
    if source not in SOURCES:
        source = "upload"

    digest = storage.content_hash(data)
    dup = find_by_hash(digest)
    if dup:
        return {**dup, "duplicate": True}

    clip_id = uuid.uuid4().hex[:16]
    ext = (Path(filename).suffix or ".mp4").lower()
    if ext not in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
        ext = ".mp4"
    obj_key = f"clips/{slugify(gloss)}/{clip_id}{ext}"

    put = objects.put_bytes(
        "dataset",
        obj_key,
        data,
        content_type=content_type,
        metadata={"gloss-slug": slugify(gloss), "clip-id": clip_id},
    )

    thumb_key = None
    if thumbnail:
        thumb_key = f"thumbs/{clip_id}.jpg"
        objects.put_bytes("thumbs", thumb_key, thumbnail, content_type="image/jpeg")

    record = {
        "clip_id": clip_id,
        "gloss": gloss,
        "gloss_slug": slugify(gloss),
        "object_key": obj_key,
        "thumb_key": thumb_key,
        "bucket": put["bucket"],
        "storage_backend": put["backend"],
        "size_bytes": put["size"],
        "content_type": content_type,
        "filename": filename or f"{clip_id}{ext}",
        "split": split,
        "source": source,
        "uploaded_by": uploaded_by,
        "sha": digest,
        "created_at": time.time(),
        **(extra or {}),
    }

    kv.set_json(_clip_key(clip_id), record, ttl=CLIP_TTL)
    kv.set_json(f"ds:sha:{digest}", clip_id, ttl=CLIP_TTL)
    kv.sadd(_gloss_key(gloss), clip_id)
    kv.sadd(GLOSSES_KEY, gloss)
    kv.push_history(INDEX_KEY, clip_id, cap=INDEX_CAP)
    kv.delete(STATS_KEY)
    _journal_append(record)
    return record


def update_clip(clip_id: str, **fields) -> dict | None:
    rec = get_clip(clip_id)
    if not rec:
        return None
    old_gloss = rec.get("gloss")
    allowed = {"gloss", "split", "note", "verified", "source"}
    for k, v in fields.items():
        if k in allowed and v is not None:
            rec[k] = v
    if "gloss" in fields and fields["gloss"] and fields["gloss"] != old_gloss:
        rec["gloss_slug"] = slugify(rec["gloss"])
        kv.srem(_gloss_key(old_gloss), clip_id)
        kv.sadd(_gloss_key(rec["gloss"]), clip_id)
        kv.sadd(GLOSSES_KEY, rec["gloss"])
    rec["updated_at"] = time.time()
    kv.set_json(_clip_key(clip_id), rec, ttl=CLIP_TTL)
    kv.delete(STATS_KEY)
    _journal_append(rec)
    return rec


def delete_clip(clip_id: str) -> bool:
    rec = get_clip(clip_id)
    if not rec:
        return False
    objects.remove("dataset", rec["object_key"])
    if rec.get("thumb_key"):
        objects.remove("thumbs", rec["thumb_key"])
    kv.srem(_gloss_key(rec.get("gloss", "")), clip_id)
    kv.delete(_clip_key(clip_id), f"ds:sha:{rec.get('sha','')}", STATS_KEY)
    _journal_append({"clip_id": clip_id, "deleted": True, "created_at": time.time()})
    return True


# ─── đọc ────────────────────────────────────────────────────────────────────
def get_clip(clip_id: str) -> dict | None:
    return kv.get_json(_clip_key(clip_id))


def get_clips(clip_ids: list[str]) -> list[dict]:
    """Đọc nhiều clip trong một vòng, bỏ qua id đã biến mất."""
    recs = kv.get_many([_clip_key(c) for c in clip_ids])
    return [r for r in recs if r]


def find_by_hash(digest: str) -> dict | None:
    clip_id = kv.get_json(f"ds:sha:{digest}")
    return get_clip(clip_id) if clip_id else None


def list_clips(
    *,
    gloss: str | None = None,
    split: str | None = None,
    source: str | None = None,
    query: str = "",
    limit: int = 60,
    offset: int = 0,
) -> dict:
    if gloss:
        ids: Iterable[str] = sorted(kv.smembers(_gloss_key(gloss)))
    else:
        ids = kv.history(INDEX_KEY, limit=INDEX_CAP)

    q = query.strip().lower()
    rows: list[dict] = []
    for rec in get_clips(list(dict.fromkeys(ids))):
        if split and rec.get("split") != split:
            continue
        if source and rec.get("source") != source:
            continue
        if q and q not in str(rec.get("gloss", "")).lower() and q not in str(rec.get("filename", "")).lower():
            continue
        rows.append(rec)

    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "clips": page}


def list_glosses() -> list[dict]:
    out = []
    for g in kv.smembers(GLOSSES_KEY):
        ids = kv.smembers(_gloss_key(g))
        if not ids:
            continue
        out.append({"gloss": g, "clips": len(ids)})
    out.sort(key=lambda r: (-r["clips"], r["gloss"]))
    return out


def stats(ttl: int = 60) -> dict:
    cached = kv.get_json(STATS_KEY)
    if cached:
        return cached

    ids = kv.history(INDEX_KEY, limit=INDEX_CAP)
    by_split = {s: 0 for s in SPLITS}
    by_source = {s: 0 for s in SOURCES}
    per_gloss: dict[str, int] = {}
    total_bytes = 0
    n = 0
    for rec in get_clips(list(dict.fromkeys(ids))):
        n += 1
        total_bytes += int(rec.get("size_bytes") or 0)
        by_split[rec.get("split", "unassigned")] = by_split.get(rec.get("split", "unassigned"), 0) + 1
        by_source[rec.get("source", "upload")] = by_source.get(rec.get("source", "upload"), 0) + 1
        g = rec.get("gloss", "?")
        per_gloss[g] = per_gloss.get(g, 0) + 1

    counts = sorted(per_gloss.values())
    out = {
        "clips": n,
        "glosses": len(per_gloss),
        "total_bytes": total_bytes,
        "by_split": by_split,
        "by_source": by_source,
        "clips_per_gloss_avg": round(n / max(len(per_gloss), 1), 2),
        "clips_per_gloss_min": counts[0] if counts else 0,
        "clips_per_gloss_max": counts[-1] if counts else 0,
        "top_glosses": sorted(per_gloss.items(), key=lambda kv_: -kv_[1])[:12],
        # Ngưỡng 4 không phải con số tuỳ tiện: đo trên tập test Top-200 cho thấy
        # từ có ≥4 clip đạt 97,2 % còn từ có ≤3 clip chỉ 69 % — bậc thang rất rõ
        # (xem scripts/analyze_errors_v2.py). Đây chính là các từ nên quay thêm.
        "target_clips_per_gloss": MIN_CLIPS_PER_GLOSS,
        "needs_more": sorted([g for g, c in per_gloss.items() if c < MIN_CLIPS_PER_GLOSS])[:24],
        "needs_more_total": sum(1 for c in per_gloss.values() if c < MIN_CLIPS_PER_GLOSS),
        "storage": objects.info()["backend"],
        "generated_at": time.time(),
    }
    kv.set_json(STATS_KEY, out, ttl=ttl)
    return out


def export_csv() -> str:
    """Xuất split CSV giống định dạng LT_SignDiff (video,label,split)."""
    lines = ["video,label,split"]
    ids = list(dict.fromkeys(kv.history(INDEX_KEY, limit=INDEX_CAP)))
    for rec in get_clips(ids):
        name = str(rec.get("filename", rec.get("clip_id", ""))).replace(",", "_")
        label = str(rec.get("gloss", "")).replace(",", " ")
        lines.append(f"{name},{label},{rec.get('split', 'unassigned')}")
    return "\n".join(lines) + "\n"


def import_corpus(
    label_csv: str | Path,
    videos_root: str | Path,
    *,
    glosses: list[str] | None = None,
    limit_per_gloss: int = 3,
    max_clips: int = 400,
) -> dict:
    """Nạp một phần corpus VSL sẵn có trên đĩa vào MinIO để duyệt trong UI."""
    import csv as _csv
    from collections import defaultdict

    label_csv, videos_root = Path(label_csv), Path(videos_root)
    if not label_csv.is_file():
        return {"error": f"Không thấy {label_csv}", "imported": 0}

    wanted = set(glosses) if glosses else None
    by_label: dict[str, list[str]] = defaultdict(list)
    with open(label_csv, encoding="utf-8-sig", newline="") as f:
        for row in _csv.DictReader(f):
            lab = str(row.get("LABEL") or row.get("label") or "").strip()
            vid = str(row.get("VIDEO") or row.get("video") or "").strip()
            if not lab or not vid:
                continue
            if wanted and lab not in wanted:
                continue
            if len(by_label[lab]) < limit_per_gloss:
                by_label[lab].append(vid)

    imported = skipped = failed = 0
    for lab, vids in by_label.items():
        for i, vid in enumerate(vids):
            if imported >= max_clips:
                break
            src = videos_root / vid
            if not src.is_file():
                failed += 1
                continue
            try:
                data = src.read_bytes()
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            rec = add_clip(
                gloss=lab,
                data=data,
                filename=vid,
                split="train" if i == 0 else "unassigned",
                source="corpus",
                uploaded_by="corpus-import",
            )
            if rec.get("duplicate"):
                skipped += 1
            else:
                imported += 1
    kv.delete(STATS_KEY)
    return {"imported": imported, "skipped_duplicate": skipped, "failed": failed, "glosses": len(by_label)}
