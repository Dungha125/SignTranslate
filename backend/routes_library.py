# -*- coding: utf-8 -*-
"""Kho video từ vựng — người học xem clip mẫu rồi tự làm lại để máy chấm.

  GET    /api/library/overview     số liệu tổng quan + tiến độ của người dùng
  GET    /api/library/words        duyệt/lọc/tìm từ trong bộ từ vựng
  GET    /api/library/word         chi tiết một từ (danh sách clip mẫu)
  POST   /api/library/clips        thêm clip mẫu cho một từ (auth)
  DELETE /api/library/clips/{id}   xoá clip mẫu (auth)
  POST   /api/library/practice     chấm một lượt luyện tập từ webcam (auth)
  DELETE /api/library/progress     xoá tiến độ học của người dùng (auth)

Clip mẫu nằm chung kho với dataset nhưng mang `source="library"` nên không lọt
vào dữ liệu huấn luyện (xem `dataset_store.list_clips` / `export_csv`).

Tiến độ học lưu ở Redis theo tài khoản: `lib:prog:<username>` là dict
gloss → {attempts, passes, best_pct, best_rank, last_at, passed_at}.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

import auth as _auth
import dataset_store as ds
import routes_dataset as _rd
from storage import kv

router = APIRouter()

# Xếp hạng tối đa vẫn được coi là "gần đúng" — trong top-3 nghĩa là động tác đã
# đi đúng hướng, chỉ còn lẫn với từ có hình tay tương tự.
NEAR_RANK = 3
LIBRARY_CLIPS_CAP = 5000
_INDEX_TTL = 60

# main.py nạp hai hàm này vào lúc startup để router không phải import vòng.
_vocab_provider: Callable[[], list[str]] | None = None
_predict: Callable[[list[str]], tuple[list[dict], float]] | None = None


def bind(*, vocab_provider: Callable[[], list[str]], predict: Callable[[list[str]], tuple[list[dict], float]]) -> None:
    global _vocab_provider, _predict
    _vocab_provider = vocab_provider
    _predict = predict


# ─── vocab + chỉ mục clip ───────────────────────────────────────────────────
def _vocab() -> list[str]:
    return _vocab_provider() if _vocab_provider else []


def _library_index() -> dict[str, list[dict]]:
    """gloss → danh sách clip mẫu (mới nhất trước). Cache 60s trong Redis."""
    cached = kv.get_json("lib:index")
    if cached:
        return cached
    page = ds.list_clips(source="library", limit=LIBRARY_CLIPS_CAP)
    out: dict[str, list[dict]] = {}
    for rec in page["clips"]:
        out.setdefault(rec["gloss"], []).append(
            {
                "clip_id": rec["clip_id"],
                "filename": rec.get("filename", ""),
                "size_bytes": rec.get("size_bytes", 0),
                "created_at": rec.get("created_at", 0),
                "has_thumb": bool(rec.get("thumb_key")),
            }
        )
    kv.set_json("lib:index", out, ttl=_INDEX_TTL)
    return out


def _invalidate_index() -> None:
    kv.delete("lib:index")


def _clip_urls(clip: dict) -> dict:
    return {
        **clip,
        "stream_url": f"/api/dataset/clips/{clip['clip_id']}/stream",
        "thumb_url": f"/api/dataset/clips/{clip['clip_id']}/thumb" if clip.get("has_thumb") else None,
    }


# ─── tiến độ ────────────────────────────────────────────────────────────────
def _prog_key(username: str) -> str:
    return f"lib:prog:{username}"


def _progress(username: str) -> dict[str, dict]:
    return kv.get_json(_prog_key(username)) or {}


def _record_attempt(username: str, gloss: str, *, rank: int | None, pct: float) -> dict:
    prog = _progress(username)
    cur = prog.get(gloss) or {"attempts": 0, "passes": 0, "best_pct": 0.0, "best_rank": None, "passed_at": None}
    cur["attempts"] = int(cur["attempts"]) + 1
    if rank == 1:
        cur["passes"] = int(cur["passes"]) + 1
        cur["passed_at"] = cur.get("passed_at") or time.time()
    if pct > float(cur.get("best_pct") or 0):
        cur["best_pct"] = round(pct, 2)
    if rank is not None and (cur.get("best_rank") is None or rank < int(cur["best_rank"])):
        cur["best_rank"] = rank
    cur["last_at"] = time.time()
    prog[gloss] = cur
    kv.set_json(_prog_key(username), prog, ttl=None)
    return cur


# ─── duyệt từ ───────────────────────────────────────────────────────────────
def _word_row(gloss: str, clips: list[dict], prog: dict | None) -> dict:
    first = clips[0] if clips else None
    thumb = next((c for c in clips if c.get("has_thumb")), None)
    return {
        "gloss": gloss,
        "slug": ds.slugify(gloss),
        "clips": len(clips),
        "thumb_url": f"/api/dataset/clips/{thumb['clip_id']}/thumb" if thumb else None,
        "preview_url": f"/api/dataset/clips/{first['clip_id']}/stream" if first else None,
        "attempts": int((prog or {}).get("attempts", 0)),
        "best_pct": float((prog or {}).get("best_pct", 0.0)),
        "best_rank": (prog or {}).get("best_rank"),
        "passed": bool((prog or {}).get("passes", 0)),
    }


@router.get("/api/library/words")
def library_words(
    q: str = "",
    filter: str = Query("all", pattern="^(all|has_video|no_video|learned|learning|todo)$"),
    limit: int = Query(60, ge=1, le=400),
    offset: int = Query(0, ge=0),
    username: str = Depends(_auth.require_auth),
):
    index = _library_index()
    prog = _progress(username)

    # Bộ từ = từ vựng của model, cộng thêm từ chỉ có trong kho clip (nếu người
    # dùng tự thêm clip cho từ ngoài bộ 200).
    glosses = list(dict.fromkeys([*_vocab(), *index.keys()]))

    needle = q.strip().lower()
    rows = []
    for g in glosses:
        if needle and needle not in g.lower():
            continue
        clips = index.get(g, [])
        p = prog.get(g)
        row = _word_row(g, clips, p)
        if filter == "has_video" and not row["clips"]:
            continue
        if filter == "no_video" and row["clips"]:
            continue
        if filter == "learned" and not row["passed"]:
            continue
        if filter == "learning" and (row["passed"] or not row["attempts"]):
            continue
        if filter == "todo" and (row["passed"] or not row["clips"]):
            continue
        rows.append(row)

    # Từ có video lên trước để người mới vào là có thứ để học ngay.
    rows.sort(key=lambda r: (r["clips"] == 0, r["gloss"].lower()))
    return {"total": len(rows), "offset": offset, "limit": limit, "words": rows[offset : offset + limit]}


@router.get("/api/library/word")
def library_word(gloss: str, username: str = Depends(_auth.require_auth)):
    gloss = gloss.strip()
    if not gloss:
        raise HTTPException(422, "Thiếu tên từ")
    index = _library_index()
    clips = index.get(gloss)
    if clips is None and gloss not in _vocab():
        raise HTTPException(404, f"Không có từ “{gloss}” trong kho")
    prog = _progress(username).get(gloss)
    return {
        "gloss": gloss,
        "slug": ds.slugify(gloss),
        "in_vocab": gloss in _vocab(),
        "clips": [_clip_urls(c) for c in (clips or [])],
        "progress": prog,
    }


@router.get("/api/library/overview")
def library_overview(username: str = Depends(_auth.require_auth)):
    index = _library_index()
    vocab = _vocab()
    prog = _progress(username)

    with_video = sum(1 for g in vocab if index.get(g))
    clips = sum(len(v) for v in index.values())
    attempts = sum(int(p.get("attempts", 0)) for p in prog.values())
    passes = sum(int(p.get("passes", 0)) for p in prog.values())
    learned = sum(1 for p in prog.values() if p.get("passes"))
    learning = sum(1 for p in prog.values() if not p.get("passes") and p.get("attempts"))

    recent = sorted(
        ({"gloss": g, **p} for g, p in prog.items() if p.get("last_at")),
        key=lambda r: -float(r["last_at"]),
    )[:8]

    return {
        "words_total": len(vocab),
        "words_with_video": with_video,
        "clips": clips,
        "learned": learned,
        "learning": learning,
        "attempts": attempts,
        "passes": passes,
        "pass_rate_pct": round(100.0 * passes / attempts, 1) if attempts else None,
        "coverage_pct": round(100.0 * with_video / len(vocab), 1) if vocab else 0.0,
        "recent": recent,
    }


# ─── quản lý clip mẫu ───────────────────────────────────────────────────────
@router.post("/api/library/clips")
async def library_add_clip(
    file: UploadFile = File(...),
    gloss: str = Form(...),
    username: str = Depends(_auth.require_auth),
):
    data = await file.read()
    if not data:
        raise HTTPException(422, "File rỗng")
    if len(data) > _rd.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {_rd.MAX_UPLOAD_MB} MB")

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    thumb = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        thumb = _rd._thumbnail_from_video(tmp_path)
    except Exception:  # noqa: BLE001 - thumbnail là phụ, không chặn upload
        thumb = None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    rec = ds.add_clip(
        gloss=gloss,
        data=data,
        filename=file.filename or f"clip{suffix}",
        content_type=file.content_type or "video/mp4",
        split="unassigned",
        source="library",
        uploaded_by=username,
        thumbnail=thumb,
    )
    _invalidate_index()
    return {
        "clip_id": rec["clip_id"],
        "gloss": rec["gloss"],
        "duplicate": bool(rec.get("duplicate")),
        "size_bytes": rec.get("size_bytes", 0),
        "stream_url": f"/api/dataset/clips/{rec['clip_id']}/stream",
    }


@router.delete("/api/library/clips/{clip_id}")
def library_delete_clip(clip_id: str, username: str = Depends(_auth.require_auth)):
    rec = ds.get_clip(clip_id)
    if not rec:
        raise HTTPException(404, "Không tìm thấy clip")
    if rec.get("source") != "library":
        raise HTTPException(400, "Clip này không thuộc kho từ vựng")
    ds.delete_clip(clip_id)
    _invalidate_index()
    return {"ok": True, "clip_id": clip_id}


# ─── luyện tập ──────────────────────────────────────────────────────────────
class PracticeRequest(BaseModel):
    frames_b64: list[str]
    gloss: str


@router.post("/api/library/practice")
def library_practice(req: PracticeRequest, username: str = Depends(_auth.require_auth)):
    if _predict is None:
        raise HTTPException(503, "Model chưa sẵn sàng")
    gloss = req.gloss.strip()
    if not gloss:
        raise HTTPException(422, "Thiếu từ cần luyện")
    if gloss not in _vocab():
        raise HTTPException(400, f"Từ “{gloss}” không nằm trong bộ từ model nhận được")

    preds, elapsed_ms = _predict(req.frames_b64)
    if not preds:
        raise HTTPException(422, "Không tách được bộ xương từ các frame — thử ghi lại với ánh sáng tốt hơn")

    hit = next((p for p in preds if p["gloss"] == gloss), None)
    rank = int(hit["rank"]) if hit else None
    pct = float(hit["confidence_pct"]) if hit else 0.0
    passed = rank == 1
    near = rank is not None and 1 < rank <= NEAR_RANK

    stat = _record_attempt(username, gloss, rank=rank, pct=pct)

    if passed:
        verdict, message = "pass", f"Chính xác — model nhận ra “{gloss}”."
    elif near:
        verdict, message = (
            "near",
            f"Gần đúng: “{gloss}” đứng thứ {rank}. Model đang nghiêng về “{preds[0]['gloss']}”.",
        )
    else:
        verdict, message = (
            "miss",
            f"Chưa đúng — model đọc ra “{preds[0]['gloss']}”. Xem lại clip mẫu và chú ý hình tay.",
        )

    return {
        "gloss": gloss,
        "verdict": verdict,
        "passed": passed,
        "message": message,
        "rank": rank,
        "confidence_pct": round(pct, 2),
        "top_gloss": preds[0]["gloss"],
        "predictions": preds[:5],
        "elapsed_ms": elapsed_ms,
        "progress": stat,
    }


@router.delete("/api/library/progress")
def library_reset_progress(username: str = Depends(_auth.require_auth)):
    kv.delete(_prog_key(username))
    return {"ok": True}
