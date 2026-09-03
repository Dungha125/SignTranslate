# -*- coding: utf-8 -*-
"""Lịch sử dịch, phản hồi người dùng và bảng số liệu — tất cả lưu trong Redis.

  GET    /api/history                   lịch sử dịch gần đây
  DELETE /api/history                   xoá lịch sử (auth)
  POST   /api/history/{id}/feedback     đánh dấu đúng/sai + gloss thật (auth)
  GET    /api/insights/overview         số liệu tổng quan cho dashboard
  GET    /api/insights/hard-cases       các gloss bị nhận sai nhiều nhất
  GET    /api/sentence                  câu đang dựng trong phiên
  POST   /api/sentence/append           thêm gloss vào câu
  POST   /api/sentence/reset            xoá câu
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import auth as _auth
from storage import kv

router = APIRouter()

HISTORY_KEY = "hist:all"
HISTORY_CAP = 500
STATS_TOTAL = "stat:total"
STATS_LATENCY = "stat:latency"     # danh sách ms gần đây
STATS_GLOSS = "stat:gloss"         # dict gloss -> count
STATS_MODEL = "stat:model"         # dict model -> {n, ms_sum}
FEEDBACK_KEY = "fb:index"
CONFUSION_KEY = "fb:confusion"     # dict "pred>true" -> count


# ═══════════════════════════════════════════════════════════════════════════
# Ghi nhận (gọi từ main.py sau mỗi lần dịch)
# ═══════════════════════════════════════════════════════════════════════════
def record_translation(
    *,
    model_id: str,
    display_name: str,
    predictions: list[dict],
    elapsed_ms: float,
    source: str = "upload",
    username: str = "anonymous",
    filename: str = "",
    cached: bool = False,
) -> str:
    """Đẩy một lượt dịch vào lịch sử + cập nhật số liệu. Trả về entry id."""
    entry_id = uuid.uuid4().hex[:12]
    top = predictions[0] if predictions else {}
    entry = {
        "id": entry_id,
        "model_id": model_id,
        "display_name": display_name,
        "source": source,
        "username": username,
        "filename": filename,
        "top_gloss": top.get("gloss", ""),
        "top_score": top.get("score", 0.0),
        "predictions": predictions[:5],
        "elapsed_ms": round(float(elapsed_ms), 1),
        "cached": cached,
        "created_at": time.time(),
        "feedback": None,
    }
    kv.push_history(HISTORY_KEY, entry, cap=HISTORY_CAP)
    kv.set_json(f"hist:e:{entry_id}", entry, ttl=60 * 60 * 24 * 7)

    kv.incr(STATS_TOTAL)
    lat = kv.get_json(STATS_LATENCY) or []
    lat.insert(0, round(float(elapsed_ms), 1))
    kv.set_json(STATS_LATENCY, lat[:300], ttl=None)

    if top.get("gloss"):
        counts = kv.get_json(STATS_GLOSS) or {}
        counts[top["gloss"]] = int(counts.get(top["gloss"], 0)) + 1
        kv.set_json(STATS_GLOSS, counts, ttl=None)

    models = kv.get_json(STATS_MODEL) or {}
    m = models.get(model_id) or {"n": 0, "ms_sum": 0.0, "display_name": display_name}
    m["n"] = int(m["n"]) + 1
    m["ms_sum"] = float(m["ms_sum"]) + float(elapsed_ms)
    m["display_name"] = display_name
    models[model_id] = m
    kv.set_json(STATS_MODEL, models, ttl=None)
    return entry_id


# ═══════════════════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════════════════
@router.get("/api/history")
def get_history(limit: int = Query(40, ge=1, le=200), model_id: str | None = None):
    rows = kv.history(HISTORY_KEY, limit=HISTORY_CAP)
    merged = []
    for r in rows:
        latest = kv.get_json(f"hist:e:{r.get('id')}")
        merged.append(latest or r)
    if model_id:
        merged = [r for r in merged if r.get("model_id") == model_id]
    return {"total": len(merged), "entries": merged[:limit]}


@router.delete("/api/history")
def clear_history(username: str = Depends(_auth.require_auth)):
    for r in kv.history(HISTORY_KEY, limit=HISTORY_CAP):
        kv.delete(f"hist:e:{r.get('id')}")
    kv.delete(HISTORY_KEY)
    return {"ok": True}


class FeedbackRequest(BaseModel):
    correct: bool
    true_gloss: str | None = None


@router.post("/api/history/{entry_id}/feedback")
def post_feedback(
    entry_id: str,
    body: FeedbackRequest,
    username: str = Depends(_auth.require_auth),
):
    entry = kv.get_json(f"hist:e:{entry_id}")
    if not entry:
        raise HTTPException(404, "Không tìm thấy lượt dịch (có thể đã hết hạn 7 ngày)")

    entry["feedback"] = {
        "correct": bool(body.correct),
        "true_gloss": (body.true_gloss or "").strip() or None,
        "by": username,
        "at": time.time(),
    }
    kv.set_json(f"hist:e:{entry_id}", entry, ttl=60 * 60 * 24 * 30)
    kv.sadd(FEEDBACK_KEY, entry_id)

    if not body.correct and entry["feedback"]["true_gloss"]:
        conf = kv.get_json(CONFUSION_KEY) or {}
        key = f"{entry.get('top_gloss','?')}>{entry['feedback']['true_gloss']}"
        conf[key] = int(conf.get(key, 0)) + 1
        kv.set_json(CONFUSION_KEY, conf, ttl=None)
    return entry


# ═══════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 1)


@router.get("/api/insights/overview")
def insights_overview():
    lat = [float(x) for x in (kv.get_json(STATS_LATENCY) or [])]
    gloss_counts = kv.get_json(STATS_GLOSS) or {}
    models = kv.get_json(STATS_MODEL) or {}

    ids = kv.smembers(FEEDBACK_KEY)
    n_fb = n_ok = 0
    for eid in ids:
        e = kv.get_json(f"hist:e:{eid}")
        fb = (e or {}).get("feedback")
        if not fb:
            continue
        n_fb += 1
        n_ok += 1 if fb.get("correct") else 0

    entries = kv.history(HISTORY_KEY, limit=HISTORY_CAP)
    now = time.time()
    buckets = [0] * 12               # 12 khung 1 giờ gần nhất
    conf_hist = [0] * 10             # phân bố độ tin cậy 0-100 theo bước 10
    by_source: dict[str, int] = {}
    for e in entries:
        age_h = (now - float(e.get("created_at", now))) / 3600.0
        if 0 <= age_h < 12:
            buckets[11 - int(age_h)] += 1
        score = float(e.get("top_score") or 0.0)
        conf_hist[min(9, max(0, int(score * 10)))] += 1
        by_source[e.get("source", "?")] = by_source.get(e.get("source", "?"), 0) + 1

    return {
        "total_translations": int(kv.get_json(STATS_TOTAL) or 0) or len(entries),
        "latency": {
            "n": len(lat),
            "avg_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
            "p50_ms": _percentile(lat, 50),
            "p90_ms": _percentile(lat, 90),
            "min_ms": round(min(lat), 1) if lat else 0.0,
            "max_ms": round(max(lat), 1) if lat else 0.0,
            "recent": lat[:60],
        },
        "top_glosses": sorted(gloss_counts.items(), key=lambda kv_: -kv_[1])[:12],
        "models": [
            {
                "model_id": mid,
                "display_name": m.get("display_name", mid),
                "count": int(m.get("n", 0)),
                "avg_ms": round(float(m.get("ms_sum", 0)) / max(int(m.get("n", 1)), 1), 1),
            }
            for mid, m in sorted(models.items(), key=lambda kv_: -int(kv_[1].get("n", 0)))
        ],
        "feedback": {
            "n": n_fb,
            "correct": n_ok,
            "accuracy_pct": round(100.0 * n_ok / n_fb, 1) if n_fb else None,
        },
        "activity_12h": buckets,
        "confidence_hist": conf_hist,
        "by_source": by_source,
    }


@router.get("/api/insights/hard-cases")
def hard_cases(limit: int = Query(15, ge=1, le=60)):
    conf = kv.get_json(CONFUSION_KEY) or {}
    rows = []
    for key, n in conf.items():
        pred, _, true = key.partition(">")
        rows.append({"predicted": pred, "true": true, "count": int(n)})
    rows.sort(key=lambda r: -r["count"])
    by_true: dict[str, int] = {}
    for r in rows:
        by_true[r["true"]] = by_true.get(r["true"], 0) + r["count"]
    return {
        "pairs": rows[:limit],
        "needs_enroll": sorted(by_true.items(), key=lambda kv_: -kv_[1])[:limit],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SENTENCE BUILDER
# ═══════════════════════════════════════════════════════════════════════════
def _sentence_key(username: str) -> str:
    return f"sent:{username}"


class AppendRequest(BaseModel):
    gloss: str
    score: float = 0.0


def _render(tokens: list[dict]) -> str:
    """Ghép gloss thành câu: bỏ lặp liền kề, viết hoa đầu câu."""
    words: list[str] = []
    for t in tokens:
        g = str(t.get("gloss", "")).strip()
        if not g:
            continue
        if words and words[-1].lower() == g.lower():
            continue
        words.append(g)
    if not words:
        return ""
    s = " ".join(words)
    return s[0].upper() + s[1:]


@router.get("/api/sentence")
def get_sentence(username: str = Depends(_auth.require_auth)):
    tokens = kv.get_json(_sentence_key(username)) or []
    return {"tokens": tokens, "text": _render(tokens)}


@router.post("/api/sentence/append")
def append_sentence(body: AppendRequest, username: str = Depends(_auth.require_auth)):
    gloss = body.gloss.strip()
    if not gloss:
        raise HTTPException(422, "Gloss rỗng")
    tokens = kv.get_json(_sentence_key(username)) or []
    tokens.append({"gloss": gloss, "score": round(float(body.score), 4), "at": time.time()})
    tokens = tokens[-40:]
    kv.set_json(_sentence_key(username), tokens, ttl=60 * 60 * 12)
    return {"tokens": tokens, "text": _render(tokens)}


@router.post("/api/sentence/pop")
def pop_sentence(username: str = Depends(_auth.require_auth)):
    tokens = kv.get_json(_sentence_key(username)) or []
    if tokens:
        tokens.pop()
    kv.set_json(_sentence_key(username), tokens, ttl=60 * 60 * 12)
    return {"tokens": tokens, "text": _render(tokens)}


@router.post("/api/sentence/reset")
def reset_sentence(username: str = Depends(_auth.require_auth)):
    kv.delete(_sentence_key(username))
    return {"tokens": [], "text": ""}
