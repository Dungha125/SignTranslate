# -*- coding: utf-8 -*-
"""
SignTranslate API — FastAPI backend, chạy trên một model duy nhất: LT-SignDiff v2.

  POST /api/auth/login | /api/auth/logout | GET /api/auth/me
  GET  /api/models · /api/health
  GET  /api/vocab/{model_id} · /api/gallery/{model_id}
  POST /api/translate/video · /api/translate/frames
  POST /api/enroll/frames · /api/gallery/rebuild/{model_id}

Các router khác: routes_dataset (kho clip huấn luyện), routes_insights (lịch sử +
thống kê), routes_library (kho video từ vựng + luyện tập).
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Log của hệ thống viết bằng tiếng Việt, kể cả tên gloss. Console Windows mặc
# định là cp1252 nên một dòng log có thể ném UnicodeEncodeError và làm hỏng cả
# bước nạp model — buộc UTF-8 ngay từ đầu.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import auth as _auth
import storage as _storage
import dataset_store as _ds
import routes_dataset as _routes_dataset
import routes_insights as _routes_insights
import routes_library as _routes_library
from extractor import SkeletonExtractor
from inference_lt_signdiff_v2 import LTSignDiffV2Model

# ─── model registry ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent

# Hệ thống chạy trên một kiến trúc duy nhất: LT-SignDiff v2. Các nhánh CurriVSL /
# WBDPNet / HSP-BiMamba vẫn còn trong repo cho mục đích so sánh trong bài báo,
# nhưng không nằm trong đường chạy của dịch vụ nữa.
MODEL_REGISTRY: dict[str, dict] = {
    "lt_signdiff_v2_top200": {
        "type": "lt_signdiff_v2",
        "ckpt": str(os.environ.get(
            "LT_SIGNDIFF_V2_CKPT",
            _REPO / "sign_translate" / "models" / "lt_signdiff_v2_top200" / "best.pt",
        )),
        # Bộ trích xuất phải trả 143 khớp và **giữ nguyên số frame** (num_frames=0);
        # việc lấy mẫu về T=32 do pipeline đặc trưng v2 lo.
        "num_joints": 143,
        "extract_joints": 143,
        "num_frames": 0,
        "model_num_frames": 32,
        "capture_buffer_frames": 48,
        "use_tta": True,
        "display_name": "LT-SignDiff v2",
        "description": "Encoder không gian-thời gian · 74 khớp × 10 kênh · hợp nhất classifier + prototype + kNN",
        "splits_dir": str(_REPO / "LT_SignDiff" / "data" / "splits_top200"),
    },
}

DEFAULT_MODEL = os.environ.get("SIGN_TRANSLATE_DEFAULT_MODEL", "lt_signdiff_v2_top200")
if DEFAULT_MODEL not in MODEL_REGISTRY:
    DEFAULT_MODEL = next(iter(MODEL_REGISTRY))
TOP_K = int(os.environ.get("SIGN_TRANSLATE_TOP_K", "10"))

# Thông số ghi webcam khớp với T của skeleton lúc suy luận.
CAPTURE_DEFAULTS: dict[str, dict] = {
    "lt_signdiff_v2": {
        "capture_buffer_frames": 48,
        "capture_num_frames": 32,
        "capture_min_frames": 20,
        "capture_interval_ms": 90,
        "capture_jpeg_quality": 0.9,
        "use_tta": True,
    },
}


def _capture_cfg(cfg: dict) -> dict:
    mtype = cfg.get("type", "lt_signdiff_v2")
    base = dict(CAPTURE_DEFAULTS.get(mtype, CAPTURE_DEFAULTS["lt_signdiff_v2"]))
    if cfg.get("model_num_frames"):
        base["capture_num_frames"] = int(cfg["model_num_frames"])
    if cfg.get("capture_buffer_frames"):
        base["capture_buffer_frames"] = int(cfg["capture_buffer_frames"])
    if base.get("capture_buffer_frames") and not cfg.get("capture_min_frames"):
        base["capture_min_frames"] = max(8, int(base["capture_buffer_frames"]) - 10)
    return base

# ─── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="LT-SignDiff Sign Translate API", version="5.0.0")

# Frontend chạy trên domain khác (Vercel) nên phải khai báo origin cụ thể;
# để trống hoặc "*" thì mở cho mọi nguồn như bản dev.
_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("SIGN_TRANSLATE_CORS_ORIGINS", "*").split(",") if o.strip()
] or ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=_CORS_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.include_router(_routes_dataset.router)
app.include_router(_routes_insights.router)
app.include_router(_routes_library.router)

_models: dict[str, object] = {}
_extractor: SkeletonExtractor | None = None

# Cache kết quả suy luận theo hash nội dung — bấm dịch lại cùng video là tức thì.
INFER_CACHE_TTL = int(os.environ.get("SIGN_TRANSLATE_CACHE_TTL", str(60 * 60 * 12)))


def _extract_cfg(model_id: str) -> tuple[int, int | None]:
    cfg = MODEL_REGISTRY[model_id]
    extract_joints = int(cfg.get("extract_joints", cfg["num_joints"]))
    num_frames = cfg.get("num_frames")
    if num_frames is not None:
        num_frames = int(num_frames)
    return extract_joints, num_frames


def _extract_skeleton_video(video_path: str | Path, model_id: str) -> np.ndarray | None:
    extract_joints, num_frames = _extract_cfg(model_id)
    return _extractor.extract_from_video_path(
        video_path, num_joints=extract_joints, num_frames=num_frames
    )


def _extract_skeleton_frames(frames: list[np.ndarray], model_id: str) -> np.ndarray | None:
    extract_joints, num_frames = _extract_cfg(model_id)
    return _extractor.extract_from_frames(
        frames, num_joints=extract_joints, num_frames=num_frames
    )


@app.on_event("startup")
def startup():
    global _extractor
    _extractor = SkeletonExtractor()
    h = _storage.health()
    print(f"[storage] redis={h['redis']['backend']} object_store={h['object_store']['backend']}")
    try:
        n = _ds.restore_from_journal()
        if n:
            print(f"[dataset] khôi phục {n} clip từ journal")
    except Exception as exc:  # noqa: BLE001
        print(f"[dataset] restore lỗi: {exc}")
    for mid, cfg in MODEL_REGISTRY.items():
        p = Path(cfg["ckpt"])
        if not p.is_file():
            print(f"[WARN] ckpt not found: {p}")
            continue
        try:
            _models[mid] = LTSignDiffV2Model(
                p,
                num_frames=int(cfg.get("model_num_frames", 32)),
                top_k=TOP_K,
                use_tta=bool(cfg.get("use_tta", True)),
            )
        except Exception as e:  # noqa: BLE001 - một model lỗi không được chặn boot
            print(f"[WARN] {mid}: {e}")

    # Kho từ vựng cần vocab của model để biết 200 từ nào cần có video mẫu.
    _routes_library.bind(
        vocab_provider=lambda: _vocab_of(DEFAULT_MODEL),
        predict=_predict_for_library,
    )


@app.on_event("shutdown")
def shutdown():
    if _extractor:
        _extractor.close()


# ─── schemas ─────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PredictionItem(BaseModel):
    rank: int; gloss: str; score: float; confidence_pct: float

class TranslateResponse(BaseModel):
    model_id: str; display_name: str
    predictions: list[PredictionItem]
    elapsed_ms: float; model_loaded: bool
    entry_id: str | None = None     # id trong lịch sử, dùng để gửi phản hồi
    cached: bool = False            # kết quả lấy từ cache Redis

class FramesRequest(BaseModel):
    frames_b64: list[str]
    model_id: str = DEFAULT_MODEL

class EnrollRequest(BaseModel):
    frames_b64: list[str]
    gloss: str
    model_id: str = DEFAULT_MODEL


class EnrollResponse(BaseModel):
    gloss: str
    total_refs: int
    class_id: int
    message: str


# ─── helpers ─────────────────────────────────────────────────────────────────
def _get_model(model_id: str) -> LTSignDiffV2Model:
    m = _models.get(model_id)
    if not m:
        raise HTTPException(404, f"Model '{model_id}' chưa load. Có: {list(_models)}")
    return m


def _vocab_of(model_id: str) -> list[str]:
    m = _models.get(model_id)
    if not m:
        return []
    return [m.id2label[i] for i in sorted(m.id2label.keys())]


def _predict_for_library(frames_b64: list[str]) -> tuple[list[dict], float]:
    """Suy luận cho chế độ luyện tập. Trả (predictions, elapsed_ms).

    Không ghi vào lịch sử dịch: một buổi luyện tập có hàng chục lượt thử nên sẽ
    làm nhiễu số liệu vận hành ở tab Thống kê.
    """
    t0 = time.time()
    if _extractor is None:
        raise HTTPException(503, "Extractor chưa sẵn sàng")
    frames = _decode_frames(frames_b64)
    if not frames:
        raise HTTPException(422, "Không decode được frame nào")
    cap = _capture_cfg(MODEL_REGISTRY[DEFAULT_MODEL])
    if len(frames) < int(cap["capture_min_frames"]):
        raise HTTPException(422, f"Cần ít nhất {cap['capture_min_frames']} frame, mới nhận {len(frames)}")
    sk = _extract_skeleton_frames(frames, DEFAULT_MODEL)
    if sk is None:
        raise HTTPException(422, "Không trích được skeleton — giữ hai tay trong khung hình")
    preds = _get_model(DEFAULT_MODEL).predict(sk)
    return preds, round((time.time() - t0) * 1000, 1)

def _run_inference(
    sk: np.ndarray,
    model_id: str,
    t0: float,
    *,
    source: str = "upload",
    username: str = "anonymous",
    filename: str = "",
    cache_token: str | None = None,
) -> TranslateResponse:
    """Suy luận + ghi lịch sử. `cache_token` bật cache Redis theo hash nội dung."""
    display_name = MODEL_REGISTRY[model_id]["display_name"]
    ckey = f"infer:{model_id}:{cache_token}" if cache_token else None
    cached = False
    preds = None

    if ckey:
        hit = _storage.kv.get_json(ckey)
        if hit:
            preds, cached = hit, True

    if preds is None:
        preds = _get_model(model_id).predict(sk)
        if ckey:
            _storage.kv.set_json(ckey, preds, ttl=INFER_CACHE_TTL)

    elapsed = round((time.time() - t0) * 1000, 1)
    try:
        entry_id = _routes_insights.record_translation(
            model_id=model_id,
            display_name=display_name,
            predictions=preds,
            elapsed_ms=elapsed,
            source=source,
            username=username,
            filename=filename,
            cached=cached,
        )
    except Exception as exc:  # noqa: BLE001 - lịch sử hỏng không được chặn kết quả
        print(f"[history] bỏ qua: {exc}")
        entry_id = None

    return TranslateResponse(
        model_id=model_id,
        display_name=display_name,
        predictions=[PredictionItem(**p) for p in preds],
        elapsed_ms=elapsed,
        model_loaded=True,
        entry_id=entry_id,
        cached=cached,
    )

def _decode_frames(frames_b64: list[str]) -> list[np.ndarray]:
    frames = []
    for b64 in frames_b64:
        # Client trong repo đã cắt sẵn tiền tố, nhưng canvas.toDataURL của một
        # client khác thì không — cắt ở đây để không trả 422 khó hiểu.
        if b64.startswith("data:"):
            b64 = b64.partition(",")[2]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/login")
def login(req: LoginRequest):
    token = _auth.login(req.username, req.password)
    if not token:
        raise HTTPException(401, "Sai tài khoản hoặc mật khẩu")
    return {"token": token, "username": req.username}

@app.post("/api/auth/logout")
def logout(authorization: str = None):
    if authorization and authorization.startswith("Bearer "):
        _auth.logout(authorization[7:])
    return {"ok": True}

@app.get("/api/auth/me")
def me(username: str = Depends(_auth.require_auth)):
    return {"username": username}


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS / HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/models")
def list_models():
    return [
        {
            "id": mid, "display_name": cfg["display_name"],
            "description": cfg["description"], "num_joints": cfg["num_joints"],
            "type": cfg.get("type", "lt_signdiff_v2"),
            "loaded": mid in _models,
            "num_classes": len(_models[mid].id2label) if mid in _models else 0,
            **_capture_cfg(cfg),
        }
        for mid, cfg in MODEL_REGISTRY.items()
    ]

@app.get("/api/health")
def health():
    default_id = DEFAULT_MODEL if DEFAULT_MODEL in MODEL_REGISTRY else next(iter(MODEL_REGISTRY))
    default_cfg = MODEL_REGISTRY[default_id]
    default_loaded = default_id in _models
    return {
        "status": "ok",
        "default_model": default_id,
        "model_loaded": default_loaded,
        "display_name": default_cfg.get("display_name", default_id),
        "num_classes": len(_models[default_id].id2label) if default_loaded else 0,
        "ckpt_path": str(default_cfg.get("ckpt", "")),
        "models": {
            mid: {
                "loaded": mid in _models,
                "num_classes": len(_models[mid].id2label) if mid in _models else 0,
            }
            for mid in MODEL_REGISTRY
        },
        "storage": {
            "redis": _storage.kv.backend,
            "redis_ok": _storage.kv.available,
            "object_store": _storage.objects.backend,
            "object_store_ok": _storage.objects.available,
        },
    }


@app.get("/api/vocab/{model_id}")
def get_vocab(model_id: str):
    """Return gloss list for a loaded closed-set model (e.g. top-200)."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model_id: {model_id}")
    m = _models.get(model_id)
    if not m:
        raise HTTPException(404, f"Model '{model_id}' chưa load")
    glosses = [m.id2label[i] for i in sorted(m.id2label.keys())]
    return {
        "model_id": model_id,
        "num_classes": len(glosses),
        "glosses": glosses,
    }


@app.get("/api/gallery/{model_id}")
def get_gallery(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model_id: {model_id}")
    m = _models.get(model_id)
    if not m:
        raise HTTPException(404, f"Model '{model_id}' chưa load")
    return {"model_id": model_id, **m.gallery_info()}


@app.post("/api/enroll/frames", response_model=EnrollResponse)
async def enroll_frames(req: EnrollRequest):
    if _extractor is None:
        raise HTTPException(503, "Extractor not ready")
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(400, f"model_id không hợp lệ: {req.model_id}")
    m = _get_model(req.model_id)

    gloss = req.gloss.strip()
    if not gloss:
        raise HTTPException(400, "Chọn một từ trong bộ từ vựng")

    frames = _decode_frames(req.frames_b64)
    if not frames:
        raise HTTPException(422, "Không decode được frame nào")
    cap = _capture_cfg(MODEL_REGISTRY[req.model_id])
    if len(frames) < int(cap["capture_min_frames"]):
        raise HTTPException(422, f"Cần ít nhất {cap['capture_min_frames']} frame")

    sk = _extract_skeleton_frames(frames, req.model_id)
    if sk is None:
        raise HTTPException(422, "Không trích được skeleton")
    try:
        info = m.enroll(sk, gloss)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return EnrollResponse(
        gloss=info["gloss"],
        total_refs=info["total_refs"],
        class_id=info["class_id"],
        message=f"Đã thêm mẫu cho '{gloss}' · gallery {info['total_refs']} refs",
    )


@app.post("/api/gallery/rebuild/{model_id}")
def rebuild_gallery(model_id: str):
    """Rebuild gallery from train+val split (admin utility)."""
    cfg = MODEL_REGISTRY.get(model_id) or {}
    if not cfg:
        raise HTTPException(404, f"Unknown model_id: {model_id}")
    import subprocess

    script = _REPO / "LT_SignDiff" / "scripts" / "rebuild_gallery_v2.py"
    if not script.is_file():
        raise HTTPException(500, f"Missing {script}")
    cmd = [sys.executable, str(script), "--ckpt", str(cfg["ckpt"])]
    if cfg.get("splits_dir"):
        cmd += ["--splits", str(cfg["splits_dir"])]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO / "LT_SignDiff"),
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    if r.returncode != 0:
        raise HTTPException(500, r.stderr or r.stdout or "rebuild failed")
    m = _models.get(model_id)
    if m is not None:
        m._load_gallery()
    return {"ok": True, "output": r.stdout[-2000:]}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATE
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/translate/video", response_model=TranslateResponse)
async def translate_video(
    file: UploadFile = File(...),
    model_id: str = Form(DEFAULT_MODEL),
    save_to_dataset: bool = Form(False),
    gloss: str = Form(""),
):
    t0 = time.time()
    if _extractor is None: raise HTTPException(503, "Extractor not ready")
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        if model_id not in MODEL_REGISTRY:
            raise HTTPException(400, f"model_id không hợp lệ: {model_id}")
        token = _storage.content_hash(data)
        sk = _extract_skeleton_video(tmp_path, model_id)
        if sk is None: raise HTTPException(422, "Không trích được skeleton từ video")
        resp = _run_inference(
            sk, model_id, t0,
            source="upload",
            filename=file.filename or "",
            cache_token=token,
        )
        # Tuỳ chọn: giữ lại video vừa dịch trong kho MinIO để mở rộng dataset.
        if save_to_dataset:
            label = (gloss.strip() or (resp.predictions[0].gloss if resp.predictions else "")).strip()
            if label:
                try:
                    _ds.add_clip(
                        gloss=label,
                        data=data,
                        filename=file.filename or f"clip{suffix}",
                        content_type=file.content_type or "video/mp4",
                        split="unassigned",
                        source="upload",
                        uploaded_by="translate",
                        thumbnail=_routes_dataset._thumbnail_from_video(tmp_path),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[dataset] không lưu được clip: {exc}")
        return resp
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/translate/frames", response_model=TranslateResponse)
async def translate_frames(req: FramesRequest):
    t0 = time.time()
    if _extractor is None: raise HTTPException(503, "Extractor not ready")
    frames = _decode_frames(req.frames_b64)
    if not frames: raise HTTPException(422, "Không decode được frame nào")
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(400, f"model_id không hợp lệ: {req.model_id}")
    cap = _capture_cfg(MODEL_REGISTRY[req.model_id])
    min_frames = int(cap["capture_min_frames"])
    if len(frames) < min_frames:
        raise HTTPException(
            422,
            f"Cần ít nhất {min_frames} frame (đã nhận {len(frames)}). "
            f"Ghi lâu hơn ~{cap['capture_num_frames'] * cap['capture_interval_ms'] / 1000:.0f}s.",
        )
    sk = _extract_skeleton_frames(frames, req.model_id)
    if sk is None: raise HTTPException(422, "Không trích được skeleton")
    # Webcam luôn khác nhau từng lần ghi → không cache.
    return _run_inference(sk, req.model_id, t0, source="webcam")
