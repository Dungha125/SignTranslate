# -*- coding: utf-8 -*-
"""
CurriVSL Sign Translation API — FastAPI backend v3
  POST /api/auth/login
  POST /api/auth/logout
  GET  /api/auth/me
  GET  /api/models
  GET  /api/health
  POST /api/translate/video
  POST /api/translate/frames
  POST /api/learn/train        (auth required)
  GET  /api/learn/status/{id}  (auth required)
  GET  /api/learn/jobs         (auth required)
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
from extractor import SkeletonExtractor
from extractor_hsp import extract_hsp_from_frames, extract_hsp_from_video
from inference import CurriVSLModel
from inference_wbdpnet import WBDPNetModel
from inference_hsp_bimamba import HSPBiMambaModel
from inference_lt_signdiff import LTSignDiffModel
from inference_lt_signdiff_v2 import LTSignDiffV2Model
import train_new_word as _tnw
import train_new_word_hsp as _tnw_hsp

# ─── model registry ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent

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
        "display_name": "LT-SignDiff v2 Top200",
        "description": "Encoder không gian-thời gian v2 · 74 khớp × 10 kênh · hợp nhất classifier + prototype + kNN",
        "splits_dir": str(_REPO / "LT_SignDiff" / "data" / "splits_top200"),
    },
    "lt_signdiff_top30_masked": {
        "type": "lt_signdiff",
        "ckpt": str(os.environ.get(
            "LT_SIGNDIFF_TOP30_CKPT",
            _REPO / "sign_translate" / "models" / "lt_signdiff_top30_masked" / "best.pt",
        )),
        "num_joints": 110,
        "num_frames": 16,
        "model_num_frames": 16,
        "capture_buffer_frames": 32,
        "use_tta": True,
        "display_name": "LT-SignDiff Top30",
        "description": "Top-200 backbone masked to 30 densest glosses (offline Top-1 73.3%; enroll 2–3 mẫu/gloss → ~80%)",
        "splits_dir": str(_REPO / "LT_SignDiff" / "data" / "splits_top30"),
        "rebuild_config": str(_REPO / "LT_SignDiff" / "configs" / "vsl_top30.yaml"),
    },
    "lt_signdiff_top200": {
        "type": "lt_signdiff",
        "ckpt": str(os.environ.get(
            "LT_SIGNDIFF_CKPT",
            _REPO / "sign_translate" / "models" / "lt_signdiff_top200" / "best.pt",
        )),
        "num_joints": 110,
        "num_frames": 16,
        "model_num_frames": 16,
        "capture_buffer_frames": 32,
        "use_tta": True,
        "display_name": "LT-SignDiff Top200",
        "description": "LT-SignDiff closed-set 200 glosses (hand+face 110 joints)",
        "splits_dir": str(_REPO / "LT_SignDiff" / "data" / "splits_top200"),
        "rebuild_config": str(_REPO / "LT_SignDiff" / "configs" / "vsl_top200.yaml"),
    },
    "currivsl_110": {
        "type": "curivsl",
        "ckpt": str(os.environ.get(
            "CURRIVSL_CKPT_110",
            _REPO / "Model_full" / "runs_stageC_wlasl2000_handface110" / "stageC_curriculum.pt",
        )),
        "num_joints": 110,
        "display_name": "CurriVSL_110",
        "description": "Hand + Face (110 joints)",
    },
    "currivsl_42": {
        "type": "curivsl",
        "ckpt": str(os.environ.get(
            "CURRIVSL_CKPT_42",
            _REPO / "Curri" / "models_stageC" / "stageC_curriculum.pt",
        )),
        "num_joints": 42,
        "display_name": "CurriVSL_42",
        "description": "Hand only (42 joints)",
    },
    "wbdpnet_v2_143": {
        "type": "wbdpnet",
        "ckpt": str(os.environ.get(
            "WBDPNET_CKPT_143",
            _REPO
            / "sign_translate"
            / "models"
            / "wbdpnet_vsl_v2_be_bundle"
            / "wbdpnet_vsl_v2_be"
            / "runs"
            / "wbdpnet_vsl_v2"
            / "best.pt",
        )),
        "num_joints": 143,
        "display_name": "WBDPNet_143",
        "description": "Whole-body dual-reference prototype network (hand+face+pose)",
    },
    "hsp_bimamba_top100": {
        "type": "hsp_bimamba",
        "ckpt": str(os.environ.get(
            "HSP_BIMAMBA_CKPT",
            _REPO
            / "sign_translate"
            / "models"
            / "hsp_bimamba_vsl_top100_mp75"
            / "best.pt",
        )),
        "num_joints": 75,
        "extract_joints": 143,
        "num_frames": 0,
        "model_num_frames": 30,
        "capture_buffer_frames": 50,
        "use_tta": False,
        "display_name": "HSP-BiMamba Top100",
        "description": "VSL top-100 glosses, MP75 skeleton (B5, K=4)",
    },
}

DEFAULT_MODEL = os.environ.get("SIGN_TRANSLATE_DEFAULT_MODEL", "lt_signdiff_top30_masked")
TOP_K = int(os.environ.get("CURRIVSL_TOP_K", "10"))

# Webcam capture settings per model family (aligned with skeleton T at inference).
CAPTURE_DEFAULTS: dict[str, dict] = {
    "lt_signdiff_v2": {
        "capture_buffer_frames": 48,
        "capture_num_frames": 32,
        "capture_min_frames": 20,
        "capture_interval_ms": 90,
        "capture_jpeg_quality": 0.9,
        "use_tta": True,
    },
    "lt_signdiff": {
        "capture_buffer_frames": 32,
        "capture_num_frames": 16,
        "capture_min_frames": 16,
        "capture_interval_ms": 100,
        "capture_jpeg_quality": 0.9,
        "use_tta": True,
    },
    "hsp_bimamba": {
        "capture_buffer_frames": 50,
        "capture_num_frames": 30,
        "capture_min_frames": 40,
        "capture_interval_ms": 100,
        "capture_jpeg_quality": 0.92,
        "use_tta": False,
    },
    "curivsl": {
        "capture_num_frames": 16,
        "capture_min_frames": 8,
        "capture_interval_ms": 100,
        "capture_jpeg_quality": 0.75,
    },
    "wbdpnet": {
        "capture_num_frames": 16,
        "capture_min_frames": 8,
        "capture_interval_ms": 100,
        "capture_jpeg_quality": 0.75,
    },
}


def _capture_cfg(cfg: dict) -> dict:
    mtype = cfg.get("type", "curivsl")
    base = dict(CAPTURE_DEFAULTS.get(mtype, CAPTURE_DEFAULTS["curivsl"]))
    if cfg.get("model_num_frames"):
        base["capture_num_frames"] = int(cfg["model_num_frames"])
    if cfg.get("capture_buffer_frames"):
        base["capture_buffer_frames"] = int(cfg["capture_buffer_frames"])
    if base.get("capture_buffer_frames") and not cfg.get("capture_min_frames"):
        base["capture_min_frames"] = max(8, int(base["capture_buffer_frames"]) - 10)
    return base

# ─── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="LT-SignDiff Sign Translate API", version="5.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.include_router(_routes_dataset.router)
app.include_router(_routes_insights.router)

_models: dict[str, object] = {}
_extractor: SkeletonExtractor | None = None

# Cache kết quả suy luận theo hash nội dung — bấm dịch lại cùng video là tức thì.
INFER_CACHE_TTL = int(os.environ.get("SIGN_TRANSLATE_CACHE_TTL", str(60 * 60 * 12)))


# ─── reload callback (called after training) ─────────────────────────────────
def _reload_model(model_id: str, ckpt_path: Path, num_joints: int, top_k: int):
    try:
        cfg = MODEL_REGISTRY.get(model_id, {})
        mtype = cfg.get("type", "curivsl")
        if mtype == "wbdpnet":
            _models[model_id] = WBDPNetModel(ckpt_path, num_joints=num_joints, top_k=top_k)
        elif mtype == "hsp_bimamba":
            _models[model_id] = HSPBiMambaModel(
                ckpt_path,
                num_joints=num_joints,
                num_frames=int(cfg.get("model_num_frames", 30)),
                top_k=top_k,
                use_tta=bool(cfg.get("use_tta", True)),
            )
        elif mtype == "lt_signdiff_v2":
            _models[model_id] = LTSignDiffV2Model(
                ckpt_path,
                num_frames=int(cfg.get("model_num_frames", 32)),
                top_k=top_k,
                use_tta=bool(cfg.get("use_tta", True)),
            )
        elif mtype == "lt_signdiff":
            _models[model_id] = LTSignDiffModel(
                ckpt_path,
                num_joints=num_joints,
                num_frames=int(cfg.get("model_num_frames", 16)),
                top_k=top_k,
                use_tta=bool(cfg.get("use_tta", True)),
            )
        else:
            _models[model_id] = CurriVSLModel(ckpt_path, num_joints=num_joints, top_k=top_k)
        print(f"[Reload] {model_id} reloaded from {ckpt_path}")
    except Exception as e:
        print(f"[Reload] failed for {model_id}: {e}")


def _extract_cfg(model_id: str) -> tuple[int, int | None]:
    cfg = MODEL_REGISTRY[model_id]
    extract_joints = int(cfg.get("extract_joints", cfg["num_joints"]))
    num_frames = cfg.get("num_frames")
    if num_frames is not None:
        num_frames = int(num_frames)
    return extract_joints, num_frames


def _extract_skeleton_video(video_path: str | Path, model_id: str) -> np.ndarray | None:
    cfg = MODEL_REGISTRY[model_id]
    if cfg.get("type") == "hsp_bimamba":
        return extract_hsp_from_video(
            video_path,
            num_frames=int(cfg.get("model_num_frames", 30)),
        )
    extract_joints, num_frames = _extract_cfg(model_id)
    return _extractor.extract_from_video_path(
        video_path, num_joints=extract_joints, num_frames=num_frames
    )


def _extract_skeleton_frames(frames: list[np.ndarray], model_id: str) -> np.ndarray | None:
    cfg = MODEL_REGISTRY[model_id]
    if cfg.get("type") == "hsp_bimamba":
        cap = _capture_cfg(cfg)
        return extract_hsp_from_frames(
            frames,
            num_frames=int(cfg.get("model_num_frames", 30)),
            buffer_frames=int(cap.get("capture_buffer_frames", 50)),
        )
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
        if p.is_file():
            try:
                if cfg.get("type") == "wbdpnet":
                    _models[mid] = WBDPNetModel(p, num_joints=cfg["num_joints"], top_k=TOP_K)
                elif cfg.get("type") == "hsp_bimamba":
                    _models[mid] = HSPBiMambaModel(
                        p,
                        num_joints=cfg["num_joints"],
                        num_frames=int(cfg.get("model_num_frames", 30)),
                        top_k=TOP_K,
                        use_tta=bool(cfg.get("use_tta", True)),
                    )
                elif cfg.get("type") == "lt_signdiff_v2":
                    _models[mid] = LTSignDiffV2Model(
                        p,
                        num_frames=int(cfg.get("model_num_frames", 32)),
                        top_k=TOP_K,
                        use_tta=bool(cfg.get("use_tta", True)),
                    )
                elif cfg.get("type") == "lt_signdiff":
                    _models[mid] = LTSignDiffModel(
                        p,
                        num_joints=cfg["num_joints"],
                        num_frames=int(cfg.get("model_num_frames", 16)),
                        top_k=TOP_K,
                        use_tta=bool(cfg.get("use_tta", True)),
                    )
                else:
                    _models[mid] = CurriVSLModel(p, num_joints=cfg["num_joints"], top_k=TOP_K)
            except Exception as e:
                print(f"[WARN] {mid}: {e}")
        else:
            print(f"[WARN] ckpt not found: {p}")


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


class LearnRequest(BaseModel):
    frames_b64: list[str]
    label: str
    model_id: str = DEFAULT_MODEL

# ─── helpers ─────────────────────────────────────────────────────────────────
def _get_model(model_id: str) -> CurriVSLModel:
    m = _models.get(model_id)
    if not m:
        raise HTTPException(404, f"Model '{model_id}' chưa load. Có: {list(_models)}")
    return m

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
            "type": cfg.get("type", "curivsl"),
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


class CompareRequest(BaseModel):
    frames_b64: list[str] = []
    model_ids: list[str]


@app.post("/api/translate/compare")
async def translate_compare(req: CompareRequest):
    """Chạy cùng một đoạn ghi qua nhiều model để so sánh trực tiếp."""
    if _extractor is None:
        raise HTTPException(503, "Extractor not ready")
    frames = _decode_frames(req.frames_b64)
    if not frames:
        raise HTTPException(422, "Không decode được frame nào")
    ids = [m for m in req.model_ids if m in MODEL_REGISTRY and m in _models][:4]
    if not ids:
        raise HTTPException(400, "Không có model_id hợp lệ nào đang được load")

    out = []
    for mid in ids:
        t0 = time.time()
        try:
            sk = _extract_skeleton_frames(frames, mid)
            if sk is None:
                raise ValueError("không trích được skeleton")
            preds = _get_model(mid).predict(sk)
            out.append(
                {
                    "model_id": mid,
                    "display_name": MODEL_REGISTRY[mid]["display_name"],
                    "predictions": preds[:5],
                    "elapsed_ms": round((time.time() - t0) * 1000, 1),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - một model lỗi không chặn phần còn lại
            out.append(
                {
                    "model_id": mid,
                    "display_name": MODEL_REGISTRY[mid]["display_name"],
                    "predictions": [],
                    "elapsed_ms": round((time.time() - t0) * 1000, 1),
                    "error": str(exc),
                }
            )

    votes: dict[str, float] = {}
    for r in out:
        for p in r["predictions"][:3]:
            votes[p["gloss"]] = votes.get(p["gloss"], 0.0) + float(p["score"])
    consensus = sorted(votes.items(), key=lambda kv: -kv[1])[:5]
    return {
        "results": out,
        "consensus": [{"gloss": g, "score": round(s / max(len(out), 1), 4)} for g, s in consensus],
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
    if not isinstance(m, (LTSignDiffModel, LTSignDiffV2Model)):
        raise HTTPException(400, "Gallery chỉ hỗ trợ LT-SignDiff")
    return {"model_id": model_id, **m.gallery_info()}


@app.post("/api/enroll/frames", response_model=EnrollResponse)
async def enroll_frames(req: EnrollRequest):
    if _extractor is None:
        raise HTTPException(503, "Extractor not ready")
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(400, f"model_id không hợp lệ: {req.model_id}")
    m = _models.get(req.model_id)
    if not isinstance(m, (LTSignDiffModel, LTSignDiffV2Model)):
        raise HTTPException(400, "Enroll chỉ hỗ trợ LT-SignDiff Top-200")

    gloss = req.gloss.strip()
    if not gloss:
        raise HTTPException(400, "Chọn gloss trong bộ Top-200")

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
    mtype = cfg.get("type")
    if mtype not in ("lt_signdiff", "lt_signdiff_v2"):
        raise HTTPException(400, "Chỉ rebuild được LT-SignDiff")
    import subprocess

    is_v2 = mtype == "lt_signdiff_v2"
    script = _REPO / "LT_SignDiff" / "scripts" / (
        "rebuild_gallery_v2.py" if is_v2 else "rebuild_gallery.py"
    )
    if not script.is_file():
        raise HTTPException(500, f"Missing {script}")
    cmd = [sys.executable, str(script), "--ckpt", str(cfg["ckpt"])]
    if not is_v2:
        cmd += ["--export-ckpt", str(cfg["ckpt"])]
        if cfg.get("rebuild_config"):
            cmd += ["--config", str(cfg["rebuild_config"])]
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
    if isinstance(m, (LTSignDiffModel, LTSignDiffV2Model)):
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


# ═══════════════════════════════════════════════════════════════════════════════
# LEARN (auth required)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/learn/train")
async def learn_train(
    req: LearnRequest,
    username: str = Depends(_auth.require_auth),
):
    if _extractor is None: raise HTTPException(503, "Extractor not ready")
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(400, f"model_id không hợp lệ: {req.model_id}")
    if not req.label.strip():
        raise HTTPException(400, "Nhãn không được để trống")

    cfg = MODEL_REGISTRY[req.model_id]
    mtype = cfg.get("type", "curivsl")
    if mtype not in ("curivsl", "hsp_bimamba"):
        raise HTTPException(501, f"Model '{req.model_id}' hiện không hỗ trợ /api/learn/train.")

    frames = _decode_frames(req.frames_b64)
    if not frames:
        raise HTTPException(422, "Không có frame hợp lệ")

    cap = _capture_cfg(cfg)
    min_frames = int(cap.get("capture_min_frames", 8))
    if mtype == "hsp_bimamba" and len(frames) < min_frames:
        raise HTTPException(
            422,
            f"Cần ít nhất {min_frames} frame (đã nhận {len(frames)}). Ghi ~5s ký hiệu.",
        )

    sk = _extract_skeleton_frames(frames, req.model_id)
    if sk is None:
        raise HTTPException(422, "Không trích được skeleton từ webcam")
    ckpt_path = Path(cfg["ckpt"])
    if not ckpt_path.is_file():
        raise HTTPException(404, f"Checkpoint chưa có: {ckpt_path}")

    label = req.label.strip()
    if mtype == "hsp_bimamba":
        job_id = _tnw_hsp.start_training(
            skeleton=sk,
            label=label,
            model_id=req.model_id,
            ckpt_path=ckpt_path,
            num_joints=cfg["num_joints"],
            top_k=TOP_K,
            reload_model_cb=_reload_model,
        )
    else:
        job_id = _tnw.start_training(
            skeleton=sk,
            label=label.lower(),
            model_id=req.model_id,
            ckpt_path=ckpt_path,
            num_joints=cfg["num_joints"],
            top_k=TOP_K,
            reload_model_cb=_reload_model,
        )
    return {"job_id": job_id, "message": f"Bắt đầu huấn luyện từ '{label}'"}


@app.get("/api/learn/vocab")
def learn_vocab(model_id: str, username: str = Depends(_auth.require_auth)):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(400, f"model_id không hợp lệ: {model_id}")
    m = _models.get(model_id)
    if not m:
        raise HTTPException(404, f"Model '{model_id}' chưa load")
    labels = sorted(m.id2label.values()) if hasattr(m, "id2label") else []
    return {"model_id": model_id, "num_classes": len(labels), "labels": labels}


@app.get("/api/learn/status/{job_id}")
def learn_status(job_id: str, username: str = Depends(_auth.require_auth)):
    job = _tnw.get_job(job_id)
    if not job: raise HTTPException(404, "Job không tồn tại")
    return job.as_dict()


@app.get("/api/learn/jobs")
def learn_jobs(
    model_id: str | None = None,
    username: str = Depends(_auth.require_auth),
):
    return _tnw.list_jobs(model_id)
