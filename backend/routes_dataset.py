# -*- coding: utf-8 -*-
"""Router cho kho dữ liệu + hạ tầng lưu trữ (Redis / MinIO).

  GET    /api/storage/health          trạng thái Redis + MinIO
  POST   /api/storage/reconnect       thử kết nối lại (auth)
  GET    /api/dataset/stats           số liệu tổng hợp
  GET    /api/dataset/glosses         danh sách gloss + số clip
  GET    /api/dataset/clips           duyệt/lọc clip
  POST   /api/dataset/clips           upload video vào kho (auth)
  POST   /api/dataset/clips/webcam    lưu clip webcam base64 (auth)
  PATCH  /api/dataset/clips/{id}      sửa gloss/split (auth)
  DELETE /api/dataset/clips/{id}      xoá clip (auth)
  GET    /api/dataset/clips/{id}/stream  phát video từ MinIO
  GET    /api/dataset/export.csv      xuất split CSV
  POST   /api/dataset/import-corpus   nạp corpus VSL trên đĩa vào MinIO (auth)
"""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import auth as _auth
import dataset_store as ds
import storage

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent
MAX_UPLOAD_MB = int(os.environ.get("SIGN_TRANSLATE_MAX_UPLOAD_MB", "80"))


def _thumbnail_from_video(path: str | Path, width: int = 320) -> bytes | None:
    """Lấy frame giữa clip làm thumbnail JPEG."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


def _thumbnail_from_frame_b64(b64: str, width: int = 320) -> bytes | None:
    try:
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return buf.tobytes() if ok else None
    except Exception:  # noqa: BLE001
        return None


def _frames_to_webm(frames_b64: list[str], fps: int = 12) -> tuple[bytes | None, bytes | None]:
    """Ghép frame JPEG base64 thành 1 file mp4 (trả bytes video + thumbnail)."""
    imgs = []
    for b64 in frames_b64:
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
        except Exception:  # noqa: BLE001
            continue
    if not imgs:
        return None, None
    h, w = imgs[0].shape[:2]
    # mkstemp trả về fd đang mở; trên Windows nếu không đóng thì unlink sẽ báo
    # "file đang được tiến trình khác dùng".
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    tmp = Path(tmp_name)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img in imgs:
        writer.write(cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img)
    writer.release()
    try:
        data = tmp.read_bytes() if tmp.is_file() else None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:      # file tạm còn bị giữ — dọn sau, không chặn request
            pass
    ok, buf = cv2.imencode(".jpg", imgs[len(imgs) // 2], [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    return data, (buf.tobytes() if ok else None)


# ═══════════════════════════════════════════════════════════════════════════
# STORAGE
# ═══════════════════════════════════════════════════════════════════════════
@router.get("/api/storage/health")
def storage_health():
    h = storage.health()
    h["object_store"]["usage"] = storage.objects.usage()
    return h


@router.post("/api/storage/reconnect")
def storage_reconnect(username: str = Depends(_auth.require_auth)):
    return {
        "redis": storage.kv.reconnect(),
        "minio": storage.objects.reconnect(),
        "health": storage.health(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════
@router.get("/api/dataset/stats")
def dataset_stats():
    return ds.stats()


@router.get("/api/dataset/glosses")
def dataset_glosses():
    return ds.list_glosses()


@router.get("/api/dataset/clips")
def dataset_clips(
    gloss: str | None = None,
    split: str | None = None,
    source: str | None = None,
    q: str = "",
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    page = ds.list_clips(gloss=gloss, split=split, source=source, query=q, limit=limit, offset=offset)
    for c in page["clips"]:
        c["stream_url"] = f"/api/dataset/clips/{c['clip_id']}/stream"
        c["thumb_url"] = f"/api/dataset/clips/{c['clip_id']}/thumb" if c.get("thumb_key") else None
    return page


@router.post("/api/dataset/clips")
async def dataset_upload(
    file: UploadFile = File(...),
    gloss: str = Form(...),
    split: str = Form("unassigned"),
    source: str = Form("upload"),
    username: str = Depends(_auth.require_auth),
):
    data = await file.read()
    if not data:
        raise HTTPException(422, "File rỗng")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_MB} MB")

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    thumb = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        thumb = _thumbnail_from_video(tmp_path)
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
        split=split,
        source=source,
        uploaded_by=username,
        thumbnail=thumb,
    )
    rec["stream_url"] = f"/api/dataset/clips/{rec['clip_id']}/stream"
    return rec


class WebcamClipRequest(BaseModel):
    frames_b64: list[str]
    gloss: str
    split: str = "unassigned"
    fps: int = 12


@router.post("/api/dataset/clips/webcam")
def dataset_webcam(req: WebcamClipRequest, username: str = Depends(_auth.require_auth)):
    if not req.frames_b64:
        raise HTTPException(422, "Không có frame nào")
    video, thumb = _frames_to_webm(req.frames_b64, fps=max(4, min(req.fps, 30)))
    if not video:
        raise HTTPException(422, "Không dựng được video từ frame")
    if thumb is None:
        thumb = _thumbnail_from_frame_b64(req.frames_b64[len(req.frames_b64) // 2])
    rec = ds.add_clip(
        gloss=req.gloss,
        data=video,
        filename=f"webcam_{ds.slugify(req.gloss)}.mp4",
        content_type="video/mp4",
        split=req.split,
        source="webcam",
        uploaded_by=username,
        thumbnail=thumb,
        extra={"frames": len(req.frames_b64), "fps": req.fps},
    )
    rec["stream_url"] = f"/api/dataset/clips/{rec['clip_id']}/stream"
    return rec


class ClipPatch(BaseModel):
    gloss: str | None = None
    split: str | None = None
    note: str | None = None
    verified: bool | None = None


@router.patch("/api/dataset/clips/{clip_id}")
def dataset_patch(clip_id: str, body: ClipPatch, username: str = Depends(_auth.require_auth)):
    rec = ds.update_clip(clip_id, **body.model_dump(exclude_none=True))
    if not rec:
        raise HTTPException(404, "Không tìm thấy clip")
    return rec


@router.delete("/api/dataset/clips/{clip_id}")
def dataset_delete(clip_id: str, username: str = Depends(_auth.require_auth)):
    if not ds.delete_clip(clip_id):
        raise HTTPException(404, "Không tìm thấy clip")
    return {"ok": True, "clip_id": clip_id}


@router.get("/api/dataset/clips/{clip_id}/stream")
def dataset_stream(clip_id: str):
    rec = ds.get_clip(clip_id)
    if not rec:
        raise HTTPException(404, "Không tìm thấy clip")
    gen = storage.objects.stream("dataset", rec["object_key"])
    if gen is None:
        raise HTTPException(404, "Object không tồn tại trong kho")
    return StreamingResponse(
        gen,
        media_type=rec.get("content_type") or "video/mp4",
        headers={
            "Content-Length": str(rec.get("size_bytes") or 0),
            "Cache-Control": "public, max-age=3600",
            "Accept-Ranges": "none",
        },
    )


@router.get("/api/dataset/clips/{clip_id}/thumb")
def dataset_thumb(clip_id: str):
    rec = ds.get_clip(clip_id)
    if not rec or not rec.get("thumb_key"):
        raise HTTPException(404, "Không có thumbnail")
    data = storage.objects.get_bytes("thumbs", rec["thumb_key"])
    if data is None:
        raise HTTPException(404, "Thumbnail không tồn tại")
    return StreamingResponse(
        iter([data]),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/dataset/export.csv", response_class=PlainTextResponse)
def dataset_export():
    return PlainTextResponse(
        ds.export_csv(),
        headers={"Content-Disposition": 'attachment; filename="vsl_dataset.csv"'},
    )


class ImportRequest(BaseModel):
    limit_per_gloss: int = 2
    max_clips: int = 120
    glosses: list[str] | None = None


@router.post("/api/dataset/import-corpus")
def dataset_import(req: ImportRequest, username: str = Depends(_auth.require_auth)):
    label_csv = _REPO / "Dataset" / "Text" / "label_backup.csv"
    if not label_csv.is_file():
        label_csv = _REPO / "Dataset" / "Text" / "label.csv"
    result = ds.import_corpus(
        label_csv,
        _REPO / "Dataset" / "Videos",
        glosses=req.glosses,
        limit_per_gloss=max(1, min(req.limit_per_gloss, 10)),
        max_clips=max(1, min(req.max_clips, 2000)),
    )
    return result
