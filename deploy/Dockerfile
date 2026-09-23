# Backend SignTranslate — build context là thư mục monorepo rút gọn (app/),
# vì backend resolve _REPO = <ctx> và cần LT_SignDiff/ + vsl_hsp_bimamba_k4/ làm sibling.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libGL + libglib cho opencv/mediapipe, ffmpeg để ghi clip webcam ra mp4.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/sign_translate/backend

# Torch CPU-only trước: bản trên PyPI kéo theo ~2.5GB thư viện CUDA không dùng đến.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt ./requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt \
    && python -c "import cv2, numpy, mediapipe, torch; print(cv2.__version__, numpy.__version__, torch.__version__)"

COPY LT_SignDiff /app/LT_SignDiff
COPY sign_translate/backend /app/sign_translate/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
