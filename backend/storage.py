# -*- coding: utf-8 -*-
"""Redis cache + MinIO object store cho SignTranslate.

Cả hai đều **tùy chọn**: nếu service không chạy thì module tự chuyển sang chế
độ fallback trong bộ nhớ / đĩa cục bộ, API vẫn hoạt động bình thường. Nhờ vậy
người dùng không bắt buộc phải bật Docker mới chạy được backend.

Biến môi trường
---------------
REDIS_URL              redis://localhost:6379/0    (rỗng = tắt Redis)
MINIO_ENDPOINT         localhost:9000              (rỗng = tắt MinIO)
MINIO_ACCESS_KEY       signtranslate
MINIO_SECRET_KEY       signtranslate123
MINIO_SECURE           0
MINIO_BUCKET_VIDEOS    vsl-videos
MINIO_BUCKET_DATASET   vsl-dataset
MINIO_BUCKET_THUMBS    vsl-thumbs
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

# ─── config ──────────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0").strip()
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000").strip()
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "signtranslate")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "signtranslate123")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "0") not in ("0", "", "false", "False")

BUCKET_VIDEOS = os.environ.get("MINIO_BUCKET_VIDEOS", "vsl-videos")
BUCKET_DATASET = os.environ.get("MINIO_BUCKET_DATASET", "vsl-dataset")
BUCKET_THUMBS = os.environ.get("MINIO_BUCKET_THUMBS", "vsl-thumbs")

LOCAL_FALLBACK_DIR = Path(
    os.environ.get(
        "SIGN_TRANSLATE_LOCAL_STORE",
        str(Path(__file__).resolve().parent / "_localstore"),
    )
)

KEY_PREFIX = "st:"          # namespace mọi key Redis
DEFAULT_TTL = 60 * 60 * 6   # 6 giờ
RETRY_COOLDOWN = 20.0       # giây chờ trước khi thử kết nối lại sau khi rớt


# ═══════════════════════════════════════════════════════════════════════════
# KV: Redis, fallback = LRU dict trong tiến trình
# ═══════════════════════════════════════════════════════════════════════════
class _MemoryKV:
    """Fallback tối giản: LRU + TTL, đủ dùng khi không có Redis."""

    def __init__(self, max_items: int = 4096):
        self._d: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._sets: dict[str, set[str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    @staticmethod
    def _expired(exp: float | None) -> bool:
        return exp is not None and time.time() > exp

    def get(self, key: str):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            val, exp = item
            if self._expired(exp):
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return val

    def set(self, key: str, value, ttl: int | None = None):
        with self._lock:
            self._d[key] = (value, time.time() + ttl if ttl else None)
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def delete(self, *keys: str):
        with self._lock:
            for k in keys:
                self._d.pop(k, None)
                self._sets.pop(k, None)
                self._lists.pop(k, None)

    def incr(self, key: str, ttl: int | None = None) -> int:
        with self._lock:
            val, exp = self._d.get(key, (0, None))
            if self._expired(exp):
                val, exp = 0, None
            val = int(val) + 1
            self._d[key] = (val, exp or (time.time() + ttl if ttl else None))
            return val

    def sadd(self, key: str, *members: str):
        with self._lock:
            self._sets.setdefault(key, set()).update(members)

    def srem(self, key: str, *members: str):
        with self._lock:
            self._sets.get(key, set()).difference_update(members)

    def smembers(self, key: str) -> set[str]:
        with self._lock:
            return set(self._sets.get(key, set()))

    def lpush(self, key: str, value: str, cap: int | None = None):
        with self._lock:
            lst = self._lists.setdefault(key, [])
            lst.insert(0, value)
            if cap:
                del lst[cap:]

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            lst = self._lists.get(key, [])
            return lst[start : (None if stop == -1 else stop + 1)]


class KVStore:
    """Giao diện KV thống nhất; dùng Redis khi có, không thì fallback bộ nhớ."""

    def __init__(self):
        self._mem = _MemoryKV()
        self._redis = None
        self.backend = "memory"
        self.error: str | None = None
        self._next_retry = 0.0
        self._connect()

    def _connect(self) -> None:
        if not REDIS_URL:
            self.error = "REDIS_URL trống — chạy chế độ in-memory"
            return
        self._next_retry = time.time() + RETRY_COOLDOWN
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=2.0,
            )
            client.ping()
            self._redis = client
            self.backend = "redis"
            self.error = None
        except Exception as exc:  # noqa: BLE001 - degrade thay vì chết
            self._redis = None
            self.backend = "memory"
            self.error = f"{type(exc).__name__}: {exc}"

    def reconnect(self) -> bool:
        self._connect()
        return self._redis is not None

    @property
    def available(self) -> bool:
        self._maybe_retry()
        return self._redis is not None

    def _maybe_retry(self) -> None:
        """Redis chết rồi sống lại (ví dụ restart Docker) thì tự nối lại,
        nhưng không thử liên tục để mỗi request khỏi trả giá timeout."""
        if self._redis is None and REDIS_URL and time.time() >= self._next_retry:
            self._connect()

    def _drop_redis(self, exc: Exception) -> None:
        self._redis = None
        self.backend = "memory"
        self.error = f"{type(exc).__name__}: {exc}"
        self._next_retry = time.time() + RETRY_COOLDOWN

    # ── JSON helpers ────────────────────────────────────────────────────────
    def get_json(self, key: str):
        self._maybe_retry()
        key = KEY_PREFIX + key
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        val = self._mem.get(key)
        return json.loads(val) if isinstance(val, str) else val

    def get_many(self, keys: list[str]) -> list:
        """Đọc nhiều key trong một vòng (MGET) — duyệt kho clip là O(1) round-trip
        thay vì một lượt gọi Redis cho mỗi clip."""
        self._maybe_retry()
        if not keys:
            return []
        full = [KEY_PREFIX + k for k in keys]
        if self._redis is not None:
            try:
                raws = self._redis.mget(full)
                out = []
                for raw in raws:
                    try:
                        out.append(json.loads(raw) if raw else None)
                    except Exception:  # noqa: BLE001
                        out.append(None)
                return out
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        vals = []
        for k in full:
            v = self._mem.get(k)
            vals.append(json.loads(v) if isinstance(v, str) else v)
        return vals

    def set_json(self, key: str, value, ttl: int | None = DEFAULT_TTL) -> None:
        self._maybe_retry()
        key = KEY_PREFIX + key
        payload = json.dumps(value, ensure_ascii=False)
        if self._redis is not None:
            try:
                self._redis.set(key, payload, ex=ttl)
                return
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        self._mem.set(key, payload, ttl)

    def delete(self, *keys: str) -> None:
        self._maybe_retry()
        full = [KEY_PREFIX + k for k in keys]
        if not full:
            return
        if self._redis is not None:
            try:
                self._redis.delete(*full)
                return
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        self._mem.delete(*full)

    def incr(self, key: str, ttl: int | None = None) -> int:
        self._maybe_retry()
        key = KEY_PREFIX + key
        if self._redis is not None:
            try:
                n = int(self._redis.incr(key))
                if n == 1 and ttl:
                    self._redis.expire(key, ttl)
                return n
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        return self._mem.incr(key, ttl)

    # ── set ────────────────────────────────────────────────────────────────
    def sadd(self, key: str, *members: str) -> None:
        self._maybe_retry()
        if not members:
            return
        key = KEY_PREFIX + key
        if self._redis is not None:
            try:
                self._redis.sadd(key, *members)
                return
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        self._mem.sadd(key, *members)

    def srem(self, key: str, *members: str) -> None:
        self._maybe_retry()
        if not members:
            return
        key = KEY_PREFIX + key
        if self._redis is not None:
            try:
                self._redis.srem(key, *members)
                return
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        self._mem.srem(key, *members)

    def smembers(self, key: str) -> set[str]:
        self._maybe_retry()
        key = KEY_PREFIX + key
        if self._redis is not None:
            try:
                return set(self._redis.smembers(key))
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        return self._mem.smembers(key)

    # ── list (lịch sử gần đây) ─────────────────────────────────────────────
    def push_history(self, key: str, value, cap: int = 200) -> None:
        self._maybe_retry()
        key = KEY_PREFIX + key
        payload = json.dumps(value, ensure_ascii=False)
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.lpush(key, payload)
                pipe.ltrim(key, 0, cap - 1)
                pipe.execute()
                return
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        self._mem.lpush(key, payload, cap=cap)

    def history(self, key: str, limit: int = 50) -> list:
        self._maybe_retry()
        key = KEY_PREFIX + key
        raws: list[str] = []
        if self._redis is not None:
            try:
                raws = list(self._redis.lrange(key, 0, limit - 1))
            except Exception as exc:  # noqa: BLE001
                self._drop_redis(exc)
        if not raws:
            raws = self._mem.lrange(key, 0, limit - 1)
        out = []
        for r in raws:
            try:
                out.append(json.loads(r))
            except Exception:  # noqa: BLE001
                continue
        return out

    def info(self) -> dict:
        d = {"backend": self.backend, "available": self.available, "url": REDIS_URL or None}
        if self.error:
            d["error"] = self.error
        if self._redis is not None:
            try:
                raw = self._redis.info()
                d.update(
                    {
                        "version": raw.get("redis_version"),
                        "used_memory_human": raw.get("used_memory_human"),
                        "connected_clients": raw.get("connected_clients"),
                        "keys": self._redis.dbsize(),
                        "uptime_days": raw.get("uptime_in_days"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                d["error"] = str(exc)
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Object store: MinIO, fallback = thư mục cục bộ
# ═══════════════════════════════════════════════════════════════════════════
class ObjectStore:
    """Lưu video/clip. MinIO khi có, không thì ghi xuống `_localstore/`."""

    def __init__(self):
        self._client = None
        self.backend = "local"
        self.error: str | None = None
        self.buckets = {
            "videos": BUCKET_VIDEOS,
            "dataset": BUCKET_DATASET,
            "thumbs": BUCKET_THUMBS,
        }
        self._next_retry = 0.0
        self._connect()

    def _connect(self) -> None:
        if not MINIO_ENDPOINT:
            self.error = "MINIO_ENDPOINT trống — lưu cục bộ"
            LOCAL_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
            return
        try:
            from minio import Minio  # type: ignore

            client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_SECURE,
            )
            for b in self.buckets.values():
                if not client.bucket_exists(b):
                    client.make_bucket(b)
            self._client = client
            self.backend = "minio"
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self.backend = "local"
            self.error = f"{type(exc).__name__}: {exc}"
            self._next_retry = time.time() + RETRY_COOLDOWN
            LOCAL_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

    def reconnect(self) -> bool:
        self._connect()
        return self._client is not None

    def _maybe_retry(self) -> None:
        """MinIO sống lại sau khi restart thì tự nối lại, có thời gian chờ
        để không phải trả giá timeout ở mọi request."""
        if self._client is None and MINIO_ENDPOINT and time.time() >= self._next_retry:
            self._connect()

    def _drop_client(self, exc: Exception) -> None:
        self._client = None
        self.backend = "local"
        self.error = f"{type(exc).__name__}: {exc}"
        self._next_retry = time.time() + RETRY_COOLDOWN

    @property
    def available(self) -> bool:
        self._maybe_retry()
        return self._client is not None

    def _local_path(self, bucket: str, key: str) -> Path:
        p = LOCAL_FALLBACK_DIR / bucket / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── ghi ────────────────────────────────────────────────────────────────
    def put_bytes(
        self,
        bucket_alias: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> dict:
        self._maybe_retry()
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is not None:
            try:
                self._client.put_object(
                    bucket,
                    key,
                    io.BytesIO(data),
                    length=len(data),
                    content_type=content_type,
                    metadata=metadata or None,
                )
                return {"bucket": bucket, "key": key, "size": len(data), "backend": "minio"}
            except Exception as exc:  # noqa: BLE001
                self._drop_client(exc)
        path = self._local_path(bucket, key)
        path.write_bytes(data)
        return {"bucket": bucket, "key": key, "size": len(data), "backend": "local"}

    def put_file(
        self,
        bucket_alias: str,
        key: str,
        file_path: str | Path,
        content_type: str = "video/mp4",
    ) -> dict:
        return self.put_bytes(bucket_alias, key, Path(file_path).read_bytes(), content_type)

    # ── đọc ────────────────────────────────────────────────────────────────
    def get_bytes(self, bucket_alias: str, key: str) -> bytes | None:
        self._maybe_retry()
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is not None:
            resp = None
            try:
                resp = self._client.get_object(bucket, key)
                return resp.read()
            except Exception:  # noqa: BLE001
                return None
            finally:
                if resp is not None:
                    resp.close()
                    resp.release_conn()
        path = self._local_path(bucket, key)
        return path.read_bytes() if path.is_file() else None

    def stream(self, bucket_alias: str, key: str, chunk: int = 1 << 18) -> Iterator[bytes] | None:
        """Generator cho StreamingResponse — phát video không nạp hết vào RAM."""
        self._maybe_retry()
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is not None:
            try:
                resp = self._client.get_object(bucket, key)
            except Exception:  # noqa: BLE001
                return None

            def _gen():
                try:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        yield buf
                finally:
                    resp.close()
                    resp.release_conn()

            return _gen()
        path = self._local_path(bucket, key)
        if not path.is_file():
            return None

        def _gen_local():
            with open(path, "rb") as f:
                while True:
                    buf = f.read(chunk)
                    if not buf:
                        break
                    yield buf

        return _gen_local()

    def presigned_url(self, bucket_alias: str, key: str, minutes: int = 60) -> str | None:
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is None:
            return None
        try:
            return self._client.presigned_get_object(bucket, key, expires=timedelta(minutes=minutes))
        except Exception:  # noqa: BLE001
            return None

    def exists(self, bucket_alias: str, key: str) -> bool:
        self._maybe_retry()
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is not None:
            try:
                self._client.stat_object(bucket, key)
                return True
            except Exception:  # noqa: BLE001
                return False
        return self._local_path(bucket, key).is_file()

    def remove(self, bucket_alias: str, key: str) -> bool:
        self._maybe_retry()
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        if self._client is not None:
            try:
                self._client.remove_object(bucket, key)
                return True
            except Exception:  # noqa: BLE001
                return False
        p = self._local_path(bucket, key)
        if p.is_file():
            p.unlink()
            return True
        return False

    def list_keys(self, bucket_alias: str, prefix: str = "", limit: int = 1000) -> list[dict]:
        bucket = self.buckets.get(bucket_alias, bucket_alias)
        out: list[dict] = []
        if self._client is not None:
            try:
                for obj in self._client.list_objects(bucket, prefix=prefix, recursive=True):
                    out.append(
                        {
                            "key": obj.object_name,
                            "size": obj.size,
                            "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                        }
                    )
                    if len(out) >= limit:
                        break
                return out
            except Exception:  # noqa: BLE001
                return out
        base = LOCAL_FALLBACK_DIR / bucket
        if base.is_dir():
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(base).as_posix()
                if not rel.startswith(prefix):
                    continue
                st = p.stat()
                out.append(
                    {
                        "key": rel,
                        "size": st.st_size,
                        "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                    }
                )
                if len(out) >= limit:
                    break
        return out

    def usage(self) -> dict:
        """Tổng dung lượng theo bucket (đủ nhanh cho vài nghìn object)."""
        stats = {}
        for alias in self.buckets:
            objs = self.list_keys(alias, limit=100000)
            stats[alias] = {
                "objects": len(objs),
                "bytes": sum(int(o["size"] or 0) for o in objs),
            }
        return stats

    def info(self) -> dict:
        host = MINIO_ENDPOINT.split(":")[0] if MINIO_ENDPOINT else None
        d = {
            "backend": self.backend,
            "available": self.available,
            "endpoint": MINIO_ENDPOINT or None,
            "buckets": self.buckets,
            "console": f"http://{host}:9001" if host else None,
        }
        if self.error:
            d["error"] = self.error
        if not self.available:
            d["local_dir"] = str(LOCAL_FALLBACK_DIR)
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Singletons + tiện ích
# ═══════════════════════════════════════════════════════════════════════════
kv = KVStore()
objects = ObjectStore()


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def cache_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def health() -> dict:
    return {"redis": kv.info(), "object_store": objects.info()}
