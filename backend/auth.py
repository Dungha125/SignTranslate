# -*- coding: utf-8 -*-
"""Xác thực bằng token đơn giản, phiên lưu trong Redis.

Trước đây token nằm trong một dict của tiến trình, nên mỗi lần khởi động lại
backend (nạp model mới, sửa code, uvicorn --reload) là toàn bộ người dùng bị
đăng xuất. Redis đã có sẵn trong hệ thống nên phiên đặt ở đó hợp lý hơn; khi
Redis không chạy thì `storage.KVStore` tự lùi về bộ nhớ tiến trình — hành vi
đúng bằng bản cũ, không tệ hơn.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time

from fastapi import Header, HTTPException

from storage import kv

TOKEN_TTL = int(os.environ.get("SIGN_TRANSLATE_TOKEN_TTL", str(3600 * 12)))


def _h(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# Mật khẩu mặc định chỉ dành cho bản demo cục bộ; đặt SIGN_TRANSLATE_USERS
# dạng "user1:pass1,user2:pass2" để thay.
def _load_users() -> dict[str, str]:
    raw = os.environ.get("SIGN_TRANSLATE_USERS", "").strip()
    if not raw:
        return {"admin": _h("admin123")}
    users: dict[str, str] = {}
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        name, _, pw = pair.partition(":")
        name, pw = name.strip(), pw.strip()
        if name and pw:
            users[name] = _h(pw)
    return users or {"admin": _h("admin123")}


USERS: dict[str, str] = _load_users()


def _key(token: str) -> str:
    return f"auth:{token}"


def login(username: str, password: str) -> str | None:
    if USERS.get(username) != _h(password):
        return None
    token = secrets.token_hex(32)
    kv.set_json(_key(token), {"username": username, "at": time.time()}, ttl=TOKEN_TTL)
    return token


def verify_token(token: str) -> str | None:
    data = kv.get_json(_key(token))
    if not data:
        return None
    # Chạm vào phiên là gia hạn — người dùng đang thao tác thì không nên bị đá ra.
    kv.set_json(_key(token), data, ttl=TOKEN_TTL)
    return str(data.get("username")) or None


def logout(token: str) -> None:
    kv.delete(_key(token))


def require_auth(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Cần đăng nhập")
    username = verify_token(authorization[7:])
    if not username:
        raise HTTPException(401, "Token không hợp lệ hoặc đã hết hạn")
    return username
