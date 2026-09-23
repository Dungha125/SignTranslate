# -*- coding: utf-8 -*-
"""Smoke test cục bộ cho kho từ vựng.

Chạy uvicorn ở cửa sổ khác rồi:  python _smoke_library.py [base_url]
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
TOKEN = None


def call(method, path, *, params=None, body=None):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


st, h = call("GET", "/api/health")
print("health:", st, h.get("status"), h.get("models_loaded"), h.get("default_model"))

st, r = call("POST", "/api/auth/login", body={"username": "admin", "password": "admin123"})
print("login:", st)
if st != 200:
    print(r)
    raise SystemExit(1)
TOKEN = r["token"]

print("overview:", *call("GET", "/api/library/overview"))

st, w = call("GET", "/api/library/words", params={"limit": 5})
print("words:", st, w["total"], [x["gloss"] for x in w["words"]])

g = w["words"][0]["gloss"]
st, d = call("GET", "/api/library/word", params={"gloss": g})
print("word:", st, d.get("gloss"), "clips:", len(d.get("clips", [])), "in_vocab:", d.get("in_vocab"))

for f in ("all", "has_video", "no_video", "learned", "learning", "todo"):
    st, x = call("GET", "/api/library/words", params={"filter": f, "limit": 1})
    print(f"  filter {f:10s} -> {st} {x.get('total')}")

print("bad gloss:", call("GET", "/api/library/word", params={"gloss": "khong-ton-tai"})[0])
print("practice ngoài vocab:", call("POST", "/api/library/practice", body={"frames_b64": [], "gloss": "zzz"})[0])
print("words không auth:", (lambda t: t)(None) or "")
TOKEN = None
print("  ->", call("GET", "/api/library/words")[0])
