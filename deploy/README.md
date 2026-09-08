# Triển khai backend lên VPS

Frontend nằm trên Vercel (`https://trans.mophongthuattoan.com`); mọi thứ còn lại chạy
trên VPS `180.93.32.135` tại `https://api.trans.mophongthuattoan.com`.

## Kiến trúc trên server

```
/opt/signtranslate/
├── docker-compose.yml          # backend + redis
├── .env                        # tài khoản, CORS, giới hạn (chmod 600)
├── data/localstore/            # video + thumbnail (thay cho MinIO)
└── app/                        # build context, cũng là _REPO của backend
    ├── Dockerfile
    ├── requirements-prod.txt
    ├── LT_SignDiff/            # package inference v1/v2 + splits + configs
    ├── vsl_hsp_bimamba_k4/     # package HSP-BiMamba
    └── sign_translate/
        ├── backend/
        └── models/             # checkpoint (mount read-only vào container)
```

Backend nghe ở `127.0.0.1:8020`, Caddy (`/etc/caddy/sites/signtranslate.caddy`) làm
reverse proxy và tự cấp chứng chỉ Let's Encrypt.

Không chạy MinIO: VPS chỉ có ~2.5 GB RAM trống nên object store dùng thẳng đĩa qua
`SIGN_TRANSLATE_LOCAL_STORE`. Redis vẫn cần cho phiên đăng nhập, cache suy luận,
metadata dataset và lịch sử.

## Model đã nạp

| Model | Checkpoint | Ghi chú |
|---|---|---|
| `lt_signdiff_v2_top200` | 27 MB | mặc định, 200 gloss, gallery 760 |
| `lt_signdiff_top30_masked` | 122 MB | 30 gloss, dùng cho luồng enroll |
| `hsp_bimamba_top100` | 27 MB | 100 gloss, MP75 |

Toàn bộ model nạp vào RAM lúc khởi động (~570 MB RSS). Muốn thêm model thì copy
checkpoint vào `app/sign_translate/models/<id>/` rồi `docker compose restart backend`;
model nào thiếu file sẽ bị bỏ qua kèm cảnh báo `[WARN] ckpt not found`.

## Vận hành

```bash
cd /opt/signtranslate

docker compose ps                      # trạng thái
docker compose logs -f backend         # log
docker compose restart backend         # nạp lại model / đổi .env
docker compose up -d --build           # sau khi cập nhật code
```

Cập nhật code từ máy dev: đồng bộ `backend/`, `LT_SignDiff/`, `vsl_hsp_bimamba_k4/`
vào `/opt/signtranslate/app/` rồi chạy `docker compose up -d --build`.

Đổi cấu hình Caddy:

```bash
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl reload caddy
```

## Biến môi trường

Xem `.env.example`. Hai biến bắt buộc phải đặt đúng ở production:

- `SIGN_TRANSLATE_USERS` — `"user:pass"` phân tách bằng dấu phẩy. Bỏ trống thì code
  rơi về `admin/admin123`.
- `SIGN_TRANSLATE_CORS_ORIGINS` — danh sách origin của frontend. Bỏ trống hoặc `*`
  sẽ mở cho mọi nguồn.

## Frontend

`VITE_API_URL` được đặt trong `frontend/vercel.json` (`build.env`) nên Vercel không
cần cấu hình thêm. Khi để trống — trường hợp chạy dev — frontend dùng đường dẫn
tương đối và Vite proxy chuyển `/api` sang `localhost:8000`.
