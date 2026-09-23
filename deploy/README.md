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
    ├── LT_SignDiff/            # package inference + splits + configs
    └── sign_translate/
        ├── backend/
        └── models/             # checkpoint (mount read-only vào container)
```

Backend nghe ở `127.0.0.1:8020`, Caddy (`/etc/caddy/sites/signtranslate.caddy`) làm
reverse proxy và tự cấp chứng chỉ Let's Encrypt.

Không chạy MinIO: VPS chỉ có ~2.5 GB RAM trống nên object store dùng thẳng đĩa qua
`SIGN_TRANSLATE_LOCAL_STORE`. Redis vẫn cần cho phiên đăng nhập, cache suy luận,
metadata dataset và lịch sử.

## Model

Dịch vụ chạy duy nhất `lt_signdiff_v2_top200` (27 MB, 200 gloss, gallery 760 tham
chiếu, TTA bật). Các kiến trúc khác trong repo — CurriVSL, WBDPNet, HSP-BiMamba —
chỉ còn dùng để so sánh trong bài báo, không nằm trong đường chạy của dịch vụ và
không được copy lên server.

Checkpoint thiếu file thì startup chỉ in `[WARN] ckpt not found` rồi chạy tiếp với
`model_loaded: false`; kiểm tra bằng `curl -s localhost:8020/api/health`.

## Kho video từ vựng

Clip mẫu của tab *Từ vựng* nằm chung object store với dataset huấn luyện nhưng mang
`source="library"`, nên bị loại khỏi `/api/dataset/clips`, `/api/dataset/stats` và
`export.csv`. Tiến độ học lưu ở Redis theo tài khoản (`lib:prog:<username>`).

Nạp clip mẫu từ máy dev (cần `Dataset/Text/label.csv` + `Dataset/Videos/`):

```bash
python seed_library.py --dry-run          # xem kế hoạch
python seed_library.py --per-gloss 2      # nạp, bỏ qua từ đã đủ clip
```

Hiện trạng: 200/200 từ có clip mẫu, 399 clip, ~330 MB trong `data/localstore/`.

## Vận hành

```bash
cd /opt/signtranslate

docker compose ps                      # trạng thái
docker compose logs -f backend         # log
docker compose restart backend         # nạp lại model / đổi .env
docker compose up -d --build           # sau khi cập nhật code
```

Cập nhật code từ máy dev: đồng bộ `backend/` và `LT_SignDiff/` vào
`/opt/signtranslate/app/` rồi chạy `docker compose up -d --build`.

Đĩa của VPS dùng chung với 7 project khác và hay chạm ngưỡng 90%. Dọn an toàn bằng
`docker builder prune -f` (chỉ xoá cache lơ lửng, không ảnh hưởng image đang dùng).

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
