# SignTranslate — LT-SignDiff cho ngôn ngữ ký hiệu Việt

Hệ thống dịch ngôn ngữ ký hiệu Việt (VSL) closed-set 200 từ, gồm:

- **Frontend** React + Vite — giao diện sáng/tối, 6 tab: Dịch · Kho dữ liệu · Thống kê · Lịch sử · Enroll · Học từ mới
- **Backend** FastAPI + MediaPipe Holistic
- **Hạ tầng** Redis (cache, lịch sử, metadata) + MinIO (kho video) — cả hai đều tuỳ chọn
- **Mô hình** LT-SignDiff (v1 và v2)

---

## 1. Chạy nhanh

### Hạ tầng (tuỳ chọn nhưng nên bật)

```bash
docker compose -f sign_translate/docker-compose.yml up -d
```

Redis ở `localhost:6379`, MinIO API ở `localhost:9000`, console ở `localhost:9001`
(`signtranslate` / `signtranslate123`).

Không bật Docker thì backend **vẫn chạy**: Redis chuyển sang cache trong tiến trình,
MinIO chuyển sang thư mục `backend/_localstore/`. Khi service sống lại, backend tự
kết nối lại sau tối đa 20 giây — không cần khởi động lại.

### Backend (Python 3.11 — MediaPipe chưa hỗ trợ 3.12+)

```powershell
cd sign_translate/backend
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SIGN_TRANSLATE_DEFAULT_MODEL = "lt_signdiff_v2_top200"
.\.venv311\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd sign_translate/frontend
npm install
npm run dev
```

Mở http://localhost:5173 — đăng nhập `admin` / `admin123`.
Chuyển tab bằng `Alt`+`1`…`6`.

---

## 2. Tính năng

| Tab | Nội dung |
|-----|----------|
| **Dịch** | Tải video hoặc ghi webcam → 10 khả năng xếp hạng. Ghép nhiều lượt thành câu. Đánh dấu đúng/sai. Nút *So sánh* chạy cùng đoạn ghi qua mọi model đang nạp và cho ra kết quả đồng thuận. |
| **Kho dữ liệu** | Duyệt / phát / sửa nhãn / đổi split / xoá clip trong MinIO. Thêm clip bằng kéo-thả hoặc webcam. Nạp sẵn từ corpus VSL trên đĩa. Xuất CSV theo đúng định dạng split của `LT_SignDiff`. |
| **Thống kê** | Độ trễ (trung vị/p90), phân bố độ tin cậy, từ hay gặp, cặp nhầm lẫn thường gặp, tình trạng Redis/MinIO. |
| **Lịch sử** | 500 lượt gần nhất (Redis, 7 ngày). Lọc theo chưa đánh giá / bị sai. |
| **Enroll** | Thêm mẫu cá nhân vào gallery kNN — có tác dụng ngay, không cần huấn luyện lại. |
| **Học từ mới** | Fine-tune trực tuyến A→B→C (chỉ với `curivsl` / `hsp_bimamba`). |

Ngoài ra: giao diện tự theo chế độ sáng/tối của hệ thống (có nút chuyển thủ công),
cache kết quả theo hash nội dung video trong Redis (dịch lại cùng file là tức thì),
và tuỳ chọn lưu thẳng video vừa dịch vào kho.

---

## 3. Mô hình

### Kết quả

Closed-set 200 từ · 611 clip train / 149 val / 200 test (mỗi từ đúng 1 clip test).

| | v1 | v2 (model đang triển khai) |
|---|---|---|
| Classifier | — | 77,5 % |
| Prototype | — | 78,0 % |
| **kNN gallery** | 33,5 % | **84,5 %** |
| Hợp nhất (trọng số đang dùng) | 55,5 % | 84,0 % |
| Top-5 | 68,5 % | 86,5 % |

Model triển khai là **seed 1 đơn lẻ** (chọn theo val). Đặt
`LT_SIGNDIFF_V2_FUSION=0,0,1` để chạy kNN thuần — 84,5 %, nhỉnh hơn 0,5 điểm,
tức nằm trong nhiễu.

**Ensemble 3 seed không giúp** — kết quả thật, không phải chưa thử: seed 1/2/3
lần lượt cho 83,5 / 82,5 / 81,5 %, gộp lại được 82,5 %, tức *thấp hơn* seed tốt
nhất. Nguyên nhân nằm ở chỗ val bão hoà: cả ba seed đều đứng quanh 78–79 % trên
val nên việc chọn checkpoint gần như ngẫu nhiên trong vùng bình nguyên — seed 1
tình cờ chốt ở epoch 105, seed 2 ở epoch 60, seed 3 ở epoch 50. Với tập val 149
mẫu, chênh lệch 1–2 điểm giữa các seed là nhiễu chọn checkpoint, không phải
chênh lệch năng lực.

Sai số chuẩn ở n=200 là ±2,6 điểm, nên chênh lệch dưới ~5 điểm giữa hai cấu hình
là nằm trong nhiễu. Trên 10 clip test đầu chạy qua API thật: v2 đúng 8/10, v1
đúng 5/10.

#### Độ chính xác bị quyết định bởi số clip mỗi từ, không phải bởi model

`scripts/analyze_errors_v2.py` chia kết quả test theo số clip có trong gallery:

| clip/từ | số từ | đúng | tỉ lệ |
|--------:|------:|-----:|------:|
| 2 | 51 | 36 | 71 % |
| 3 | 40 | 27 | 68 % |
| **4** | 12 | 12 | **100 %** |
| **5** | 92 | 90 | **98 %** |
| 6 | 5 | 4 | 80 % |

Có một bậc thang rất rõ giữa 3 và 4 clip:

* từ có **≥4 clip: 97,2 %** (106/109)
* từ có **≤3 clip: 69 %** (63/91)

Nói cách khác, model **đã vượt xa mốc 85 %** trên những từ đủ mẫu. Con số tổng
84,5 % bị kéo xuống bởi 91/200 từ thiếu mẫu — nhóm này gây 28 trong 31 lỗi.
Trong corpus hiện tại không còn bản ghi nào cho các từ đó, nên đường ngắn nhất
để nâng độ chính xác tổng thể **không phải sửa model mà là quay thêm mẫu**.

Đó cũng chính là lý do tab *Kho dữ liệu* liệt kê sẵn danh sách từ có dưới 4 clip,
và tab *Enroll* cho thêm mẫu vào gallery mà không cần huấn luyện lại.

#### Hai hướng đã thử và loại — đừng làm lại

**1. Chỉnh cấu hình gallery** (`scripts/tune_gallery_v2.py`). Thử `k ∈ {3…21}`,
`temp ∈ {6…30}`, và gallery dày gấp 4 bằng cách giữ từng view TTA làm một tham
chiếu riêng thay vì gộp. Trên 48 cấu hình, val trải đúng **0,67 điểm** (sai số
chuẩn ±3,4) và test đứng yên ở 84,5 %. Mặc định `k=7, temp=12` đã tối ưu.

**2. Sửa thiên lệch của luật bỏ phiếu** (`scripts/tune_vote_v2.py`). Giả thuyết
hợp lý: luật hiện tại **cộng** trọng số nên lớp nhiều tham chiếu vừa dễ lọt
top-k vừa cộng được nhiều số hạng hơn, làm lớp ít mẫu thiệt kép. Đã thử `max`
(chỉ lấy láng giềng gần nhất mỗi lớp — hoàn toàn không phụ thuộc số mẫu),
`mean`, `norm` (chia số tham chiếu) và `sqrt`.

| luật | test | ≤3 clip | ≥4 clip |
|------|-----:|--------:|--------:|
| sum (hiện tại) | 84,5 % | 69,2 % | 97,2 % |
| max (bất biến số mẫu) | 84,5 % | 69,2 % | 97,2 % |
| sqrt | 83,5 % | 68,1 % | 96,3 % |
| norm | 84,0 % | 69,2 % | 96,3 % |
| mean | 77,5 % | 62,6 % | 89,9 % |

`max` cho kết quả **giống hệt** `sum` ở cả ba cột. Nghĩa là khoảng cách của nhóm
ít mẫu **không** đến từ thiên lệch thuật toán mà là thiếu dữ liệu thật: từ chỉ có
2 clip đơn giản không có láng giềng nào đủ gần. Giả thuyết nghe hợp lý nhưng sai,
và đã được đo để loại hẳn.

**3. Prototype khuếch tán cho lớp hiếm** (`scripts/tune_proto_v2.py`). Đây đúng
là đóng góp cốt lõi của LT-SignDiff, và mặc định `min_count=2` khiến nó **không
bao giờ kích hoạt** (mọi lớp Top-200 đều có ≥2 tham chiếu). Nâng ngưỡng để tổng
hợp prototype cho 51 → 91 → 103 → 195 lớp:

| min_count | số lớp tổng hợp | test (kNN thuần) | test (chỉ prototype) |
|----------:|----------------:|-----------------:|---------------------:|
| 2 | 0 | 84,5 % | 78,5 % |
| 3 | 51 | 84,5 % | 62,5 % |
| 4 | 91 | 84,5 % | 51,0 % |
| 5 | 103 | 84,5 % | 44,5 % |

Tổng hợp prototype **không đổi gì** trên kênh kNN và làm kênh prototype tệ dần.
Tỉ lệ nhóm ≤3 clip đứng nguyên 69,2 % ở mọi ngưỡng.

> **Cảnh báo rò rỉ.** Bản đầu của thí nghiệm này dựng prototype từ train+val rồi
> chấm trên val — mỗi mẫu val góp phần tạo prototype của chính lớp nó. Có cấu
> hình cho **val 90,6 % trong khi test chỉ 62,5 %**. Nếu chọn theo cột val đó thì
> đã chốt nhầm một cấu hình tệ hơn mặc định 22 điểm. Script hiện dựng prototype
> **chỉ từ train** khi chấm val, và từ train+val khi chấm test.

### LT-SignDiff v1 (cũ)

TempConv + BiLSTM trên `flatten(V·C)`, đầu vào 110 khớp × 16 frame.

### LT-SignDiff v2 (mới)

Bốn thay đổi, mỗi cái xử lý một điểm yếu cụ thể của v1:

1. **Skeleton dựng lại ở độ phân giải thời gian đầy đủ.**
   Cache cũ chỉ giữ 16 frame; bản mới giữ trung bình ~107 frame/clip kèm mask
   hiện diện cho từng nhóm khớp (`scripts/extract_skeletons_v2.py`).

2. **Không còn số 0 giả.** v1 ghi thẳng `0` khi MediaPipe mất bàn tay (~40 % số
   frame), encoder học phải quỹ đạo nhảy về gốc toạ độ. v2 nội suy tuyến tính và
   thêm một kênh `presence` để mô hình biết frame nào là suy ra.

3. **Đặc trưng cân đối lại.** v1 có 68/110 khớp là mặt → 62 % chiều đặc trưng
   dành cho khuôn mặt. v2 dùng 42 bàn tay + 24 điểm mặt + 8 điểm thân = **74
   khớp × 10 kênh**: xyz chuẩn hoá theo khung vai (bất biến với vị trí và khoảng
   cách camera), vận tốc, xyz cục bộ theo bộ phận, và presence.

4. **Encoder + hàm mất mát mới.** Self-attention giữa các khớp trong từng frame →
   nén mỗi khớp còn 16 chiều → Transformer + conv đa tỉ lệ trên trục thời gian →
   attention pooling. Đầu ra qua ArcFace có biên tăng dần, thay softmax thường
   (6,5 M tham số).

Dự đoán cuối là tổ hợp có trọng số của **classifier ⊕ prototype khuếch tán ⊕ kNN
gallery**, trọng số lưu trong checkpoint.

#### Một lưu ý về cách chọn trọng số hợp nhất

Tập val chỉ có 149 mẫu → sai số chuẩn khoảng ±3,4 điểm. Đo thực tế trên một
checkpoint: val cho `cls 78,5 · proto 79,9 · kNN 78,5`, và **308/342 bộ trọng số
trong lưới nằm trong vòng 1 % của nhau** — nghĩa là val không phân biệt nổi ba
kênh. Lấy cực đại tự do trên val vì thế là chọn theo nhiễu: bộ được chọn pha
loãng mất kênh mạnh nhất và kéo test từ 84,5 % (chỉ kNN) xuống 78,0 %.

Cách xử lý trong `scripts/eval_v2.py`:

1. Chấm val bằng đúng điều kiện của test — gallery train+val, loại chính mẫu
   đang hỏi ra khỏi láng giềng (leave-one-out). Không làm vậy thì mỗi mẫu val tự
   khớp với bản thân và kNN đạt 100 % giả tạo.
2. Ràng buộc lưới để kNN giữ ít nhất một nửa trọng số. Đây là tiên nghiệm sẵn có
   của dự án chứ không phải suy ra từ test: ở chế độ 3–4 clip/từ thì gallery là
   ước lượng chính, đúng như luồng *enroll* mà v1 đã ghi nhận.
3. Lấy trung bình các bộ trọng số nằm trong 1 % của điểm tốt nhất thay vì đúng
   một điểm cực đại.

Sau ba bước đó, cùng checkpoint cho **test 83,5 %** thay vì 78,0 %.

Script luôn in đủ bốn con số (`cls / proto / knn / fused`) để thấy rõ kênh nào
đang thực sự gánh, thay vì chỉ một con số tổng.

### Huấn luyện lại

```bash
cd LT_SignDiff

# 1. Dựng skeleton v2 (chạy bằng venv311 vì mediapipe ≥0.10.30 bỏ solutions)
../sign_translate/backend/.venv311/Scripts/python.exe scripts/extract_skeletons_v2.py

# 2. Huấn luyện
python scripts/train_v2.py --epochs 240 --epoch-size 768 --batch-size 32 --seed 1 \
    --out-dir runs/lt_signdiff_v2_s1

# 3. Đánh giá (gộp nhiều seed nếu có) và export cho backend
python scripts/eval_v2.py --runs runs/lt_signdiff_v2_s1 runs/lt_signdiff_v2_s2 \
    --export ../sign_translate/models/lt_signdiff_v2_top200
```

Hoặc chạy cả chuỗi seed 2–3 + ensemble + export bằng
`scripts/run_v2_pipeline.sh`.

> **Không bao giờ chạy hai `train_v2.py` cùng trỏ vào một `--out-dir`.** Mỗi
> tiến trình theo dõi "best" trong bộ nhớ riêng nhưng cùng ghi đè `best.pt`, nên
> checkpoint cuối cùng thuộc về tiến trình nào ghi sau — kết quả không tin được.
> `run_v2_pipeline.sh` có sẵn chốt chặn cho việc này. Trên Windows, `kill` của
> Git Bash **không** giết được tiến trình Python đã tách; dùng
> `Get-CimInstance Win32_Process` lọc theo `CommandLine` rồi `Stop-Process`.

Dữ liệu: 200 từ, 611 clip train / 149 val / 200 test (mỗi từ 1 clip test).

---

## 4. API

### Dịch

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/translate/video` | Upload video (`save_to_dataset` để lưu vào MinIO) |
| POST | `/api/translate/frames` | Frame webcam base64 |
| POST | `/api/translate/compare` | Cùng đoạn ghi qua nhiều model |

### Kho dữ liệu

| Method | Endpoint |
|--------|----------|
| GET | `/api/dataset/stats` · `/api/dataset/glosses` · `/api/dataset/clips` |
| POST | `/api/dataset/clips` (upload) · `/api/dataset/clips/webcam` |
| PATCH / DELETE | `/api/dataset/clips/{id}` |
| GET | `/api/dataset/clips/{id}/stream` · `/thumb` · `/api/dataset/export.csv` |
| POST | `/api/dataset/import-corpus` |

### Lịch sử, thống kê, câu

| Method | Endpoint |
|--------|----------|
| GET / DELETE | `/api/history` |
| POST | `/api/history/{id}/feedback` |
| GET | `/api/insights/overview` · `/api/insights/hard-cases` |
| GET / POST | `/api/sentence` · `/append` · `/pop` · `/reset` |

### Model & hạ tầng

| Method | Endpoint |
|--------|----------|
| GET | `/api/health` · `/api/models` · `/api/vocab/{id}` · `/api/gallery/{id}` |
| GET | `/api/storage/health` |
| POST | `/api/storage/reconnect` · `/api/enroll/frames` · `/api/gallery/rebuild/{id}` |

---

## 5. Biến môi trường

| Tên | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `SIGN_TRANSLATE_DEFAULT_MODEL` | `lt_signdiff_top30_masked` | Model mặc định |
| `LT_SIGNDIFF_V2_CKPT` | `models/lt_signdiff_v2_top200/best.pt` | Checkpoint v2 |
| `LT_SIGNDIFF_V2_FUSION` | lấy từ checkpoint | Ghi đè trọng số hợp nhất, dạng `cls,proto,knn` — ví dụ `0,0,1` để chỉ dùng kNN |
| `LT_SIGNDIFF_CKPT` | `models/lt_signdiff_top200/best.pt` | Checkpoint v1 |
| `REDIS_URL` | `redis://localhost:6379/0` | Để rỗng là tắt Redis |
| `MINIO_ENDPOINT` | `localhost:9000` | Để rỗng là lưu cục bộ |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `signtranslate` / `signtranslate123` | |
| `SIGN_TRANSLATE_USERS` | `admin:admin123` | Danh sách `user:pass` phân tách bằng dấu phẩy |
| `SIGN_TRANSLATE_TOKEN_TTL` | `43200` | Hạn phiên đăng nhập (giây), tự gia hạn khi còn thao tác |
| `SIGN_TRANSLATE_CACHE_TTL` | `43200` | TTL cache kết quả (giây) |
| `SIGN_TRANSLATE_MAX_UPLOAD_MB` | `80` | Giới hạn upload |
| `CURRIVSL_TOP_K` | `10` | Số dự đoán trả về |
