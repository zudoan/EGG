# EEG Alcoholism Classification (SMNI_CMI)

## Cấu trúc project

- `phat.ipynb`: notebook gốc (GIỮ NGUYÊN, không xoá)
- `SMNI_CMI_TRAIN/Train`, `SMNI_CMI_TEST/Test`: dữ liệu
- `src/`: mã nguồn Python đã tách theo pipeline
- `saved_models/`: nơi lưu model + scaler sau khi chạy
 - `web/`: dashboard React (multi-page + 3D)

## Chạy pipeline end-to-end

```bash
python -m src.main
```

## Chạy Web Dashboard

### 1) Chạy API (FastAPI)

Mở terminal tại thư mục project:

```bash
python -m uvicorn src.app:app --reload --port 8000
```

Mở docs:

- `http://127.0.0.1:8000/docs`

### 2) Chạy Web (React)

Mở terminal thứ 2:

```bash
npm install
npm run dev
```

Sau đó mở:

- `http://127.0.0.1:5173`

## Ghi chú

- Pipeline dùng `SMNI_CMI_TRAIN` làm train và `SMNI_CMI_TEST` làm test.
- Có lọc trùng nội dung train/test bằng MD5 (config: `REMOVE_TEST_DUPLICATES`).
- Sẽ train các model bandpower + (nếu có TensorFlow) các model spectrogram.
