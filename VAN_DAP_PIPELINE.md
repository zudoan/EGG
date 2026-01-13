# Tài liệu vấn đáp — EEG Alcoholism Classification (SMNI_CMI)

## 1) Mục tiêu bài toán
- **Bài toán**: phân loại một phiên EEG (một file CSV ~ một trial) thuộc nhóm:
  - **Control**: nhãn `c` → mã hoá `0`
  - **Alcoholic**: nhãn `a` → mã hoá `1`
- **Input**: tín hiệu EEG đa kênh theo thời gian (nhiều điện cực, nhiều mẫu thời gian).
- **Output**: `pred_label ∈ {0,1}` và `pred_prob` (xác suất thuộc lớp `1`).

## 2) Cấu trúc project (đúng pipeline)
### Root
- `README.md`: hướng dẫn chạy nhanh.
- `requirements.txt`: thư viện Python.
- `saved_models/`: nơi lưu model, scaler, EDA json, và các file phụ trợ phục vụ web.
- `src/`: mã nguồn Python theo pipeline.
- `web/`: dashboard React.

### `src/` (pipeline Python)
- `src/config.py`
  - Khai báo hằng số (sampling rate, số mẫu, dải tần, đường dẫn glob...).
- `src/preprocessing.py`
  - Đọc CSV → chuẩn hoá dataframe → dựng ma trận kênh×thời gian.
  - Xử lý thiếu, trích xuất PSD/bandpower/spectrogram.
  - **Outlier clipping** utilities: `fit_clip_bounds`, `apply_clip_bounds`.
- `src/feature_engineering.py`
  - Ghép nhiều file thành ma trận đặc trưng train/test.
  - Loại trùng train/test bằng MD5 (giảm rò rỉ).
- `src/eda.py`
  - Thống kê phân bố nhãn train/test.
- `src/diagnostics.py`
  - Kiểm tra chất lượng dữ liệu (`nan/inf`, skew, outlier rate theo IQR/z-score).
  - Chẩn đoán lệch phân phối train/test.
- `src/model_phat.py`
  - Huấn luyện các mô hình:
    - Nhóm **classical** trên **bandpower vector**.
    - Nhóm **deep learning** trên **spectrogram image** (nếu có TensorFlow).
- `src/evaluation.py`
  - Tính metric (accuracy, AUC, report, confusion matrix).
  - Export model/scaler ra `saved_models/`.
- `src/main.py`
  - Entry-point chạy end-to-end pipeline: build dataset → clip outlier → train → evaluate → lưu artifacts.
- `src/app.py`
  - FastAPI backend phục vụ web: `/predict`, `/analyze`, `/eda`, `/metrics`, `/models`.

### `web/` (dashboard)
- `web/src/pages/` chứa từng trang.
  - `GioiThieu.tsx`: biểu đồ tổng quan train/test, phân bố nhãn.
  - `DuDoan.tsx`: upload file → gọi `/predict` + `/analyze` → vẽ time-series/PSD/bandpower/spectrogram.
  - `KienThucEEG.tsx`: sơ đồ điện cực 2D + kiến thức dải tần.
- `web/src/components/`
  - Component biểu đồ/visual chung, ví dụ `ElectrodeMap2D.tsx`.
- `web/src/lib/`
  - `api.ts`: wrapper gọi API.
  - `eeg_electrode_knowledge.ts`: kiến thức điện cực + nguồn.
  - `eeg_knowledge.ts`: kiến thức dải tần + nguồn.

## 3) Pipeline xử lý dữ liệu (theo đúng flow)

### Bước A — Chuẩn hoá dữ liệu thô (CSV → long dataframe)
**File**: `src/preprocessing.py` → `normalize_long_df(df_raw)`
- Đổi tên cột về dạng chuẩn, strip/lower.
- Map nhãn: `subject identifier` (`a/c`) → `label` (`1/0`).

**Ý nghĩa**: đưa dữ liệu về định dạng nhất quán trước khi pivot thành ma trận.

### Bước B — Dựng ma trận EEG (channel × time)
**File**: `src/preprocessing.py` → `build_epoch_matrix_from_long(df_long, trial_number)`
- Pivot thành `mat[channel, sample_num]`.
- Reindex đủ `N_SAMPLES`.
- **Xử lý thiếu**:
  - Loại kênh có missing ratio > 5%.
  - Nội suy theo thời gian + `bfill/ffill`.

**Ý nghĩa**:
- Mô hình/đặc trưng cần tensor số đều kích thước.
- EEG thực tế hay thiếu/đứt đoạn; phải có policy rõ.

### Bước C — Trích xuất đặc trưng (feature engineering)
**File**: `src/preprocessing.py` → `extract_features_from_file(file_path)`

1) **Time-domain (TD) cơ bản**
- mean, std, min, max theo từng kênh.

2) **PSD (Welch)**
- `welch_psd_features(x_ch_t)`

3) **Bandpower theo dải**
- `bandpower(psd, freqs, band)`
- Tạo feature dạng: `bp_{band}_{channel}`.

4) **Spectrogram**
- `spectrogram_tf(...)` → lấy log10 và mean qua kênh.

**Ý nghĩa**:
- TD: mô tả biên độ chung.
- PSD/Bandpower: EEG rất giàu thông tin miền tần số.
- Spectrogram: giữ thông tin biến thiên theo thời gian (time–frequency).

### Bước D — Ghép dataset train/test
**File**: `src/feature_engineering.py`
- `build_dataset_from_files(files_use, ...)`
  - Lặp từng file → `extract_features_from_file` → stack thành ma trận.
  - Kiểm tra đồng bộ cột/shape; file không khớp thì skip.
- `filter_test_duplicates(train_paths, test_paths)`
  - Tính MD5, loại file test trùng nội dung train.
- `build_train_test_bundles(train_files, test_files, remove_test_duplicates)`

**Ý nghĩa**:
- Tạo X/y dùng train.
- Tránh rò rỉ dữ liệu do trùng file giữa train/test.

### Bước E — Outlier handling (được thực thi thật)
**File**: `src/preprocessing.py` + `src/main.py`
- Fit ngưỡng clip trên **train**:
  - `clip_bounds = fit_clip_bounds(X_bp_train, lo_q=0.01, hi_q=0.99)`
- Apply cho train/test:
  - `apply_clip_bounds(X, clip_bounds)`
- Lưu ra `saved_models/bp_clip_bounds.json` để inference dùng.

**Tại sao cần**:
- Bandpower thường heavy-tailed (đuôi dài), dễ có giá trị cực đoan do nhiễu (EMG, tiếp xúc điện cực kém).
- Clip theo quantile là cách “an toàn”, ít phá cấu trúc dữ liệu hơn z-score cắt cứng.

### Bước F — Train model
**File**: `src/model_phat.py`

#### Nhóm 1: Classical ML trên bandpower vector
- Chuẩn hoá: `RobustScaler` (chịu outlier tốt hơn StandardScaler).
- Mô hình:
  - Logistic Regression
  - KNN
  - MLP
  - Random Forest
  - XGBoost (nếu có)

#### Nhóm 2: Deep Learning trên spectrogram
- Small CNN
- EfficientNetB0 transfer learning

**Ý nghĩa chọn hướng**:
- Bandpower vector nhỏ gọn, dễ giải thích.
- Spectrogram giữ cấu trúc ảnh time–frequency, hợp CNN.

### Bước G — Đánh giá và lưu artifacts
**File**: `src/evaluation.py`
- `evaluate_binary(...)`: accuracy, AUC, report, confusion matrix.
- `export_models(...)`: lưu model `.joblib` / `.keras`, lưu `scaler_bp.joblib`.

**File**: `src/main.py`
- Ghi `saved_models/metrics.json`.
- Ghi `saved_models/eda.json`.

## 4) Web/API vận hành thế nào

### API (FastAPI)
**File**: `src/app.py`
- `/predict`:
  - Extract bandpower feature.
  - **Apply clip bounds** từ `bp_clip_bounds.json` (nếu có).
  - Apply scaler (`scaler_bp.joblib`) nếu có.
  - Predict bằng model tốt nhất trong `saved_models/`.
- `/analyze`:
  - Trả các chart data: time-series theo kênh, PSD, bandpower_by_band, spectrogram.
- `/eda`, `/metrics`, `/models`: cung cấp dữ liệu tổng quan.

### Web (React)
- `DuDoan.tsx` gọi song song `/predict` và `/analyze`.
- `GioiThieu.tsx` gọi `/eda` để vẽ tổng quan.
- `KienThucEEG.tsx` hiển thị kiến thức điện cực/dải tần (có nguồn).

## 5) Giải thích mô hình (để vấn đáp)

### Logistic Regression (LR)
- **Tinh thần**: mô hình tuyến tính trên vector feature.
- **Ưu**:
  - Dễ giải thích (hệ số coef).
  - Nhanh, ổn định, ít overfit khi feature không quá phức.
- **Nhược**:
  - Không học được quan hệ phi tuyến phức tạp giữa các dải/kênh.

### Random Forest (RF)
- **Tinh thần**: ensemble nhiều decision trees (bagging).
- **Ưu**:
  - Bắt phi tuyến tốt.
  - Ít cần scaling.
- **Nhược**:
  - Khó giải thích sâu.
  - Có thể kém nếu dữ liệu nhiễu mạnh hoặc feature nhiều chiều mà ít mẫu.

### KNN
- **Tinh thần**: dự đoán theo k láng giềng gần nhất.
- **Ưu**:
  - Đơn giản, baseline tốt.
- **Nhược**:
  - Nhạy scaling và curse-of-dimensionality.
  - Chậm khi dataset lớn.

### MLP (ANN)
- **Tinh thần**: mạng fully-connected học phi tuyến trên vector.
- **Ưu**:
  - Học phi tuyến tốt hơn LR.
- **Nhược**:
  - Nhạy hyperparameter.
  - Dễ overfit nếu không regularize.

### XGBoost
- **Tinh thần**: gradient boosting trees.
- **Ưu**:
  - Mạnh trên tabular feature.
  - Thường cho kết quả cao.
- **Nhược**:
  - Nhiều tham số, dễ overfit nếu tuning không cẩn thận.

### CNN trên Spectrogram
- **Tinh thần**: coi spectrogram như ảnh time–frequency.
- **Ưu**:
  - Học pattern cục bộ theo thời gian/tần số.
- **Nhược**:
  - Cần compute lớn hơn.
  - Nhạy nhiễu, cần preprocessing ổn.

### EfficientNet Transfer Learning
- **Tinh thần**: dùng backbone pretrained, fine-tune.
- **Ưu**:
  - Học feature mạnh.
- **Nhược**:
  - Pretrain trên ảnh tự nhiên, không “khớp” EEG 100%.
  - Dễ overfit nếu data nhỏ.

## 6) Kiến thức cần nói khi vấn đáp (gợi ý)
- **EEG**: tín hiệu điện đo trên da đầu; mỗi điện cực phản ánh vùng não (rất tổng quát).
- **Dải tần**:
  - Delta (0.5–4): ngủ sâu
  - Theta (4–8): buồn ngủ/thư giãn
  - Alpha (8–13): thư giãn tỉnh (nhắm mắt)
  - Beta (13–30): tỉnh táo/nhận thức
  - Gamma (30–45): xử lý nhanh, dễ nhiễu EMG
- **Nhiễu phổ biến**: EOG (chớp mắt), EMG (cơ mặt), tiếp xúc điện cực kém.
- **Vì sao dùng bandpower**: giảm chiều, dễ giải thích, mạnh cho tabular ML.
- **Vì sao cần chống leakage**: nếu file trùng train/test sẽ làm metric ảo.
- **Vì sao clip outlier + RobustScaler**: bandpower thường heavy-tailed.

## 7) Cách chạy (đúng thứ tự)
1) Train + sinh artifacts:
```bash
python -m src.main
```
2) Chạy API:
```bash
python -m uvicorn src.app:app --reload --port 8000
```
3) Chạy web:
```bash
cd web
npm install
npm run dev
```

---

Nếu bạn muốn mình “dọn râu ria” ngoài root (các script `.py` rời), bạn xác nhận giúp mình:
- Bạn muốn **xoá hẳn** hay **chuyển vào thư mục `tools/`**?
- Có script nào bắt buộc phải giữ tên vì đang nộp theo yêu cầu môn không?
