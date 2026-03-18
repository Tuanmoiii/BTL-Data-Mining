# Dự án: Phân tích và Dự báo Năng suất Cây trồng (Crop Yield Prediction)

## Mục lục
1. [Giới thiệu dự án]
2. [Thành viên nhóm]
3. [Cấu trúc repository]
4. [Dữ liệu]
5. [Quy trình thực hiện]
6. [Cài đặt và tái lập kết quả]

---

## 1. Giới thiệu dự án

**Đề tài:** Phân tích và Dự báo Năng suất Cây trồng (Đề số 7)

Dự án được thực hiện trong khuôn khổ học phần **Khai phá Dữ liệu - Học kỳ II năm học 2025-2026**.

### Mục tiêu:
- Phân tích các yếu tố ảnh hưởng đến năng suất cây trồng (thời tiết, đất đai, loại cây trồng)
- Khai phá luật kết hợp giữa các điều kiện canh tác và năng suất cao
- Phân cụm các vùng trồng trọt dựa trên đặc điểm khí hậu - thổ nhưỡng
- Xây dựng mô hình dự báo năng suất (hồi quy) và đánh giá hiệu năng
- Đề xuất các khuyến nghị canh tác dựa trên kết quả khai phá dữ liệu

### Tiêu chí thành công:
- Xây dựng được pipeline khai phá dữ liệu hoàn chỉnh, có module hóa rõ ràng
- Đạt được kết quả dự báo với MAE/RMSE thấp trên tập kiểm tra
- Phát hiện được các luật kết hợp có ý nghĩa thực tiễn (lift > 1.5, support khả quan)
- Phân cụm và diễn giải được đặc trưng của từng vùng trồng
- Đảm bảo tính tái lập (reproducible) theo đúng yêu cầu

---

## 2. Thành viên nhóm

| Họ và tên | MSSV | Vai trò |
|-----------|------|---------|
| Vũ Ngọc Tiến | 1771020668 | EDA, Preprocessing, Feature Engineering, Config, Params |
| Bùi Quang Tuấn | 1771020718 | Mining (Association, Clustering), Visualization, Reproducibility |
| Phong Ngọc Anh | 1771020056 | Modeling (Regression), Evaluation, README, Scripts |

---

## 3. Cấu trúc repository
```
BTL-Data-Mining/
│
├── .venv/                          # Môi trường ảo (không commit lên GitHub)
│   ├── etc/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   ├── .gitignore
│   ├── .lock
│   ├── CACHEDIR.TAG
│   └── pyvenv.cfg
│
├── config.yaml                      # File cấu hình tham số (seed, split, paths, hyperparams)
│
├── data/                            # Dữ liệu
│   ├── processed/                    # Dữ liệu sau tiền xử lý
│   │   ├── crop_clustered.csv
│   │   ├── crop_processed.csv
│   │   ├── model_results.csv
│   │   └── prediction_xgb.csv
│   │
│   └── raw/                          # Dữ liệu gốc (được .gitignore)
│       ├── pesticides.csv
│       ├── rainfall.csv
│       ├── temp.csv
│       ├── yield_df.csv
│       └── yield.csv
│
├── notebooks/                        # Các notebook báo cáo theo pipeline
│   ├── 01_EDA.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_evaluation_report.ipynb
│  
│
├── outputs/                          # Kết quả đầu ra và script chạy pipeline
│   ├── run_pipeline.py                  # Script chạy toàn bộ pipeline (1, U)
│   └── run_papermill.py                  # Script chạy notebooks bằng papermill
│
├── scripts/                           # Các script xử lý chính
│   ├── evaluate.py                      # Đánh giá mô hình (1, U)
│   ├── preprocess.py                     # Tiền xử lý dữ liệu (U)
│   └── train_model.py                     # Huấn luyện mô hình (6, U)
│
├── src/                                # Mã nguồn chính (module hóa)
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Đọc dữ liệu, kiểm tra schema
│   │   └── cleaner.py                  # Xử lý thiếu, outlier, encoding, scaling
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py                  # Feature engineering
│   │
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── association.py               # Apriori/FP-Growth tìm luật kết hợp
│   │   └── clustering.py                 # KMeans, HAC, profiling cụm
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── supervised.py                 # Linear/Ridge/XGBRegressor
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                    # MAE, RMSE, R2, silhouette
│   │   └── report.py                      # Tổng hợp bảng/biểu đồ kết quả
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                        # Hàm vẽ biểu đồ dùng chung
│
├── .gitignore                          # Các file/thư mục bỏ qua khi commit
├── LICENSE.txt                         # Giấy phép (Custom License - Educational Use)
├── README.md                           # Tài liệu hướng dẫn (tiếng Việt)
└── requirements.txt                    # Thư viện cần cài đặt
---
```
---

## 4. Dữ liệu

### Nguồn dữ liệu
**Kaggle:** [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

### Mô tả dữ liệu (Data Dictionary)

| File | Cột | Kiểu | Ý nghĩa |
|------|-----|------|---------|
| **yield.csv** | Area | object | Quốc gia / vùng trồng |
| | Item | object | Loại cây trồng |
| | Year | int | Năm thu hoạch |
| | hg/ha_yield | int | Năng suất (hectogam/ha) – **biến mục tiêu** |
| **rainfall.csv** | Area | object | Quốc gia |
| | Item | object | Loại cây trồng |
| | Year | int | Năm |
| | average_rain_fall_mm_per_year | float | Lượng mưa trung bình năm (mm) |
| **pesticides.csv** | Area | object | Quốc gia |
| | Item | object | Loại cây trồng |
| | Year | int | Năm |
| | pesticides_tonnes | float | Lượng thuốc trừ sâu (tấn) |
| **temp.csv** | Area | object | Quốc gia |
| | Item | object | Loại cây trồng |
| | Year | int | Năm |
| | avg_temp | float | Nhiệt độ trung bình năm (°C) |

### Rủi ro dữ liệu
- **Dữ liệu thiếu:** Một số vùng/năm có thể thiếu giá trị lượng mưa hoặc nhiệt độ
- **Đơn vị không đồng nhất:** Cần kiểm tra lại đơn vị của yield
- **Data Leakage:** Đảm bảo không dùng thông tin từ tương lai khi chia train/test

---

## 5. Quy trình thực hiện

### 5.1. Data Ingestion & EDA (`01_EDA.ipynb`)
- Đọc dữ liệu từ `data/raw/`, kiểm tra phân phối, tương quan, missing values
- Vẽ biểu đồ phân phối yield theo năm, vùng, loại cây

### 5.2. Tiền xử lý & Feature Engineering (`02_preprocess_feature.ipynb`)
- Gộp các bảng dữ liệu theo (Area, Item, Year)
- Xử lý missing values (median/mean imputation)
- Rời rạc hóa biến điều kiện cho luật kết hợp
- Feature engineering: lag features, tương tác nhiệt độ*mưa

### 5.3. Khai phá tri thức (`03_mining_or_clustering.ipynb`)
- **Luật kết hợp:** Apriori/FP-Growth tìm tổ hợp cho năng suất cao
- **Phân cụm:** KMeans/HAC trên vùng trồng, profiling từng cụm

### 5.4. Mô hình hóa (Hồi quy) (`04_Modeling.ipynb`)
- Baseline: Linear Regression, Ridge Regression
- Mô hình mạnh: Random Forest, XGBoost
- Đánh giá: MAE, RMSE, R²

### 5.5. Đánh giá & Insight (`05_evaluation_report.ipynb`)
- So sánh kết quả các mô hình
- Phân tích lỗi (residual analysis)
- Đề xuất khuyến nghị canh tác

---

## 6. Cài đặt và tái lập kết quả

### Yêu cầu hệ thống
- Python 3.9+
- pip / conda

### Các bước thực hiện

1. **Clone repository**
   ```bash
   git clone https://github.com/Tuanmoiii/BTL-Data-Mining.git
   cd BTL-Data-Mining
2.Tạo môi trường ảo

bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# hoặc
.venv\Scripts\activate      # Windows
3.Cài đặt thư viện

bash
pip install -r requirements.txt
4.Cấu hình dữ liệu

Tải dữ liệu từ Kaggle

Đặt các file .csv vào thư mục data/raw/

Kiểm tra và điều chỉnh đường dẫn trong config.yaml

5.Chạy toàn bộ pipeline

bash
python run_pipeline.py
6.Hoặc chạy từng bước

bash
python scripts/preprocess.py
python scripts/train_model.py
python scripts/evaluate.py

