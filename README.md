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

Đề tài: **Phân tích và Dự báo Năng suất Cây trồng** (Đề số 7)

Dự án được thực hiện trong khuôn khổ học phần **Khai phá Dữ liệu - Học kỳ II năm học 2025-2026**.

### Mục tiêu:
- Phân tích các yếu tố ảnh hưởng đến năng suất cây trồng (thời tiết, đất đai, loại cây trồng).
- Khai phá luật kết hợp giữa các điều kiện canh tác và năng suất cao.
- Phân cụm các vùng trồng trọt dựa trên đặc điểm khí hậu - thổ nhưỡng.
- Xây dựng mô hình dự báo năng suất (hồi quy) và đánh giá hiệu năng.
- Đề xuất các khuyến nghị canh tác dựa trên kết quả khai phá dữ liệu.

### Tiêu chí thành công:
- Xây dựng được pipeline khai phá dữ liệu hoàn chỉnh, có module hóa rõ ràng.
- Đạt được kết quả dự báo với MAE/RMSE thấp trên tập kiểm tra.
- Phát hiện được các luật kết hợp có ý nghĩa thực tiễn (lift > 1.5, support khả quan).
- Phân cụm và diễn giải được đặc trưng của từng vùng trồng.
- Repo đảm bảo tính tái lập (reproducible) theo đúng yêu cầu giảng viên.

---

## 2. Thành viên nhóm

| Họ và tên | MSSV | Vai trò |
|-----------|------|---------|
| [Vũ Ngọc Tiến] | [1771020668] | EDA, Preprocessing, Feature Engineering, Config, Params |
| [Bùi Quang Tuấn] | [1771020718] | Mining (Association, Clustering), Visualization , Reproducibility |
| [Phong Ngọc Anh] | [177102056] | Modeling (Regression), Evaluation, README, Scripts |
---

## 3. Cấu trúc repository
BTL-Data-Mining/
│
├── config.yaml                # Cấu hình chung (seed, paths, hyperparameters)
│
├── data/                      # Dữ liệu
│   ├── raw/                   # Dữ liệu gốc (KHÔNG commit)
│   └── processed/             # Dữ liệu sau tiền xử lý
│
├── notebooks/                 # Notebook theo pipeline phân tích
│   ├── 01_EDA.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_Modeling.ipynb
│   └── 05_evaluation_report.ipynb
│
├── src/                       # Source code (module hóa)
│   ├── data/                 # Load & clean dữ liệu
│   ├── features/             # Feature engineering
│   ├── mining/               # Association rules, clustering
│   ├── models/               # Mô hình ML
│   ├── evaluation/           # Metrics & báo cáo
│   └── visualization/        # Vẽ biểu đồ
│
├── scripts/                   # Script chạy từng bước
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
│
├── outputs/                   # Kết quả đầu ra
│   ├── figures/              # Biểu đồ
│   ├── tables/               # Bảng kết quả
│   ├── models/               # Model đã train
│   └── reports/              # Báo cáo/logs
│
├── run_pipeline.py            # Chạy toàn bộ pipeline
├── run_papermill.py           # Chạy notebook tự động
│
├── requirements.txt           # Thư viện cần cài
├── .gitignore                 # File bỏ qua khi commit
├── LICENSE.txt                # Giấy phép
└── README.md                  # Tài liệu dự án
---

## 4. Dữ liệu

### Nguồn dữ liệu:
Kaggle: [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)  
*(Hoặc link cụ thể nhóm sử dụng)*

### Mô tả dữ liệu (Data Dictionary):

| Tên file | Cột | Kiểu | Ý nghĩa |
|----------|-----|------|---------|
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
| | pesticides_tonnes | float | Lượng thuốc trừ sâu sử dụng (tấn) |
| **temp.csv** | Area | object | Quốc gia |
| | Item | object | Loại cây trồng |
| | Year | int | Năm |
| | avg_temp | float | Nhiệt độ trung bình năm (°C) |

### Rủi ro dữ liệu:
- **Dữ liệu thiếu:** Một số vùng/năm có thể thiếu giá trị lượng mưa hoặc nhiệt độ.
- **Đơn vị không đồng nhất:** Cần kiểm tra lại đơn vị của yield (hg/ha → tấn/ha nếu cần).
- **Data Leakage:** Cần đảm bảo không dùng thông tin từ tương lai (khi chia train/test theo năm).

---

## 5. Quy trình thực hiện

Dự án tuân theo pipeline khai phá dữ liệu chuẩn:

1. **Data Ingestion & EDA** (`01_EDA.ipynb`):
   - Đọc dữ liệu từ `data/raw/`, kiểm tra phân phối, tương quan, missing values.
   - Vẽ biểu đồ: phân phối yield theo năm, theo vùng, theo loại cây.

2. **Tiền xử lý & Feature Engineering** (`02_preprocess_feature.ipynb` + `src/data/`, `src/features/`):
   - Gộp các bảng dữ liệu (merge) theo (Area, Item, Year).
   - Xử lý missing (median/mean imputation, hoặc loại bỏ nếu quá nhiều).
   - Rời rạc hóa (discretization) các biến điều kiện để chuẩn bị cho luật kết hợp.
   - Feature engineering: lag features (năng suất năm trước), tương tác (nhiệt độ*mưa), ...

3. **Khai phá tri thức** (`03_mining_or_clustering.ipynb` + `src/mining/`):
   - **Luật kết hợp:** Apriori/FP-Growth trên dữ liệu đã rời rạc để tìm tổ hợp (giống cây, vùng, điều kiện thời tiết) cho năng suất cao. Đánh giá support, confidence, lift.
   - **Phân cụm:** KMeans/HAC trên các vùng trồng dựa trên đặc trưng khí hậu. Profiling từng cụm, so sánh năng suất trung bình giữa các cụm.

4. **Mô hình hóa (Hồi quy)** (`04_Modeling.ipynb` + `src/models/`):
   - Baseline: Linear Regression, Ridge Regression.
   - Mô hình mạnh: Random Forest Regressor, XGBoost Regressor.
   - Thiết lập thực nghiệm: chia train/test theo năm (tránh leakage), sử dụng cross-validation.
   - Đánh giá: MAE, RMSE, R².

5. **Đánh giá & Insight** (`05_evaluation_report.ipynb` + `src/evaluation/`):
   - So sánh kết quả các mô hình (bảng, biểu đồ).
   - Phân tích lỗi (residual analysis): mô hình sai nhiều ở vùng nào, loại cây nào?
   - Đưa ra các khuyến nghị canh tác dựa trên luật kết hợp và phân cụm.

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
### Tạo môi trường ảo (khuyến khích)

2. python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# hoặc
.venv\Scripts\activate      # Windows
### Cài đặt thư viện
3. pip install -r requirements.txt
4. Cấu hình dữ liệu

Tải dữ liệu từ Kaggle theo link đã cung cấp.

Đặt các file .csv vào thư mục data/raw/.

Kiểm tra và điều chỉnh đường dẫn trong configs/params.yaml nếu cần.
