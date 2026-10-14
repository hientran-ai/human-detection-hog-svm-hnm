# Human Detection with HOG, SVM and Hard-Negative Mining

Project phát hiện người đi bộ theo pipeline cổ điển:

1. Chuẩn hóa mẫu người/nền từ INRIA Person.
2. Trích xuất đặc trưng HOG 3.780 chiều.
3. Huấn luyện Linear SVM baseline.
4. Quét ảnh nền để thu thập hard negatives và huấn luyện model cuối.
5. Phát hiện đa tỉ lệ, sau đó lọc box bằng IoU Non-Maximum Suppression.

Repository đã kèm model HNM và scaler để chạy inference ngay. Dataset INRIA không được đưa vào Git.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Tại thư mục gốc project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra CLI:

```powershell
python -m src --help
```

## Chạy phát hiện

```powershell
python -m src detect path\to\image.jpg --output-dir outputs
```

Kết quả gồm `result_raw_boxes.jpg` và `result_final_nms.jpg`. Các tham số thường cần tinh chỉnh:

```powershell
python -m src detect image.jpg `
  --win-step 16 `
  --scale-factor 1.15 `
  --hit-threshold 0.5 `
  --nms-overlap 0.3
```

Tăng `hit-threshold` để giảm báo động giả. Giảm `win-step` để quét kỹ hơn nhưng chạy chậm hơn.

## Chuẩn bị INRIA Person

Giải nén dataset thành cấu trúc sau:

```text
inriadata/
├── Train/
│   ├── annotations/
│   ├── neg/
│   ├── pos/
│   ├── neg.lst
│   └── pos.lst
└── Test/
    ├── annotations/
    ├── neg/
    ├── pos/
    ├── neg.lst
    └── pos.lst
```

Đường dẫn dataset/model được neo theo thư mục project, không phụ thuộc working directory. Các lệnh mẫu bên dưới được chạy tại thư mục gốc để Python tìm thấy package `src`.

## Huấn luyện đầy đủ

```powershell
# Tạo các crop 64x128; --clean xóa crop cũ trước khi tạo lại.
python -m src build-dataset --clean --seed 42

# Tạo models/base_svm_model.pkl và models/base_scaler.pkl.
python -m src train

# Tạo models/hnm_svm_model.pkl và models/scaler.pkl.
python -m src mine

# Đánh giá detection-level trên tập Test.
python -m src evaluate --iou-threshold 0.5
```

Không dùng accuracy trên tập train làm kết quả cuối. Lệnh `evaluate` báo cáo precision, recall, F1, FPS và lưu tối đa 10 ảnh lỗi vào `inriadata/error_analysis/`.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Test hiện có bảo vệ logic sliding window, IoU và NMS. Để kiểm tra nhanh toàn bộ source:

```powershell
python -m compileall -q src tests
```

## Cấu trúc chính

```text
src/
├── cli.py                       # CLI thống nhất
├── config.py                    # Đường dẫn và cấu hình
├── data/dataset_builder.py      # Tạo mẫu train
├── features/hog_extractor.py    # HOG descriptor
├── models/train_baseline.py     # Baseline Linear SVM
├── models/hard_negative_mining.py
├── models/detector.py           # Multi-scale detector
├── inference.py
├── evaluate.py
└── utils/
    ├── inria_parser.py
    ├── nms.py
    └── windowing.py
```

## Lưu ý về model

- Baseline dùng riêng `base_scaler.pkl`.
- Model sau HNM dùng `scaler.pkl`.
- Không tráo scaler giữa hai model; feature scaling được fit lại sau khi thêm hard negatives.
- Các file model được tạo bởi `joblib`/scikit-learn chỉ nên được nạp từ nguồn tin cậy.
