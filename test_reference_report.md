# Tham khảo cách viết test trong repo `email-spam-classification`

### 1. Tổng quan

- Framework: **pytest** (>= 8.0.0), không dùng `unittest.TestCase`.
- Tổng cộng **55 test function** trải trên **13 file test** + 1 `conftest.py` (README ghi "43 tests" nhưng đó là số cũ, thực tế trên `main` hiện tại là 55 — không có gì sai, chỉ là README chưa cập nhật).
- Cách chạy:

  ```bash
  pytest -q
  ```

### 2. Cấu trúc thư mục `test/`

```
test/
├── conftest.py              # fixtures dùng chung, rất ít (chỉ 2 fixture)
├── test_api.py              # app/api.py
├── test_bronze_ingest.py    # src/etl/bronze_ingest.py
├── test_data_quality_check.py
├── test_drift_detector.py
├── test_evaluate.py
├── test_gmail_client.py
├── test_gmail_poller.py
├── test_gold_build.py
├── test_predict.py
├── test_silver_transform.py
├── test_split_raw.py
├── test_text_preprocessing.py
└── test_train.py
```

**Quy tắc 1-1:** mỗi file `src/...` hoặc `app/...` có đúng một file `test_<tên_module>.py` tương ứng. Không gộp nhiều module vào 1 file test, không tách 1 module ra nhiều file test.

### 3. Các pattern chính quan sát được

#### 3.1. Mỗi test có một dòng comment "Tiêu chí" ngay phía trên

Không dùng docstring bên trong hàm, mà đặt 1 dòng comment `#` mô tả rõ **tiêu chí chấp nhận** bằng tiếng Việt ngay trước `def test_...`:

```python
# Tiêu chí: API health phản hồi trạng thái degraded rõ ràng khi mô hình chưa sẵn sàng.
def test_health_returns_degraded_when_model_missing(monkeypatch):
    ...
```

→ Đọc lướt file test là hiểu ngay spec của module, không cần đọc code.

#### 3.2. Tên hàm test mô tả hành vi, không mô tả implementation

Dạng `test_<subject>_<expected_behavior>`, ví dụ:
`test_pick_threshold_falls_back_when_target_precision_is_not_hit`,
`test_ingest_is_idempotent_when_partition_exists`.

#### 3.3. Cô lập I/O bằng `monkeypatch`, không dùng mock framework

Thay vì `unittest.mock.patch`, họ ghi đè trực tiếp các **module-level constant** (đường dẫn thư mục, hằng số cấu hình) hoặc **hàm** bằng `monkeypatch.setattr`:

```python
monkeypatch.setattr(bronze_ingest, "RAW_DIR", raw_dir)
monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)
```

#### 3.4. `pytest.importorskip` cho dependency nặng/optional

```python
pytest.importorskip("fastapi")
pytest.importorskip("googleapiclient.discovery")
pytest.importorskip("xgboost")
```

→ Nếu môi trường CI/local thiếu package nặng (xgboost, google api client...), test đó **skip** thay vì fail cứng cả suite. Đặt ngay đầu file, trước `import` module đích.

#### 3.5. `pytest.approx` cho mọi so sánh số thực

Không bao giờ so sánh float bằng `==` trực tiếp khi có sai số làm tròn (PSI, precision/recall, threshold...).

#### 3.6. Dữ liệu test là DataFrame nhỏ dựng tay, không load file ngoài

Test dữ liệu (2–10 dòng) được tạo trực tiếp trong test hoặc trong `conftest.py`/helper function nội bộ file (`_gold_ready_df()` trong `test_gold_build.py`). Không có thư mục `test/fixtures/*.csv`.

#### 3.7. `conftest.py` cực kỳ mỏng

Chỉ 2 fixture dùng chung nhiều file (`bronze_like_df`, `long_body`). Phần lớn dữ liệu test vẫn để **cục bộ trong từng file** thay vì dồn hết vào conftest — tránh fixture dùng chung bị phình to và khó truy vết.

#### 3.8. Test bám theo từng bước của pipeline (Medallion) + biên hệ thống

Bao phủ toàn bộ luồng: `split_raw → bronze_ingest → silver_transform → gold_build → train → evaluate → predict → api`, cộng thêm các nhánh phụ (`drift_detector`, `gmail_client`, `gmail_poller`). Mỗi bước test cả **happy path** và **lỗi/edge case** (file thiếu, schema lạ, dữ liệu rỗng, NaN, đã tồn tại/idempotent).

#### 3.9. Test hàm thuần (pure function) không mock ML

Với logic toán (`population_stability_index`, `pick_threshold`, `evaluate`), họ không mock sklearn — dùng mảng numpy nhỏ, tính tay kết quả kỳ vọng, assert bằng công thức cụ thể. Chỉ mock/monkeypatch ở biên I/O hoặc network (API call, Gmail, model loading).

### 4. Dependency phục vụ test (từ `requirements.txt`)

```
pytest>=8.0.0
httpx>=0.27.0      # cần cho FastAPI TestClient (dù thực tế test hiện gọi thẳng hàm api.*, không dùng TestClient)
black / isort / flake8   # lint, không phải test nhưng đi kèm nhóm "Testing and code quality"
```

---

# Áp dụng vào `Recommendation-System-Taobao` 

## Các test áp dụng cho `Recommendation-System-Taobao`

Dự án hiện tại (`medallion/bronze|silver|gold`, `run_pipeline.py`) có cấu trúc ETL tương tự repo mẫu nhưng **chưa có test nào**. Đề xuất áp dụng theo đúng pattern trên:

1. Tạo `test/` ở root, 1 file test / 1 module trong `medallion/bronze`, `medallion/silver`, `medallion/gold`, `medallion/common`.
2. Thêm `conftest.py` mỏng, chỉ 1–2 fixture DataFrame mẫu dùng chung.
3. Trong code nguồn `medallion/*`, đưa các đường dẫn output (bronze/silver/gold dir) thành biến module cấp cao để có thể `monkeypatch` trong test — hiện cần kiểm tra `run_pipeline.py` xem đã theo dạng này chưa.
4. Test từng bước pipeline độc lập với `tmp_path`, không đụng `medallion/bronze|silver|gold` thật.
5. Thêm `pytest>=8.0.0` vào `requirements.txt`, chạy bằng `pytest -q`.


#### 1.1. Bronze / Silver / Gold là gì (kiến trúc Medallion)

Dự án dùng kiến trúc pipeline 3 tầng, mỗi tầng "sạch" hơn tầng trước:

| Tầng            | Ý nghĩa                                                                                                                     | Trong`medallion/`                                                                                                                                                                                                                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bronze** | Dữ liệu thô, giữ y nguyên, chỉ sửa lỗi schema hiển nhiên (tên cột dư khoảng trắng)                             | `bronze/ingest.py`: đọc `train.csv`/`test.csv` (log click) + `ad_feature.csv.zip`/`user_profile.csv.zip` → ghi parquet, events chia theo ngày (`dt=YYYY-MM-DD/`)                                                                                                                         |
| **Silver** | Dữ liệu đã làm sạch: bỏ trùng, ép kiểu, fillna giá trị thiếu thật sự —**chưa** có feature engineering | `silver/clean.py`: dedup theo (user, ad, thời điểm), assert `click` chỉ có 0/1, điền `"unknown"` cho brand/pvalue_level/new_user_class_level thiếu, assert `price > 0`                                                                                                                   |
| **Gold**   | Bảng feature sẵn sàng train model                                                                                          | `gold/features.py`: target encoding, CTR lịch sử, đặc trưng co-click, đặc trưng affinity user — theo nguyên tắc bắt buộc **point-in-time correctness**: feature của một dòng chỉ được dùng dữ liệu xảy ra *trước* dòng đó, không được nhìn thấy tương lai |

`common/config.py` giữ toàn bộ đường dẫn (`RAW_DIR`, `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`) và hằng số (`RANDOM_SEED`, `SMOOTHING`, `TARGET`, `ID_COLS`) dưới dạng biến module cấp cao — đúng điều kiện cần để `monkeypatch` được trong test (mục 3.3).

#### 1.2. Test suite đã viết — `tests/` (folder riêng, tách khỏi `medallion/`)

Vị trí `tests/` so với code nguồn — nằm ngang hàng ở root, không lồng vào trong `medallion/`:

```
Recommendation-System-Taobao/
├── medallion/               
│   ├── common/config.py
│   ├── bronze/ingest.py
│   ├── silver/clean.py
│   ├── gold/features.py
│   └── run_pipeline.py
├── tests/                     # test — folder riêng, tự đứng độc lập
│   ├── conftest.py            # 1 fixture dùng chung: ingested_at
│   ├── test_bronze_ingest.py  # 3 test  → medallion/bronze/ingest.py
│   ├── test_silver_clean.py   # 6 test  → medallion/silver/clean.py
│   └── test_gold_features.py  # 7 test  → medallion/gold/features.py
├── requirements.txt            # + pytest>=8.0.0, pyarrow>=15.0.0
└── test_reference_report.md    # báo cáo này
```

**Tổng: 16 test, tất cả pass.**

| File                      | Test                                                                       | Kiểm tra gì                                                                                        |
| ------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `test_bronze_ingest.py` | `test_ingest_events_partitions_by_day_and_tags_source`                   | events được chia đúng thư mục theo ngày, gắn đúng`_split`/`_source_file`              |
|                           |  `test_ingest_dim_strips_column_whitespace_and_tags_source`              | tên cột dư khoảng trắng được strip                                                           |
|                           | `test_ingest_writes_events_and_dim_parquets_end_to_end`                  | `ingest()` chạy đầu-cuối ra đủ 3 loại output                                                |
| `test_silver_clean.py`  | `test_fillna_unknown_maps_nan_to_unknown_string`                         | hàm thuần map NaN →`"unknown"`                                                                  |
|                           | `test_clean_events_dedups_and_keeps_fact_columns`                        | dedup đúng theo (userid, adgroup_id, time_stamp)                                                   |
|                           | `test_clean_events_rejects_non_binary_click`                             | `click` ngoài {0,1} bị chặn bằng `AssertionError`                                            |
|                           | `test_clean_dim_ad_fills_unknown_brand_and_dedups`                       | brand thiếu →`"unknown"`, dedup giữ bản mới nhất                                             |
|                           | `test_clean_dim_ad_rejects_non_positive_price`                           | `price ≤ 0` bị chặn                                                                             |
|                           | `test_clean_dim_user_fills_unknown_for_sentinel_columns`                 | pvalue_level/new_user_class_level thiếu →`"unknown"`                                             |
| `test_gold_features.py` | `test_kfold_target_encode_is_deterministic_for_fixed_seed`               | cùng seed → cùng kết quả                                                                        |
|                           | `test_kfold_target_encode_falls_back_to_global_mean_for_unseen_category` | category lạ ở test set → fallback global_mean                                                     |
|                           | `test_add_target_encodings_drops_raw_id_columns_and_keeps_keys`          | cột id thô bị drop, chỉ còn`*_key`/`*_te`                                                   |
|                           | `test_add_domain_features_excludes_current_row_from_before_counts`       | **quan trọng nhất**: `ag_clicks_before` không tính chính dòng hiện tại (chống leak) |
|                           | `test_add_item_coclick_affinity_ignores_pure_scoring_lookups`            | bộ đếm co-click không bị "ô nhiễm" bởi lượt xem không click                               |
|                           | `test_add_user_affinity_features_applies_smoothing_formula`              | công thức smoothing đúng theo thiết kế                                                         |
|                           | `test_add_interactions_computes_freq_from_train_only`                    | tần suất tương tác chỉ học từ train, không leak từ test                                    |

#### 1.3. Cách chạy

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

`requirements.txt` được bổ sung `pyarrow>=15.0.0` (pipeline cần để đọc/ghi parquet — vốn thiếu sẵn từ trước, không phải do lần này) và `pytest>=8.0.0`.

---
