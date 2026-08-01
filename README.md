# Recommendation-System-Taobao

## 1. Mô tả bài toán

**CTR (Click-Through Rate) Prediction**: ước lượng xác suất một người dùng sẽ click vào một
quảng cáo, cho một bộ ba `(user, ad, context)`. Xác suất này ảnh hưởng trực tiếp đến việc xếp
hạng/lựa chọn quảng cáo để hiển thị và doanh thu quảng cáo.

Bài toán khó vì:

- Dữ liệu quy mô lớn, high-dimensional, nhiều cột categorical có cardinality cao (`userid`,
  `adgroup_id`, `campaign_id`, `customer`, `brand`, ...).
- Quan hệ giữa feature và click phần lớn phi tuyến, khó bắt được bằng model tuyến tính đơn thuần
  (Logistic Regression) — đã verify trực tiếp trong EDA: correlation/Cliff's delta của từng
  feature riêng lẻ hầu hết ở mức "negligible", nghĩa là tín hiệu chỉ thực sự lộ ra khi các model
  tree-based tổ hợp nhiều feature yếu lại với nhau.
- Bài toán mất cân bằng lớp (class imbalance) — overall CTR trong tập dữ liệu này là ~20%, tức
  80% là lớp "không click".
- Cá nhân hóa (personalization) là có thật nhưng khá mỏng: phần lớn user chỉ xuất hiện đúng 1 lần
  trong log (xem mục 5 — EDA).

## 2. Lựa chọn metric

Có **2 tầng** đánh giá khác nhau trong bài toán này (giải thích đầy đủ, có ví dụ, nằm trong
`experiment/Full_Training_Model.ipynb`, phần "Recall@K thuộc tầng nào trong bài toán CTR?"):

| Tầng | Câu hỏi | Metric dùng |
|---|---|---|
| **CTR prediction (pointwise)** | Model ước lượng `P(click \| user, ad, context)` cho từng impression riêng lẻ có chính xác không? | **AUC** (threshold-independent, đo khả năng rank 1 click lên trên 1 non-click) |
| **Recommendation / retrieval (listwise)** | Nếu chỉ được hiển thị Top K trong cả catalog, Top K đó có chứa đúng sản phẩm user thật sự muốn click không? | **Recall@10 / @20 / @50 / @100** |

Vì sao cần cả hai: AUC tốt không đảm bảo Top K tốt — hai model AUC gần bằng nhau vẫn có thể cho
chất lượng Top K rất khác nhau (verify được trong notebook: model AUC cao nhất — RandomForest —
không phải lúc nào cũng có Recall@K cao nhất so với các model khác). Ngoài ra còn có:

- **Baseline so sánh**: Random (sàn lý thuyết, Recall@K ≈ K/300) và Popularity (không cá nhân hóa,
  luôn rank theo `ag_ctr_before` — CTR lịch sử của chính sản phẩm) — để biết model thực sự học
  được gì ngoài "cứ show sản phẩm hot nhất."
- **CTR-lift/gains chart**: đo CTR thực tế đạt được nếu chỉ hành động trên top X% impression có
  điểm dự đoán cao nhất, so với baseline 19.26% — trả lời câu hỏi kinh doanh "CTR cải thiện được
  bao nhiêu" (không có một con số duy nhất, mà là một đường cong phụ thuộc mức độ chọn lọc).

## 3. Mô tả dữ liệu — Sampling

Dữ liệu gốc là bộ **Taobao Display Ad Click** (Alibaba/Tianchi) — log hiển thị/click quảng cáo
gồm khoảng **26 triệu dòng** (`raw_sample.csv` gốc), theo dõi hoạt động của **1,061,768 users**
trong nhiều ngày.

**Hạn chế**: 26 triệu dòng vượt quá khả năng xử lý của tài nguyên hiện tại (máy cá nhân, không có
cluster/Spark), nên đã **downsample xuống còn 300,000 dòng** (`input_data/sampled_dataset.csv`),
sau đó chia theo thời gian (time-based split) thành:

| Tập | Số dòng | Khoảng thời gian |
|---|---|---|
| `train.csv` | 240,000 | Toàn bộ nằm trước `test.csv` |
| `test.csv` | 60,000 | Sau `train.csv` |
| **Tổng** | **300,000** | 2017-05-05 → 2017-05-13 (9 ngày) |

**Lưu ý quan trọng**: chỉ **log impression** (click log) bị downsample — hai bảng dimension
(`ad_feature`, `user_profile`) vẫn giữ **nguyên vẹn, đầy đủ** như dataset gốc (846,811 ads,
1,061,768 users), dùng làm catalog/hồ sơ tham chiếu đầy đủ. Đây chính là lý do EDA
(`experiment/EDA.ipynb`, mục "Customer overview") đo được: chỉ **16.41%** user trong catalog đầy
đủ thực sự "active" (tức có xuất hiện trong log 300k dòng đã sample) — không phải vì 83.59% user
kia không tồn tại, mà vì log tương tác của họ không rơi vào phần được sample.

## 4. Mô tả dữ liệu — Các bảng

| File | Vai trò | Số dòng | Nội dung chính |
|---|---|---|---|
| `input_data/train.csv`, `test.csv` (gộp lại = `sampled_dataset.csv`) | **Impression log** — mỗi dòng là 1 lần hiển thị quảng cáo | 300,000 | `userid`, `adgroup_id`, `time_stamp`, `pid` (vị trí hiển thị), `cate_id`, `campaign_id`, `customer`, `brand`, `price`, các cột nhân khẩu học của user tại thời điểm đó, `click` (nhãn 0/1), `date`, `hour`, `weekday` |
| `input_data/ad_feature.csv.zip` | **Catalog quảng cáo đầy đủ** — thuộc tính tĩnh của từng ad (chỉ **300** trong số 846,811 ad này thực sự xuất hiện trong log đã sample ở trên) | 846,811 | `adgroup_id`, `cate_id`, `campaign_id`, `customer`, `brand`, `price` |
| `input_data/user_profile.csv.zip` | **Hồ sơ người dùng đầy đủ** — thuộc tính nhân khẩu học tĩnh | 1,061,768 | `userid`, `cms_segid`, `cms_group_id`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level` |

## 5. Phần phân tích nằm trong EDA

Toàn bộ phân tích khám phá dữ liệu nằm ở **`experiment/EDA.ipynb`**, đi từ bức tranh lớn xuống
nhỏ (big picture → small):

| # | Mục | Nội dung |
|---|---|---|
| 1 | Overview | Tổng số dòng, khoảng thời gian, CTR tổng |
| 2 | By day | Volume & CTR biến động theo ngày |
| 3 | By category | Volume & CTR theo `cate_id` |
| 4 | By brand | Volume & CTR theo `brand` |
| 5 | Correlations | Numeric feature vs. `click` (Pearson, Mutual Information, Cliff's delta), ranking tổng hợp có trọng số, phân tích theo nhóm (User behavior / Domain / Time-series / User-based & Item-based) |
| 6 | Customer overview | Bao nhiêu user active, mức độ tập trung engagement |
| 7 | Customer segments | CTR theo giới tính/tuổi/shopping level/... |
| 8 | Engagement distribution | Phân phối impression/user, đường Lorenz |
| 9 | Ad performance features | Phân phối `ag_ctr_before`, ad nào đang "nóng lên"/"nguội đi", top ad theo volume vs. theo revenue |

Chi tiết model (training, so sánh AUC/Recall@K, baseline, CTR-lift, feature importance,
recommendation JSON) nằm ở **`experiment/Full_Training_Model.ipynb`**, không lặp lại ở EDA.

## 6. Cấu trúc Repo

```text
recommendation_system/
├── input_data/                    # Dữ liệu thô (mục 3, 4) — gitignored
│   ├── train.csv / test.csv       # Impression log đã sample + split theo thời gian
│   ├── sampled_dataset.csv        # train + test gộp lại, trước khi split
│   ├── ad_feature.csv.zip         # Catalog quảng cáo đầy đủ
│   └── user_profile.csv.zip       # Hồ sơ người dùng đầy đủ
│
├── medallion/                     # Pipeline Bronze → Silver → Gold, dạng module Python
│   ├── bronze/ingest.py           #   Nạp thô, partition theo ngày, không transform
│   ├── silver/clean.py            #   Dedupe, chuẩn hoá cột, xử lý null thật (lightweight)
│   ├── gold/features.py           #   Feature engineering leak-safe (target encoding, domain,
│   │                               #   user-based & item-based CF) → training_features.parquet
│   ├── gold/snapshots.py          #   Snapshot phục vụ scoring (không chạy trong pipeline chính)
│   ├── common/config.py           #   Đường dẫn & hằng số dùng chung
│   ├── run_pipeline.py            #   Orchestrator: Bronze → Silver → Gold
│   └── README.md                  #   
│                                   
│
├── data/                          # Output của medallion pipeline — gitignored, tự sinh lại được
│   ├── bronze/  silver/  gold/    #   (partition theo ngày ở bronze/silver)
│
├── experiment/                    # Notebook — nơi thực sự train & phân tích
│   ├── Baseline_Model.ipynb       #   Model baseline gốc (trước khi có medallion)
│   ├── EDA.ipynb                  #   Toàn bộ phân tích ở mục 5
│   ├── Full_Training_Model.ipynb  #   Feature engineering + 5 model + AUC/Recall@K + baseline +
│   │                               #   CTR-lift + feature importance + recommendation JSON
│   └── models/                    #   Model đã train, lưu bởi Full_Training_Model.ipynb
│
├── output_data/                   # Kết quả xuất ra để dùng ngoài (recommendation JSON, ...)
├── models/, airflow/, app/, logs/ # Thư mục scaffold cho phần mở rộng sau này (hiện đang rỗng —
│                                   # ví dụ: đóng gói thành service, orchestrate bằng Airflow,
│                                   # theo dõi log production), chưa có nội dung
├── requirements.txt               # Version cố định của toàn bộ dependency
├── Explain_metric.xlsx            # Giải thích metric bổ sung (dạng bảng tính)
└── README.md                      # File này
```

**Dựa theo cấu trúc trên, luồng hoạt động của repo là:**

1. `input_data/` (mục 3–4) → chạy `python medallion/run_pipeline.py` → sinh ra `data/bronze`,
   `data/silver`, `data/gold/training_features.parquet`.
2. `experiment/EDA.ipynb` đọc từ **Silver** (`fact_events`/`dim_ad`/`dim_user`) và **Gold**
   (`training_features.parquet`) để phân tích (mục 5).
3. `experiment/Full_Training_Model.ipynb` đọc **trực tiếp từ `input_data/`** (không qua
   medallion) và tự tính lại toàn bộ feature — đây là **trùng lặp có chủ đích**: notebook phục vụ
   train/eval tương tác, còn `medallion/gold/features.py` phục vụ chạy tự động, không phụ thuộc
   lẫn nhau khi thực thi (chi tiết & rủi ro của việc trùng lặp này nằm ở `medallion/README.md`).
4. Model tốt nhất + recommendation JSON được lưu vào `experiment/models/` và `output_data/` —
   sẵn sàng để một service (thư mục `app/`, hiện còn rỗng) hoặc job lịch (`airflow/`, hiện còn
   rỗng) đọc vào dùng trong tương lai.
