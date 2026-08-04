# Recommendation-System-Taobao

CTR (Click-Through Rate) prediction cho quảng cáo Taobao: `(user, ad, context)` →
`P(click)`. Pipeline dữ liệu Medallion (Bronze → Silver → Gold, chạy tự động qua
Airflow) nuôi một notebook huấn luyện 5 model ứng viên, chọn winner theo AUC, đánh
giá bằng cả AUC lẫn Recall@K, và xuất recommendation JSON.

> **Trạng thái hiện tại**: Bronze/Silver/Gold + Airflow đã chạy được end-to-end.
> Huấn luyện model vẫn là notebook chạy tay, **chưa** nằm trong Airflow DAG. **Chưa
> có** service serving (`app/` rỗng) — xem mục 9 "Chưa có / Roadmap".

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
  trong log (xem mục 6 — EDA).

## 2. Kiến trúc hệ thống

### 2.1 Data pipeline — đã có code

```text
input_data/*.csv(.zip)
        │
        ▼  medallion.bronze.ingest
            (gộp train+test, gắn _split/_ingested_at, strip tên cột dimension)
data/bronze/events/dt=YYYY-MM-DD/part.parquet
data/bronze/{ad_feature,user_profile}.parquet
        │
        ▼  medallion.silver.clean
            (dedupe, assert click∈{0,1} & price>0, null thật → "unknown")
data/silver/fact_events/dt=YYYY-MM-DD/part.parquet
data/silver/{dim_ad,dim_user}.parquet
        │
        ▼  medallion.gold.features
            (target encoding K-fold, domain/historical CTR, item co-click,
             user affinity, interaction freq — mọi feature "before" chỉ
             dùng dữ liệu trước timestamp hiện tại)
data/gold/training_features.parquet   (300.000 dòng / 46 cột)
data/gold/item_coclick_snapshot.pkl
        │
        ▼  experiment/Full_Training_Model.ipynb   (chạy tay, đọc thẳng input_data/,
        │   không qua medallion — xem mục 8 vì sao đây là trùng lặp có chủ đích)
        ▼
experiment/models/full_training_model_v5.joblib   (champion — RandomForest, AUC 0.607)
output_data/recommendations_top100.json
```

`gold/snapshots.py` (đóng băng snapshot sản phẩm/user phục vụ scoring cho 1
`userid` bất kỳ mà không cần chạy lại pipeline) đã có code nhưng **chưa** được
gọi trong `run_pipeline.py`.

### 2.2 Orchestration — Airflow (đã có code cho Bronze → Silver → Gold)

```text
┌─────────────────── Airflow (Docker, LocalExecutor) ───────────────────┐
│  DAG: medallion_pipeline   (schedule=None, trigger tay)                │
│                                                                          │
│   bronze_ingest  >>  silver_clean  >>  gold_features                    │
│   (mỗi task = PythonOperator gọi thẳng hàm trong medallion/,            │
│    do_xcom_push=False vì các hàm trả về pathlib.Path, không phải        │
│    kiểu JSON-serializable mà XCom yêu cầu)                              │
└──────────────────────────────────────────────────────────────────────┘
```

Repo mount thẳng vào container tại `/opt/airflow/project` (`PYTHONPATH` trỏ
tới đó), nên task Airflow gọi `medallion.*` y hệt chạy trên máy host, và ghi
kết quả thẳng ra `data/bronze|silver|gold` trên đĩa thật. Chi tiết đầy đủ +
troubleshooting: [`airflow/README.md`](airflow/README.md).

**Chưa có trong DAG**: bước huấn luyện model (vẫn ở notebook, chạy tay, tách
rời Airflow) — xem mục 9.

### 2.3 Tech stack

| Layer | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.11 | |
| Data | pandas, pyarrow, Parquet, Hive-style partitioning (`dt=YYYY-MM-DD/`) | Bronze/Silver/Gold |
| ML | scikit-learn (LogisticRegression, RandomForest, HistGradientBoosting), LightGBM, XGBoost | 5 model ứng viên, chọn winner theo AUC |
| Model artifact | `joblib` (`.joblib`/`.pkl`) | File-based, không dùng MLflow/model registry ngoài |
| Orchestration | Apache Airflow 2.10.5, LocalExecutor | Tự động hoá Bronze → Silver → Gold |
| Container | Docker + Docker Compose | Đóng gói Airflow (Airflow không chạy gốc trên Windows) |
| Metadata store | Postgres 16 | Chỉ lưu trạng thái chạy DAG/task của Airflow, không lưu dữ liệu nghiệp vụ |
| Test | pytest | Unit test cho `medallion/` (xem mục 7) |

## 3. Lựa chọn metric

Có **2 tầng** đánh giá khác nhau trong bài toán này (giải thích đầy đủ, có ví dụ, nằm trong
`experiment/Full_Training_Model.ipynb`, phần "Recall@K thuộc tầng nào trong bài toán CTR?"):

| Tầng | Câu hỏi | Metric dùng |
|---|---|---|
| **CTR prediction (pointwise)** | Model ước lượng `P(click \| user, ad, context)` cho từng impression riêng lẻ có chính xác không? | **AUC** (threshold-independent, đo khả năng rank 1 click lên trên 1 non-click) |
| **Recommendation / retrieval (listwise)** | Nếu chỉ được hiển thị Top K trong cả catalog, Top K đó có chứa đúng sản phẩm user thật sự muốn click không? | **Recall@10 / @20 / @50 / @100** |

Vì sao cần cả hai: AUC tốt không đảm bảo Top K tốt — hai model AUC gần bằng nhau vẫn có thể cho
chất lượng Top K rất khác nhau (verify được trong notebook: model AUC cao nhất — RandomForest —
không phải lúc nào cũng có Recall@K cao nhất so với các model khác). Ngoài ra còn có:

- **Baseline so sánh**: Random (sàn lý thuyết, Recall@K ≈ K/300) — để biết model thực sự học
  được điều gì ngoài việc rank ngẫu nhiên.
- **CTR-lift/gains chart**: đo CTR thực tế đạt được nếu chỉ hành động trên top X% impression có
  điểm dự đoán cao nhất, so với baseline 19.26% — trả lời câu hỏi kinh doanh "CTR cải thiện được
  bao nhiêu" (không có một con số duy nhất, mà là một đường cong phụ thuộc mức độ chọn lọc).

## 4. Mô tả dữ liệu — Sampling

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

## 5. Mô tả dữ liệu — Các bảng

| File | Vai trò | Số dòng | Nội dung chính |
|---|---|---|---|
| `input_data/train.csv`, `test.csv` (gộp lại = `sampled_dataset.csv`) | **Impression log** — mỗi dòng là 1 lần hiển thị quảng cáo | 300,000 | `userid`, `adgroup_id`, `time_stamp`, `pid` (vị trí hiển thị), `cate_id`, `campaign_id`, `customer`, `brand`, `price`, các cột nhân khẩu học của user tại thời điểm đó, `click` (nhãn 0/1), `date`, `hour`, `weekday` |
| `input_data/ad_feature.csv.zip` | **Catalog quảng cáo đầy đủ** — thuộc tính tĩnh của từng ad (chỉ **300** trong số 846,811 ad này thực sự xuất hiện trong log đã sample ở trên) | 846,811 | `adgroup_id`, `cate_id`, `campaign_id`, `customer`, `brand`, `price` |
| `input_data/user_profile.csv.zip` | **Hồ sơ người dùng đầy đủ** — thuộc tính nhân khẩu học tĩnh | 1,061,768 | `userid`, `cms_segid`, `cms_group_id`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level` |

## 6. Phần phân tích nằm trong EDA

Toàn bộ phân tích khám phá dữ liệu nằm ở **`experiment/EDA.ipynb`** (báo cáo tổng hợp:
[`EDA/EDA_FINAL.md`](EDA/EDA_FINAL.md)), đi từ bức tranh lớn xuống nhỏ (big picture → small):

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

## 7. Kết quả model

5 model ứng viên huấn luyện trên cùng bộ đặc trưng Gold, đánh giá trên tập test 60.000 dòng
(time-based split), so với baseline Random:

| Model | AUC | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|---|---|---|---|---|---|
| **RandomForest (winner)** | **0.607** | **0.099** | **0.157** | **0.340** | **0.513** |
| HistGradientBoosting | 0.606 | 0.079 | 0.143 | 0.306 | 0.495 |
| LogisticRegression | 0.605 | 0.095 | 0.165 | 0.324 | 0.495 |
| LightGBM | 0.603 | 0.073 | 0.136 | 0.301 | 0.500 |
| XGBoost | 0.603 | 0.082 | 0.144 | 0.308 | 0.495 |
| Random baseline | 0.500 | 0.034 | 0.068 | 0.173 | 0.338 |

**Winner: RandomForest** (AUC cao nhất), lưu tại
`experiment/models/full_training_model_v5.joblib` — file `joblib.dump` chứa model đã fit +
danh sách cột categorical/numeric, không có metadata JSON riêng (khác pattern
train.json/evaluate.json của một số dự án khác — xem mục 9).

CTR-lift thực tế: hành động trên top 5% impression điểm cao nhất đạt CTR 32.8% so với baseline
19.26% (lift 1.70×). Chi tiết đầy đủ nằm trong `experiment/Full_Training_Model.ipynb`.

## 8. Cấu trúc Repo

```text
recommendation_system/
├── input_data/                    # Dữ liệu thô (mục 4, 5) — gitignored
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
│   └── README.md                  #   Chi tiết layer, layout data/, quan hệ với notebook
│
├── tests/                         # Unit test cho medallion/ (pytest) — xem mục 10
│   ├── conftest.py
│   ├── test_bronze_ingest.py
│   ├── test_silver_clean.py
│   └── test_gold_features.py
│
├── airflow/                       # Orchestration — xem mục 2.2, 11
│   ├── dags/medallion_pipeline_dag.py
│   ├── Dockerfile                 #   apache/airflow:2.10.5-python3.11 + requirements-dag.txt
│   ├── docker-compose.yaml        #   postgres + airflow-init + webserver + scheduler
│   ├── requirements-dag.txt
│   └── README.md                  #   Hướng dẫn cài đặt/chạy/troubleshoot (tiếng Việt)
│
├── data/                          # Output của medallion pipeline — gitignored, tự sinh lại được
│   └── bronze/  silver/  gold/    #   (partition theo ngày ở bronze/silver)
│
├── experiment/                    # Notebook — nơi thực sự train & phân tích
│   ├── Baseline_Model.ipynb       #   Model baseline gốc (trước khi có medallion)
│   ├── EDA.ipynb                  #   Toàn bộ phân tích ở mục 6
│   ├── Full_Training_Model.ipynb  #   Feature engineering + 5 model + AUC/Recall@K + baseline +
│   │                               #   CTR-lift + feature importance + recommendation JSON
│   └── models/                    #   full_training_model_v5.joblib — champion hiện tại
│
├── EDA/
│   ├── EDA_FINAL.md               #   Báo cáo EDA tổng hợp, hướng action theo Effort×Impact
│   └── images/                    #   Biểu đồ xuất ra từ EDA.ipynb
│
├── output_data/                   # recommendations_top100.json — Top-100 user mẫu, từ notebook
├── models/, app/, logs/           # Scaffold cho phần mở rộng sau này — hiện rỗng, xem mục 12
├── scripts/                       # Wrapper PowerShell — xem mục 9-11
│   ├── setup.ps1                  #   Tạo .venv, cài requirements.txt, check input_data/
│   ├── run_pipeline.ps1           #   Chạy medallion pipeline (toàn bộ hoặc từng layer)
│   ├── run_tests.ps1              #   pytest wrapper
│   ├── airflow_up.ps1             #   docker compose up --build + chờ healthy + hướng dẫn UI
│   └── airflow_down.ps1           #   docker compose down (có tuỳ chọn -Wipe volume)
├── pytest.ini                     # testpaths=tests — tránh pytest quét nhầm airflow/logs/
├── requirements.txt               # Version cố định của toàn bộ dependency (bao gồm pytest)
├── Explain_metric.xlsx            # Giải thích metric bổ sung (dạng bảng tính)
└── README.md                      # File này
```

## 9. Local quick-start

Cách nhanh nhất — chạy script, mỗi bước tự in ra đang làm gì và bước tiếp theo là gì:

```powershell
.\scripts\setup.ps1              # tạo .venv, cài requirements.txt, kiểm tra input_data/
.\scripts\run_pipeline.ps1       # chạy Bronze -> Silver -> Gold, in nơi output được ghi
.\scripts\run_pipeline.ps1 -Layer gold   # hoặc chỉ chạy lại 1 layer (bronze/silver/gold)
```

Tương đương thủ công, nếu muốn hiểu rõ từng lệnh script đang gọi (hoặc không dùng Windows):

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; source .venv/bin/activate trên Unix
pip install -r requirements.txt

# Đặt các file dữ liệu gốc vào input_data/ (mục 4, 5): train.csv, test.csv,
# ad_feature.csv.zip, user_profile.csv.zip

python medallion/run_pipeline.py       # chạy toàn bộ Bronze -> Silver -> Gold

# Hoặc chạy riêng từng layer
python -m medallion.bronze.ingest
python -m medallion.silver.clean
python -m medallion.gold.features
```

Sau khi có `data/gold/training_features.parquet`, mở `experiment/EDA.ipynb` (phân tích, đọc từ
Silver/Gold) hoặc `experiment/Full_Training_Model.ipynb` (huấn luyện — đọc thẳng `input_data/`,
xem lưu ý về trùng lặp có chủ đích ở `medallion/README.md`).

## 10. Chạy tự động qua Airflow

```powershell
.\scripts\airflow_up.ps1     # build + start stack, tự chờ tới khi webserver healthy
# ... dùng xong ...
.\scripts\airflow_down.ps1   # dừng stack (thêm -Wipe để xoá luôn metadata Postgres)
```

`airflow_up.ps1` còn tự xử lý một lỗi thường gặp lần đầu cài Docker Desktop trên Windows: PATH
của phiên PowerShell hiện tại chưa có `docker.exe`/credential helper dù Docker Desktop đã chạy —
script tự thêm đường dẫn cài đặt mặc định vào PATH cho phiên hiện tại, không cần sửa PATH hệ thống.

Tương đương thủ công:

```bash
docker compose -f airflow/docker-compose.yaml up -d --build
```

Mở http://localhost:8080 (`admin` / `admin`), trigger DAG `medallion_pipeline` — 3 task
`bronze_ingest >> silver_clean >> gold_features` sẽ chạy và ghi kết quả thẳng vào
`data/bronze|silver|gold` trên máy host (nhờ bind-mount toàn bộ repo vào container). Hướng dẫn
đầy đủ + bảng troubleshooting: [`airflow/README.md`](airflow/README.md).

## 11. Tests

```powershell
.\scripts\run_tests.ps1              # pytest -q
.\scripts\run_tests.ps1 -Verbose     # pytest -v — in tên từng test
```

Tương đương thủ công: `pytest -q` (cần `pytest.ini` ở repo root để giới hạn discovery vào
`tests/` — nếu không, pytest sẽ cố quét cả `airflow/logs/scheduler/latest`, một reparse point mà
`pathlib` không stat được trên Windows, và collection sẽ lỗi ngay từ đầu).

16 test trong `tests/`, cô lập I/O bằng dữ liệu dựng tay (không đọc `input_data/` thật), phủ 3
module chính của `medallion/`:

| File | Số test | Phủ |
|---|---|---|
| `test_bronze_ingest.py` | 3 | Ingest, partition theo ngày, metadata `_ingested_at`/`_source_file` |
| `test_silver_clean.py` | 6 | Dedupe, assert `click`/`price`, `_fillna_unknown` (null thật → `"unknown"`, không phải sentinel `0`) |
| `test_gold_features.py` | 7 | Target encoding không leak, domain feature chỉ dùng dữ liệu quá khứ, item co-click nhân quả |

## 12. Chưa có / Roadmap

Trung thực về khoảng trống hiện tại, không tô hồng:

- **Huấn luyện model chưa nằm trong Airflow.** DAG `medallion_pipeline` dừng ở Gold; bước train +
  chọn winner (mục 7) vẫn là `Full_Training_Model.ipynb` chạy tay, tách rời hoàn toàn khỏi Airflow.
- **Không có service serving.** `app/` hiện rỗng — chưa có API/UI nào đọc `full_training_model_v5.joblib`
  để trả recommendation theo `userid` thực tế; `output_data/recommendations_top100.json` là kết
  quả tĩnh, sinh 1 lần từ notebook, không phải output của một service sống.
- **`models/`, `logs/` rỗng** — scaffold cho model registry/production logging sau này, chưa có nội
  dung.
- **`gold/snapshots.py` chưa chạy trong pipeline chính** — cần để serving tra cứu feature cho 1
  `userid` bất kỳ mà không phải chạy lại toàn bộ Gold.
- **`.github/workflows/python-publish.yml`** hiện là template PyPI-publish mặc định, **chưa** wire
  vào `pytest`/`tests/` — chưa có CI thật chạy test tự động trên mỗi PR.
- **Recall@K mới đo trên 300 sản phẩm** xuất hiện trong log đã sample, chưa xác nhận lại ở quy mô
  catalog đầy đủ (846,811 ad).

## 13. Tài liệu liên quan

| File | Nội dung |
|---|---|
| [`medallion/README.md`](medallion/README.md) | Chi tiết từng layer, layout `data/`, quan hệ (và rủi ro trùng lặp) giữa `gold/features.py` và notebook |
| [`airflow/README.md`](airflow/README.md) | Cài đặt Docker, chạy/dừng stack, sửa DAG, troubleshooting |
| [`EDA/EDA_FINAL.md`](EDA/EDA_FINAL.md) | Báo cáo EDA tổng hợp — insight, action ưu tiên theo Effort×Impact×Confidence, giới hạn model |
| [`test_reference_report.md`](test_reference_report.md) | Đối chiếu pattern viết test giữa dự án tham khảo và `tests/` của repo này |
| `Explain_metric.xlsx` | Giải thích metric bổ sung dạng bảng tính |
