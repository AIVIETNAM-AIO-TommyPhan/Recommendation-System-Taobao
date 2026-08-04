# Airflow cho pipeline Medallion

Thư mục này chứa cấu hình Airflow để chạy pipeline Bronze → Silver → Gold
(`medallion/`) thông qua giao diện Airflow thay vì gọi tay
`python medallion/run_pipeline.py`.

## Cấu trúc thư mục

```
airflow/
├── dags/
│   └── medallion_pipeline_dag.py   # DAG chính, định nghĩa 3 task
├── Dockerfile                       # image Airflow + pandas/numpy/sklearn/pyarrow
├── docker-compose.yaml              # Postgres + webserver + scheduler (LocalExecutor)
├── requirements-dag.txt             # thư viện Python mà DAG cần, cài thêm vào image
├── .env                             # AIRFLOW_UID cho docker-compose
├── logs/                            # log runtime (tự sinh ra, không commit)
└── plugins/                         # plugin Airflow tuỳ chỉnh (tự sinh ra, không commit)
```

`logs/` và `plugins/` được tạo tự động khi chạy container và đã được thêm
vào `.gitignore`.

## Nguyên lý hoạt động

- `docker-compose.yaml` mount **toàn bộ repo** (`..`) vào container tại
  `/opt/airflow/project`, và set `PYTHONPATH=/opt/airflow/project`. Nhờ vậy
  package `medallion/` import được y hệt như khi chạy trên máy host, và
  `medallion/common/config.py` (dùng `Path(__file__).resolve().parent.parent.parent`
  để suy ra `REPO_ROOT`) vẫn trỏ đúng vào project — không cần sửa code
  pipeline.
- Vì `data/`, `input_data/` nằm trong repo được mount, khi DAG chạy trong
  container, kết quả `data/bronze/`, `data/silver/`, `data/gold/` được ghi
  thẳng ra ổ đĩa Windows của bạn (không mất khi container bị xoá).
- DAG dùng `LocalExecutor` + Postgres — đủ dùng cho 1 máy, không cần
  Celery/Redis.

## Nội dung DAG (`dags/medallion_pipeline_dag.py`)

DAG `medallion_pipeline` gồm 3 task nối tiếp, mỗi task gọi thẳng hàm có
sẵn trong `medallion/` (không viết lại logic):

```python
bronze_ingest = PythonOperator(task_id="bronze_ingest", python_callable=ingest)
silver_clean  = PythonOperator(task_id="silver_clean",  python_callable=clean)
gold_features = PythonOperator(task_id="gold_features", python_callable=build_gold_features)

bronze_ingest >> silver_clean >> gold_features
```

- `schedule=None`: chỉ chạy khi bấm trigger tay, không chạy theo lịch tự
  động — vì `ingest()` đọc lại **toàn bộ** file trong `input_data/` mỗi
  lần chạy chứ không phải đọc dữ liệu mới của "ngày hôm đó", nên không có
  chu kỳ nào hợp lý để đặt lịch.
- Tách 3 task riêng thay vì gọi 1 script gộp (`run_pipeline.py`) để mỗi
  layer có log, trạng thái, và khả năng retry độc lập trên Airflow UI.

## Yêu cầu trước khi chạy

- **Docker Desktop** đã cài và đang chạy (Windows chưa hỗ trợ Airflow
  native, nên bắt buộc phải chạy qua Docker).
  ```
  winget install Docker.DockerDesktop
  ```
  Mở Docker Desktop lên, chờ nó khởi động xong (icon cá voi ở khay hệ
  thống chuyển sang trạng thái chạy) trước khi qua bước tiếp theo.

## Cách chạy

Chạy tất cả lệnh dưới đây từ thư mục gốc repo (`recommendation_system/`).

**1. Build image và khởi động toàn bộ stack:**
```bash
docker compose -f airflow/docker-compose.yaml up -d --build
```
Lần đầu sẽ mất vài phút để tải image Airflow gốc và cài
`pandas`/`numpy`/`scikit-learn`/`pyarrow`.

**2. Kiểm tra các service đã "healthy" chưa:**
```bash
docker compose -f airflow/docker-compose.yaml ps
```

**3. Mở giao diện web:** http://localhost:8080
Đăng nhập: `admin` / `admin` (tài khoản được tạo tự động ở bước
`airflow-init`).

**4. Chạy pipeline:**
- Tìm DAG tên `medallion_pipeline` trong danh sách.
- Nếu DAG đang ở trạng thái pause, bật nó lên (toggle bên trái tên DAG).
- Bấm nút ▶ (Trigger DAG) để chạy thủ công.
- Vào tab **Grid** để xem tiến trình từng task
  (`bronze_ingest → silver_clean → gold_features`).
- Click vào 1 ô task → **Logs** để xem output `print()` giống hệt khi
  chạy bằng tay (vd: `bronze/events: (xxx, xx), N daily partitions`).

**5. Dừng lại:**
```bash
docker compose -f airflow/docker-compose.yaml down
```
Thêm `-v` nếu muốn xoá luôn volume Postgres (reset sạch metadata Airflow,
lần sau chạy lại từ đầu như mới cài):
```bash
docker compose -f airflow/docker-compose.yaml down -v
```

## Sửa/thêm DAG

- Sửa trực tiếp file `dags/medallion_pipeline_dag.py` (hoặc thêm DAG mới
  trong `dags/`) — Airflow scheduler tự quét lại thư mục `dags/` mỗi ~30
  giây, **không cần build lại image**.
- Chỉ cần build lại (`--build`) khi bạn sửa `Dockerfile` hoặc
  `requirements-dag.txt` (tức là thay đổi thư viện Python cài trong
  image).

## Xử lý sự cố thường gặp

| Vấn đề | Nguyên nhân / cách xử lý |
| --- | --- |
| `docker compose` báo lỗi không tìm thấy `docker` | Docker Desktop chưa cài hoặc chưa mở |
| Task `bronze_ingest` lỗi không thấy file trong `input_data/` | Kiểm tra `input_data/` ở host có đủ `train.csv`, `test.csv`, `ad_feature.csv.zip`, `user_profile.csv.zip` chưa — thư mục này được mount qua, không copy vào image |
| Web UI ở `localhost:8080` không load được | Chờ thêm — `airflow-webserver` có healthcheck, cần vài chục giây sau khi container start |
| Sửa DAG nhưng UI không thấy thay đổi | Đợi ~30s cho scheduler quét lại, hoặc restart scheduler: `docker compose -f airflow/docker-compose.yaml restart airflow-scheduler` |
| Muốn xem log chi tiết hơn ngoài UI | `docker compose -f airflow/docker-compose.yaml logs -f airflow-scheduler` |
