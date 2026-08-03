import pandas as pd

from medallion.bronze import ingest as bronze_ingest


def _write_events_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


# Tiêu chí: events được partition đúng theo ngày và gắn đúng _split/_source_file cho từng nguồn.
def test_ingest_events_partitions_by_day_and_tags_source(tmp_path, monkeypatch, ingested_at):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    bronze_dir = tmp_path / "bronze"
    monkeypatch.setattr(bronze_ingest, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)

    _write_events_csv(
        raw_dir / "train.csv",
        {
            "userid": [1, 2],
            "time_stamp": ["2017-05-05 10:00:00", "2017-05-06 11:00:00"],
            "date": ["2017-05-05", "2017-05-06"],
        },
    )
    _write_events_csv(
        raw_dir / "test.csv",
        {"userid": [3], "time_stamp": ["2017-05-05 12:00:00"], "date": ["2017-05-05"]},
    )

    bronze_ingest._ingest_events(ingested_at)

    day1 = pd.read_parquet(bronze_dir / "events" / "dt=2017-05-05" / "part.parquet")
    day2 = pd.read_parquet(bronze_dir / "events" / "dt=2017-05-06" / "part.parquet")

    assert sorted(day1["_split"]) == ["test", "train"]
    assert day1["_source_file"].isin(["train.csv", "test.csv"]).all()
    assert (day1["_ingested_at"] == ingested_at).all()
    assert len(day2) == 1


# Tiêu chí: Tên cột có khoảng trắng thừa được strip, và mỗi dòng được gắn nguồn gốc file.
def test_ingest_dim_strips_column_whitespace_and_tags_source(tmp_path, monkeypatch, ingested_at):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)

    src_path = tmp_path / "user_profile.csv"
    pd.DataFrame({"userid": [1, 2], " new_user_class_level ": [1, None]}).to_csv(src_path, index=False)

    bronze_ingest._ingest_dim("user_profile", src_path, ingested_at)

    out = pd.read_parquet(bronze_dir / "user_profile.parquet")
    assert "new_user_class_level" in out.columns
    assert not any(col.strip() != col for col in out.columns)
    assert (out["_source_file"] == "user_profile.csv").all()


# Tiêu chí: ingest() ghi đủ 3 output (events theo ngày + 2 file dim) từ raw đầu vào thật.
def test_ingest_writes_events_and_dim_parquets_end_to_end(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    bronze_dir = tmp_path / "bronze"
    monkeypatch.setattr(bronze_ingest, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bronze_ingest, "BRONZE_DIR", bronze_dir)

    _write_events_csv(
        raw_dir / "train.csv",
        {"userid": [1], "time_stamp": ["2017-05-05 10:00:00"], "date": ["2017-05-05"]},
    )
    _write_events_csv(
        raw_dir / "test.csv",
        {"userid": [2], "time_stamp": ["2017-05-05 11:00:00"], "date": ["2017-05-05"]},
    )
    pd.DataFrame({"adgroup_id": [10], "price": [99.0]}).to_csv(
        raw_dir / "ad_feature.csv.zip", compression="zip", index=False
    )
    pd.DataFrame({"userid": [1, 2], "cms_segid": [0, 1]}).to_csv(
        raw_dir / "user_profile.csv.zip", compression="zip", index=False
    )

    returned_dir = bronze_ingest.ingest()

    assert returned_dir == bronze_dir
    assert (bronze_dir / "events" / "dt=2017-05-05" / "part.parquet").exists()
    assert (bronze_dir / "ad_feature.parquet").exists()
    assert (bronze_dir / "user_profile.parquet").exists()
