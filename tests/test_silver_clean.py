import pandas as pd
import pytest

from medallion.silver import clean as silver_clean


def _write_bronze_events(bronze_dir, rows_by_day):
    events_dir = bronze_dir / "events"
    for day, rows in rows_by_day.items():
        day_dir = events_dir / f"dt={day}"
        day_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(day_dir / "part.parquet", index=False)


# Tiêu chí: Giá trị thiếu được map về "unknown", các giá trị có sẵn giữ nguyên dạng chuỗi số nguyên.
def test_fillna_unknown_maps_nan_to_unknown_string():
    out = silver_clean._fillna_unknown(pd.Series([1.0, None, 3.0]))

    assert out.tolist() == ["1", "unknown", "3"]


# Tiêu chí: clean_events loại bỏ bản ghi trùng (userid, adgroup_id, time_stamp) và chỉ giữ đúng cột fact.
def test_clean_events_dedups_and_keeps_fact_columns(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    monkeypatch.setattr(silver_clean, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(silver_clean, "SILVER_DIR", silver_dir)

    _write_bronze_events(
        bronze_dir,
        {
            "2017-05-05": {
                "userid": [1, 1],
                "adgroup_id": [10, 10],
                "time_stamp": pd.to_datetime(["2017-05-05 10:00:00"] * 2),
                "click": [0, 0],
                "pid": ["a", "a"],
                "hour": [10, 10],
                "weekday": [4, 4],
                "date": ["2017-05-05", "2017-05-05"],
                "_split": ["train", "train"],
                "_ingested_at": [pd.Timestamp("2024-01-01")] * 2,
            }
        },
    )

    fact_events = silver_clean.clean_events()

    assert len(fact_events) == 1
    assert set(fact_events.columns) == {
        "userid", "adgroup_id", "time_stamp", "click", "pid", "hour", "weekday", "date", "_split",
    }
    assert (silver_dir / "fact_events" / "dt=2017-05-05" / "part.parquet").exists()


# Tiêu chí: clean_events báo lỗi rõ ràng nếu cột click chứa giá trị ngoài {0, 1}.
def test_clean_events_rejects_non_binary_click(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    monkeypatch.setattr(silver_clean, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(silver_clean, "SILVER_DIR", tmp_path / "silver")

    _write_bronze_events(
        bronze_dir,
        {
            "2017-05-05": {
                "userid": [1],
                "adgroup_id": [10],
                "time_stamp": pd.to_datetime(["2017-05-05 10:00:00"]),
                "click": [2],
                "pid": ["a"],
                "hour": [10],
                "weekday": [4],
                "date": ["2017-05-05"],
                "_split": ["train"],
            }
        },
    )

    with pytest.raises(AssertionError, match="binary"):
        silver_clean.clean_events()


# Tiêu chí: clean_dim_ad điền "unknown" cho brand thiếu và loại bản ghi trùng adgroup_id (giữ bản mới nhất).
def test_clean_dim_ad_fills_unknown_brand_and_dedups(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    monkeypatch.setattr(silver_clean, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(silver_clean, "SILVER_DIR", silver_dir)

    pd.DataFrame(
        {
            "adgroup_id": [1, 1],
            "cate_id": [100, 100],
            "campaign_id": [200, 201],
            "customer": [300, 300],
            "brand": [None, 400.0],
            "price": [9.9, 19.9],
        }
    ).to_parquet(bronze_dir / "ad_feature.parquet", index=False)

    dim_ad = silver_clean.clean_dim_ad()

    assert len(dim_ad) == 1
    assert dim_ad.loc[0, "campaign_id"] == 201
    assert dim_ad.loc[0, "brand"] == "400"


# Tiêu chí: clean_dim_ad chặn dữ liệu có price không dương trước khi ghi ra Silver.
def test_clean_dim_ad_rejects_non_positive_price(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    monkeypatch.setattr(silver_clean, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(silver_clean, "SILVER_DIR", silver_dir)

    pd.DataFrame(
        {
            "adgroup_id": [1],
            "cate_id": [100],
            "campaign_id": [200],
            "customer": [300],
            "brand": [400.0],
            "price": [0.0],
        }
    ).to_parquet(bronze_dir / "ad_feature.parquet", index=False)

    with pytest.raises(AssertionError, match="price"):
        silver_clean.clean_dim_ad()


# Tiêu chí: clean_dim_user điền "unknown" cho pvalue_level và new_user_class_level còn thiếu.
def test_clean_dim_user_fills_unknown_for_sentinel_columns(tmp_path, monkeypatch):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    monkeypatch.setattr(silver_clean, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(silver_clean, "SILVER_DIR", silver_dir)

    pd.DataFrame(
        {
            "userid": [1],
            "cms_segid": [0],
            "cms_group_id": [1],
            "final_gender_code": [1],
            "age_level": [2],
            "pvalue_level": [None],
            "shopping_level": [3],
            "occupation": [0],
            "new_user_class_level": [None],
        }
    ).to_parquet(bronze_dir / "user_profile.parquet", index=False)

    dim_user = silver_clean.clean_dim_user()

    assert dim_user.loc[0, "pvalue_level"] == "unknown"
    assert dim_user.loc[0, "new_user_class_level"] == "unknown"
    assert dim_user.loc[0, "cms_segid"] == 0
