"""Silver layer: clean, validate, and conform Bronze into a fact table plus
two dimension tables. Lightweight transformation only - drop duplicates,
format/cast columns, fill genuine nulls - no feature engineering.

Verified directly against the RAW ad_feature.csv/user_profile.csv (not the
already-preprocessed train/test csvs, which had silently fillna(0)'d these
same columns upstream in Sample_Dataset.ipynb): `brand`, `pvalue_level`, and
`new_user_class_level` have real NaN for missing values. `cms_segid` has ZERO
nulls in the raw source - its 0 is a genuine segment id, not a sentinel. That
corrects an assumption every earlier notebook in this repo carried over from
feat/tree's EDA (which only had the post-fillna(0) train/test csvs to look at).

`price` is a static ad attribute (verified: identical to ad_feature.csv for
every adgroup_id, 0 mismatches) so it lives in dim_ad, not the fact table.
"""
import pandas as pd

from medallion.common.config import BRONZE_DIR, SILVER_DIR


def clean_events():
    parts = sorted((BRONZE_DIR / "events").glob("dt=*/part.parquet"))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined = combined.sort_values("time_stamp").reset_index(drop=True)

    combined = combined.drop_duplicates(subset=["userid", "adgroup_id", "time_stamp"])
    assert combined["click"].isin([0, 1]).all(), "click must be binary"

    fact_events = combined[[
        "userid", "adgroup_id", "time_stamp", "click", "pid",
        "hour", "weekday", "date", "_split",
    ]].reset_index(drop=True)

    fact_dir = SILVER_DIR / "fact_events"
    fact_dir.mkdir(parents=True, exist_ok=True)
    n_days = 0
    for day, day_df in fact_events.groupby(fact_events["date"].astype(str)):
        day_dir = fact_dir / f"dt={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        day_df.to_parquet(day_dir / "part.parquet", index=False)
        n_days += 1
    print(f"silver/fact_events: {fact_events.shape}, {n_days} daily partitions")
    return fact_events


def _fillna_unknown(series):
    out = series.fillna(-1).astype("Int64").astype(str)
    out[out == "-1"] = "unknown"
    return out


def clean_dim_ad():
    ad = pd.read_parquet(BRONZE_DIR / "ad_feature.parquet")
    ad = ad.drop_duplicates(subset="adgroup_id", keep="last")
    assert (ad["price"] > 0).all(), "price must be positive"

    ad["brand"] = _fillna_unknown(ad["brand"])
    dim_ad = ad[["adgroup_id", "cate_id", "campaign_id", "customer", "brand", "price"]].reset_index(drop=True)
    dim_ad.to_parquet(SILVER_DIR / "dim_ad.parquet", index=False)
    print("silver/dim_ad.parquet:", dim_ad.shape)
    return dim_ad


def clean_dim_user():
    user = pd.read_parquet(BRONZE_DIR / "user_profile.parquet")
    user = user.drop_duplicates(subset="userid", keep="last")

    for col in ("pvalue_level", "new_user_class_level"):
        user[col] = _fillna_unknown(user[col])

    dim_user = user[[
        "userid", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
        "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
    ]].reset_index(drop=True)
    dim_user.to_parquet(SILVER_DIR / "dim_user.parquet", index=False)
    print("silver/dim_user.parquet:", dim_user.shape)
    return dim_user


def clean():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    clean_events()
    clean_dim_ad()
    clean_dim_user()
    return SILVER_DIR


if __name__ == "__main__":
    clean()
