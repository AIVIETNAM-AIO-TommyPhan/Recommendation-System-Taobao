"""Bronze layer: land raw source files from dataset/ as-is.

The event log (every ad impression - both clicked and non-clicked, `click`
is just an outcome flag on each row, not a filter) is stored partitioned by
day (data/bronze/events/dt=YYYY-MM-DD/); ad_feature/user_profile are
reference/dimension snapshots, landed whole since they aren't naturally a
time series in this dataset. No cleaning, no joins, no dedup - a faithful,
immutable copy of the source files plus ingestion metadata (column-name
whitespace stripping is the one exception, since a column literally named
"new_user_class_level " with a trailing space is a schema defect, not a
data value).
"""
import pandas as pd

from medallion.common.config import BRONZE_DIR, RAW_DIR


def _ingest_events(ingested_at):
    events_dir = BRONZE_DIR / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(RAW_DIR / f"{split}.csv", parse_dates=["time_stamp"])
        df["_split"] = split
        df["_source_file"] = f"{split}.csv"
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["_ingested_at"] = ingested_at

    n_days = 0
    for day, day_df in combined.groupby(combined["date"].astype(str)):
        day_dir = events_dir / f"dt={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        day_df.to_parquet(day_dir / "part.parquet", index=False)
        n_days += 1
    print(f"bronze/events: {combined.shape}, {n_days} daily partitions")


def _ingest_dim(name, path, ingested_at, **read_csv_kwargs):
    df = pd.read_csv(path, **read_csv_kwargs)
    df.columns = df.columns.str.strip()
    df["_ingested_at"] = ingested_at
    df["_source_file"] = path.name
    df.to_parquet(BRONZE_DIR / f"{name}.parquet", index=False)
    print(f"bronze/{name}.parquet: {df.shape}")


def ingest():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    ingested_at = pd.Timestamp.now(tz="UTC")

    _ingest_events(ingested_at)
    _ingest_dim("ad_feature", RAW_DIR / "ad_feature.csv.zip", ingested_at, compression="zip")
    _ingest_dim("user_profile", RAW_DIR / "user_profile.csv.zip", ingested_at, compression="zip")

    return BRONZE_DIR


if __name__ == "__main__":
    ingest()
