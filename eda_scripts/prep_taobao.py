"""Taobao-specific preparation, run before the generic EDA phases.

The generic type inference in ``eda_utils.infer_column_kinds`` keys off dtype, so
every ID and coded level in this dataset (``adgroup_id``, ``cate_id``,
``age_level``, ``weekday``, ...) arrives as an int and gets analysed as a
continuous numeric. Skew, IQR outliers and Pearson correlation are all
meaningless on those. This script assigns each column its real role and decodes
the ``0`` sentinels that stand in for missing values, so the downstream phases
(missing / categorical / train-test shift) analyse what the columns actually mean.

Usage:
    python prep_taobao.py ../dataset/train.csv ../dataset/test.csv --out ../eda_out/prepped
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

TARGET = "click"

# Continuous. The only genuine one in the dataset.
NUMERIC = ["price"]

# Nominal identifiers and coded levels. Integer-typed in the CSV, but the
# integers carry no magnitude — cate_id 4282 is not "more" than cate_id 278.
CATEGORICAL = [
    "adgroup_id", "cate_id", "campaign_id", "customer", "brand",  # ad side
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",  # user side
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
    "pid",                                                          # context
    "hour", "weekday", "date",                                      # time
]

# Row identifier: 147k values over 240k rows. Not a feature.
DROP = ["userid", "time_stamp"]

# Columns where 0 is not a real level but "unknown" in the source tables.
# Deliberately excluded:
#   occupation          — 0 = "not a student" is a genuine level
#   final_gender_code   — coded 1/2, no 0 present
#   cms_group_id, age_level — 0 exists but is only 0.03% of rows, consistent
#     with a small real bucket rather than a missing-value sentinel
SENTINEL_ZERO = ["brand", "pvalue_level", "new_user_class_level", "cms_segid"]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        s = df[col]
        # Normalise to Int64 *before* decoding sentinels. Doing it the other way
        # round drops the column to object dtype, which skips this cast — train
        # would then render brand as "454237" and test as "454237.0" and every
        # level would look unseen at test time.
        if pd.api.types.is_numeric_dtype(s):
            s = s.astype("Float64").astype("Int64")
        if col in SENTINEL_ZERO:
            s = s.replace(0, pd.NA)
        df[col] = s.astype("string")

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("train")
    ap.add_argument("test", nargs="?", default=None)
    ap.add_argument("--out", default="prepped")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for path in filter(None, [args.train, args.test]):
        df = prep(pd.read_csv(path))
        dest = os.path.join(args.out, os.path.basename(path))
        df.to_csv(dest, index=False)
        missing = df.isna().sum()
        print(f"{dest}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  decoded missing: {missing[missing > 0].to_dict()}")


if __name__ == "__main__":
    main()
