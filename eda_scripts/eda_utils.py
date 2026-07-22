"""Shared helpers for the EDA scripts: loading, type inference, plotting setup."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a tabular file by extension. Supports csv, tsv, parquet, xlsx/xls."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        return pd.read_csv(path)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # Fall back to csv sniffing.
    return pd.read_csv(path, sep=None, engine="python")


def infer_column_kinds(df: pd.DataFrame, target: Optional[str] = None,
                       max_cat_card: int = 20):
    """Split columns into numeric / categorical / id-or-text / constant.

    Heuristics (documented so the caller can override):
      - constant: a single unique non-null value -> useless.
      - id/text: object dtype with (near) all-unique values -> not a categorical.
      - categorical: object/category dtype, OR low-cardinality integer.
      - numeric: everything else numeric.
    """
    numeric, categorical, id_like, constant = [], [], [], []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            constant.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            # An (near) all-unique integer column is almost certainly an ID/row index,
            # not a real numeric feature — flag it so it's excluded from modeling.
            is_intlike = pd.api.types.is_integer_dtype(s) or (
                s.dropna() % 1 == 0).all()
            if is_intlike and nunique > max_cat_card and nunique > 0.95 * n:
                id_like.append(col)
            else:
                # Low-cardinality integers that look categorical are still treated as
                # numeric here; the categorical-vs-target logic can pick them up.
                numeric.append(col)
        else:
            # Object / category / bool.
            if nunique > max_cat_card and nunique > 0.5 * n:
                id_like.append(col)
            else:
                categorical.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "id_like": id_like,
        "constant": constant,
    }


def setup_matplotlib():
    """Import and configure matplotlib for headless plotting. Returns plt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })
    return plt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def skew_label(skew: float) -> str:
    a = abs(skew)
    if a > 1:
        return "strongly skewed"
    if a > 0.5:
        return "moderately skewed"
    return "roughly symmetric"


def iqr_outlier_count(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())
