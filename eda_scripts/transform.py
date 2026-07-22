"""Build a leak-free cleaning/transform pipeline from EDA findings.

This produces an sklearn ColumnTransformer so the same steps fit on train and apply
to validation/test. Import build_pipeline() to embed in a modeling script, or run as a
CLI to preview the plan and (optionally) write a transformed CSV.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from eda_utils import infer_column_kinds, load_data


def build_plan(df, target=None, high_missing=0.5, high_card=15, scale=True):
    """Return a JSON-able plan describing what will happen to each column.

    Rules mirror references/methodology.md. Callers may hand-edit the plan before
    building the pipeline (e.g. force a column to be dropped or one-hot vs target-encoded).
    """
    kinds = infer_column_kinds(df, target=target)
    n = len(df)
    plan = {"drop": list(kinds["constant"]) + list(kinds["id_like"]),
            "numeric_impute": [], "numeric_log": [], "missing_indicator": [],
            "onehot": [], "target_encode": [], "scale": bool(scale),
            "numeric_cols": [], "categorical_cols": []}

    for col in kinds["numeric"]:
        frac_missing = df[col].isna().mean()
        if frac_missing > high_missing:
            plan["drop"].append(col)
            plan["missing_indicator"].append(col)  # keep presence signal
            continue
        plan["numeric_cols"].append(col)
        if frac_missing > 0:
            plan["numeric_impute"].append(col)
        s = df[col].dropna()
        if s.nunique() > 2 and abs(float(s.skew())) > 1 and s.min() >= 0:
            plan["numeric_log"].append(col)

    for col in kinds["categorical"]:
        frac_missing = df[col].isna().mean()
        if frac_missing > high_missing:
            # High missing rate, but a category's *presence* may itself be predictive.
            # Keep the column: impute missing as its own "Missing" level and add an
            # indicator, rather than discarding a potentially useful signal.
            plan["missing_indicator"].append(col)
        plan["categorical_cols"].append(col)
        if df[col].nunique(dropna=True) <= high_card:
            plan["onehot"].append(col)
        else:
            plan["target_encode"].append(col)
    return plan


def build_pipeline(plan):
    """Construct a ColumnTransformer from a plan. Requires scikit-learn."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                       RobustScaler, StandardScaler)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if plan.get("numeric_log"):
        # Apply log1p only where all values are >= 0; pass others through unchanged.
        # This avoids NaNs on negative columns. For per-column Yeo-Johnson instead,
        # swap in PowerTransformer(method="yeo-johnson").
        def safe_log1p(X):
            X = np.asarray(X, dtype=float)
            out = X.copy()
            nonneg = np.nanmin(X, axis=0) >= 0
            out[:, nonneg] = np.log1p(X[:, nonneg])
            return out
        num_steps.append(("log", FunctionTransformer(safe_log1p, validate=False)))
    if plan.get("scale", True):
        num_steps.append(("scale", RobustScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if plan["numeric_cols"]:
        transformers.append(("num", num_pipe, plan["numeric_cols"]))
    if plan["onehot"]:
        transformers.append(("cat", cat_pipe, plan["onehot"]))
    return ColumnTransformer(transformers, remainder="drop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--apply", metavar="OUT.csv",
                    help="fit+transform and write result (demo; no train/test split)")
    ap.add_argument("--no-scale", action="store_true")
    args = ap.parse_args()

    df = load_data(args.data)
    plan = build_plan(df, target=args.target, scale=not args.no_scale)
    print(json.dumps(plan, indent=2))

    if args.apply:
        pipe = build_pipeline(plan)
        X = df.drop(columns=[args.target]) if args.target in df.columns else df
        out = pipe.fit_transform(X)
        pd.DataFrame(out).to_csv(args.apply, index=False)
        print(f"\nTransformed matrix shape: {getattr(out, 'shape', None)} -> {args.apply}")
        print("NOTE: for real modeling, fit the pipeline on TRAIN only, then transform "
              "validation/test to avoid leakage.")


if __name__ == "__main__":
    main()
