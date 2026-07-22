"""Phase 1: dataset overview & data types. Also phase 2 target balance."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from eda_utils import infer_column_kinds, load_data, skew_label


def analyze_overview(df: pd.DataFrame, target=None):
    kinds = infer_column_kinds(df, target=target)
    dup = int(df.duplicated().sum())
    wrong_dtype = []
    for col in kinds["categorical"]:
        # Object columns that are fully numeric-parseable are likely mistyped.
        s = df[col].dropna()
        if s.empty:
            continue
        parsed = pd.to_numeric(s, errors="coerce")
        if parsed.notna().mean() > 0.95:
            wrong_dtype.append(col)

    findings = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicate_rows": dup,
        "column_kinds": kinds,
        "possibly_mistyped_numeric": wrong_dtype,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    if target is not None and target in df.columns:
        t = df[target]
        if pd.api.types.is_numeric_dtype(t) and t.nunique() > 20:
            findings["target"] = {
                "name": target, "task": "regression",
                "skew": round(float(t.skew()), 3),
                "skew_label": skew_label(float(t.skew())),
            }
        else:
            vc = t.value_counts(dropna=False)
            frac = (vc / vc.sum()).round(4)
            imbalance = float(frac.max() / frac.min()) if frac.min() > 0 else float("inf")
            findings["target"] = {
                "name": target, "task": "classification",
                "class_counts": {str(k): int(v) for k, v in vc.items()},
                "class_fractions": {str(k): float(v) for k, v in frac.items()},
                "imbalance_ratio": round(imbalance, 2),
                "imbalanced": imbalance > 1.5,
            }
    return findings


def print_findings(f):
    print(f"Shape: {f['n_rows']} rows x {f['n_cols']} cols | {f['memory_mb']} MB")
    print(f"Duplicate rows: {f['duplicate_rows']}")
    k = f["column_kinds"]
    print(f"Numeric ({len(k['numeric'])}): {k['numeric']}")
    print(f"Categorical ({len(k['categorical'])}): {k['categorical']}")
    if k["id_like"]:
        print(f"ID/text-like (exclude from modeling): {k['id_like']}")
    if k["constant"]:
        print(f"Constant (drop): {k['constant']}")
    if f["possibly_mistyped_numeric"]:
        print(f"Possibly numeric stored as text: {f['possibly_mistyped_numeric']}")
    if "target" in f:
        t = f["target"]
        if t["task"] == "classification":
            print(f"Target '{t['name']}': classification, "
                  f"imbalance ratio {t['imbalance_ratio']} "
                  f"({'IMBALANCED' if t['imbalanced'] else 'balanced'})")
            print(f"  classes: {t['class_counts']}")
        else:
            print(f"Target '{t['name']}': regression, "
                  f"skew {t['skew']} ({t['skew_label']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_overview(df, target=args.target)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
