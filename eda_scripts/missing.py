"""Phase 3: missing values — how much, and is missingness informative?"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import ensure_dir, load_data, setup_matplotlib


def analyze_missing(df: pd.DataFrame, target=None, high_thresh=0.5):
    n = len(df)
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    cols = []
    for col, cnt in miss.items():
        frac = cnt / n
        entry = {"column": col, "missing": int(cnt), "fraction": round(float(frac), 4),
                 "high_missing": bool(frac > high_thresh)}
        cols.append(entry)

    # Informative-missingness check: does the missing rate vary across target classes?
    informative = []
    if target is not None and target in df.columns and not df[target].isna().all():
        t = df[target]
        classification = not (pd.api.types.is_numeric_dtype(t) and t.nunique() > 20)
        for entry in cols:
            col = entry["column"]
            ind = df[col].isna().astype(int)
            if classification:
                rates = ind.groupby(t).mean()
                spread = float(rates.max() - rates.min()) if len(rates) else 0.0
                entry["missrate_by_class"] = {str(k): round(float(v), 4)
                                              for k, v in rates.items()}
                entry["missrate_spread"] = round(spread, 4)
                if spread > 0.10:  # >10 pp difference => informative
                    entry["informative"] = True
                    informative.append(col)
                else:
                    entry["informative"] = False
            else:
                # Regression: correlation between missing-indicator and target.
                corr = float(pd.Series(ind).corr(t.astype(float)))
                entry["missind_target_corr"] = round(corr, 4) if not np.isnan(corr) else 0.0
                entry["informative"] = abs(entry["missind_target_corr"]) > 0.05
                if entry["informative"]:
                    informative.append(col)

    return {"n_rows": n, "columns_with_missing": cols,
            "informative_columns": informative,
            "any_missing": len(cols) > 0}


def plot_missing(f, out_dir):
    cols = f["columns_with_missing"]
    if not cols:
        return None
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    names = [c["column"] for c in cols][::-1]
    fracs = [c["fraction"] * 100 for c in cols][::-1]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(names) + 1)))
    ax.barh(names, fracs, color="#5b8def")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    path = os.path.join(out_dir, "missing_values.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def print_findings(f):
    if not f["any_missing"]:
        print("No missing values.")
        return
    print("Missing values:")
    for c in f["columns_with_missing"]:
        line = f"  {c['column']}: {c['fraction']*100:.1f}%"
        if c.get("high_missing"):
            line += " [>50%]"
        if "informative" in c:
            line += " [INFORMATIVE]" if c["informative"] else " [random]"
        print(line)
    if f["informative_columns"]:
        print(f"Informative missingness (add indicators): {f['informative_columns']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_missing(df, target=args.target)
    plot_missing(f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
