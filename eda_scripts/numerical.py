"""Phase 4: numerical feature distributions — skew, outliers, scale."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import (ensure_dir, infer_column_kinds, iqr_outlier_count,
                       load_data, setup_matplotlib, skew_label)


def analyze_numerical(df: pd.DataFrame, target=None):
    kinds = infer_column_kinds(df, target=target)
    num = kinds["numeric"]
    feats = []
    for col in num:
        s = df[col].dropna()
        if s.empty:
            continue
        skew = float(s.skew()) if s.nunique() > 2 else 0.0
        feats.append({
            "column": col,
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()), "mean": float(s.mean()),
            "std": float(s.std()),
            "skew": round(skew, 3), "skew_label": skew_label(skew),
            "outliers_iqr": iqr_outlier_count(s),
            "all_positive": bool(s.min() > 0),
        })
    # Scale spread across features: ratio of largest to smallest typical magnitude.
    scales = [abs(f["median"]) if f["median"] != 0 else abs(f["mean"]) for f in feats]
    scales = [x for x in scales if x > 0]
    scale_ratio = round(max(scales) / min(scales), 1) if len(scales) > 1 else 1.0
    return {"features": feats, "scale_ratio": scale_ratio,
            "very_different_scales": scale_ratio > 100}


def plot_numerical(df, f, out_dir, max_plots=8):
    feats = f["features"]
    if not feats:
        return []
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    cols = [x["column"] for x in feats][:max_plots]
    paths = []

    # Histograms grid.
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        df[col].dropna().hist(ax=ax, bins=30, color="#5b8def")
        ax.set_title(col, fontsize=9)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numerical feature distributions")
    p = os.path.join(out_dir, "numerical_histograms.png")
    fig.savefig(p); plt.close(fig); paths.append(p)

    # Boxplots for outliers (standardized so they share a scale).
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 5))
    data = []
    labels = []
    for col in cols:
        s = df[col].dropna()
        if s.std() > 0:
            data.append(((s - s.mean()) / s.std()).values)
            labels.append(col)
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("standardized value")
    ax.set_title("Boxplots (standardized) — fliers are outliers")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = os.path.join(out_dir, "numerical_boxplots.png")
    fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


def print_findings(f):
    print(f"Scale ratio across features: {f['scale_ratio']} "
          f"({'VERY different scales' if f['very_different_scales'] else 'comparable'})")
    for x in f["features"]:
        note = []
        if abs(x["skew"]) > 1:
            note.append("log/Yeo-Johnson" if not x["all_positive"] else "log transform")
        if x["outliers_iqr"] > 0:
            note.append(f"{x['outliers_iqr']} outliers")
        tail = f"  -> {', '.join(note)}" if note else ""
        print(f"  {x['column']}: skew {x['skew']} ({x['skew_label']}){tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_numerical(df, target=args.target)
    plot_numerical(df, f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
