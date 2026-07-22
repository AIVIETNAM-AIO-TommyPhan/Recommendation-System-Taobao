"""Phase 5: categorical feature distributions — cardinality, rare, encoding hint."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import (ensure_dir, infer_column_kinds, load_data,
                       setup_matplotlib)


def analyze_categorical(df: pd.DataFrame, target=None, rare_thresh=0.01,
                        high_card=15):
    kinds = infer_column_kinds(df, target=target)
    cats = kinds["categorical"]
    n = len(df)
    feats = []
    for col in cats:
        vc = df[col].value_counts(dropna=False)
        card = int(df[col].nunique(dropna=True))
        rare = [str(k) for k, v in vc.items() if v / n < rare_thresh]
        if card <= high_card:
            encoding = "one-hot"
        else:
            encoding = "target/frequency encoding (high cardinality)"
        feats.append({
            "column": col, "cardinality": card,
            "top": {str(k): int(v) for k, v in vc.head(6).items()},
            "rare_categories": rare,
            "has_rare": len(rare) > 0,
            "suggested_encoding": encoding,
        })
    return {"features": feats}


def plot_categorical(df, f, out_dir, max_plots=6):
    feats = f["features"]
    if not feats:
        return []
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    cols = [x["column"] for x in feats][:max_plots]
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts(dropna=False).head(12)
        vc.plot(kind="bar", ax=ax, color="#2b7fb8")
        ax.set_title(f"{col} (card {df[col].nunique()})", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Categorical feature counts")
    p = os.path.join(out_dir, "categorical_counts.png")
    fig.savefig(p); plt.close(fig)
    return [p]


def print_findings(f):
    for x in f["features"]:
        tail = f" | {len(x['rare_categories'])} rare -> group into 'Other'" if x["has_rare"] else ""
        print(f"  {x['column']}: cardinality {x['cardinality']} "
              f"-> {x['suggested_encoding']}{tail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_categorical(df, target=args.target)
    plot_categorical(df, f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
