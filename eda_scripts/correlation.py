"""Phase 7: correlation between numerical features + redundancy detection."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import (ensure_dir, infer_column_kinds, load_data,
                       setup_matplotlib)


def analyze_correlation(df: pd.DataFrame, target=None, strong=0.8):
    kinds = infer_column_kinds(df, target=target)
    num = kinds["numeric"]
    # Include a numeric target in the matrix if it is numeric.
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        num = num + [target]
    if len(num) < 2:
        return {"error": "need >=2 numeric columns for correlation"}
    corr = df[num].corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.iloc[i, j])
            if not np.isnan(c) and abs(c) >= strong:
                pairs.append({"a": cols[i], "b": cols[j], "corr": round(c, 3)})
    pairs.sort(key=lambda x: -abs(x["corr"]))
    return {"columns": cols, "strong_pairs": pairs,
            "matrix": {a: {b: round(float(corr.loc[a, b]), 3) for b in cols} for a in cols}}


def plot_correlation(f, out_dir):
    if "error" in f:
        return None
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    cols = f["columns"]
    M = np.array([[f["matrix"][a][b] for b in cols] for a in cols])
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 3, 0.6 * len(cols) + 3))
    im = ax.imshow(M, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, label="correlation", shrink=0.8)
    ax.set_title("Correlation heatmap (numerical features)")
    p = os.path.join(out_dir, "correlation_heatmap.png")
    fig.savefig(p); plt.close(fig)
    return p


def print_findings(f):
    if "error" in f:
        print(f["error"]); return
    if not f["strong_pairs"]:
        print("No strongly correlated pairs (|r| >= 0.8).")
        return
    print("Strongly correlated pairs (possible redundancy):")
    for p in f["strong_pairs"]:
        print(f"  {p['a']} ~ {p['b']}: r = {p['corr']}")
    print("Reminder: correlation is linear co-movement, not causation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_correlation(df, target=args.target)
    plot_correlation(f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
