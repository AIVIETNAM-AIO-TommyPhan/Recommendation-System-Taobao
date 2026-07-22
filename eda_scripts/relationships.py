"""Phase 6: feature <-> target relationships (needs a target)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import (ensure_dir, infer_column_kinds, load_data,
                       setup_matplotlib)


def analyze_relationships(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return {"error": f"target '{target}' not in columns"}
    kinds = infer_column_kinds(df, target=target)
    t = df[target]
    classification = not (pd.api.types.is_numeric_dtype(t) and t.nunique() > 20)

    numeric_sep = []
    for col in kinds["numeric"]:
        if classification:
            grp = df.groupby(t, observed=True)[col].median()
            if grp.notna().sum() < 2:
                continue
            spread = float(grp.max() - grp.min())
            overall = float(df[col].std()) or 1.0
            numeric_sep.append({
                "column": col,
                "median_by_class": {str(k): round(float(v), 3) for k, v in grp.items()},
                "separation": round(spread / overall, 3),  # spread in std units
            })
        else:
            corr = float(df[col].corr(t.astype(float)))
            numeric_sep.append({"column": col,
                                "target_corr": round(corr, 3) if not np.isnan(corr) else 0.0})
    if classification:
        numeric_sep.sort(key=lambda x: -x.get("separation", 0))
    else:
        numeric_sep.sort(key=lambda x: -abs(x.get("target_corr", 0)))

    cat_sep = []
    if classification:
        for col in kinds["categorical"]:
            ct = pd.crosstab(df[col], t, normalize="index")
            # Variation of a class's proportion across categories = usefulness proxy.
            var = float(ct.std(axis=0).max()) if not ct.empty else 0.0
            cat_sep.append({"column": col, "proportion_spread": round(var, 3)})
        cat_sep.sort(key=lambda x: -x["proportion_spread"])

    return {"task": "classification" if classification else "regression",
            "numeric_vs_target": numeric_sep, "categorical_vs_target": cat_sep}


def plot_relationships(df, target, f, out_dir, top_k=4):
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    paths = []
    t = df[target]
    if f["task"] == "classification":
        top = [x["column"] for x in f["numeric_vs_target"][:top_k]]
        if top:
            fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4))
            axes = np.atleast_1d(axes)
            classes = list(pd.Series(t).dropna().unique())
            for ax, col in zip(axes, top):
                data = [df.loc[t == c, col].dropna().values for c in classes]
                ax.boxplot(data, tick_labels=[str(c) for c in classes], showfliers=False)
                ax.set_title(col, fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            fig.suptitle("Top numeric features by class separation")
            p = os.path.join(out_dir, "numeric_by_target.png")
            fig.savefig(p); plt.close(fig); paths.append(p)

        # Best categorical as stacked-percentage bar.
        if f["categorical_vs_target"]:
            col = f["categorical_vs_target"][0]["column"]
            ct = pd.crosstab(df[col], t, normalize="index") * 100
            fig, ax = plt.subplots(figsize=(7, 4))
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("% of rows")
            ax.set_title(f"Target proportion by {col}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            p = os.path.join(out_dir, "categorical_vs_target.png")
            fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


def print_findings(f):
    if "error" in f:
        print(f["error"]); return
    print(f"Task: {f['task']}")
    print("Numeric features by target relationship (strongest first):")
    for x in f["numeric_vs_target"][:8]:
        if "separation" in x:
            print(f"  {x['column']}: class-median separation {x['separation']} std")
        else:
            print(f"  {x['column']}: target corr {x['target_corr']}")
    if f["categorical_vs_target"]:
        print("Categorical features by target relationship:")
        for x in f["categorical_vs_target"][:8]:
            print(f"  {x['column']}: proportion spread {x['proportion_spread']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    df = load_data(args.data)
    f = analyze_relationships(df, args.target)
    if "error" not in f:
        plot_relationships(df, args.target, f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
