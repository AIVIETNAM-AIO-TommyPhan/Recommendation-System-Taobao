"""Phase 8: train/test distribution shift (needs a separate test file)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eda_utils import (ensure_dir, infer_column_kinds, load_data,
                       setup_matplotlib)


def _psi(train, test, bins=10):
    """Population Stability Index — a simple shift magnitude for a numeric column."""
    train, test = train.dropna(), test.dropna()
    if train.empty or test.empty:
        return None
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    tr, _ = np.histogram(train, bins=edges)
    te, _ = np.histogram(test, bins=edges)
    tr = tr / max(tr.sum(), 1); te = te / max(te.sum(), 1)
    eps = 1e-6
    tr = np.clip(tr, eps, None); te = np.clip(te, eps, None)
    return float(np.sum((tr - te) * np.log(tr / te)))


def analyze_train_test(train: pd.DataFrame, test: pd.DataFrame, target=None):
    kinds = infer_column_kinds(train, target=target)
    shifts = []
    for col in kinds["numeric"]:
        if col in test.columns:
            psi = _psi(train[col], test[col])
            if psi is None:
                continue
            level = "large" if psi > 0.25 else ("moderate" if psi > 0.1 else "small")
            shifts.append({"column": col, "psi": round(psi, 4), "shift": level})
    shifts.sort(key=lambda x: -x["psi"])

    new_cats = {}
    for col in kinds["categorical"]:
        if col in test.columns:
            unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
            if unseen:
                new_cats[col] = [str(x) for x in list(unseen)[:20]]

    return {"numeric_shift": shifts, "unseen_categories": new_cats,
            "any_large_shift": any(s["shift"] == "large" for s in shifts)}


def plot_train_test(train, test, f, out_dir, top_k=4):
    plt = setup_matplotlib()
    ensure_dir(out_dir)
    top = [s["column"] for s in f["numeric_shift"][:top_k]] or []
    if not top:
        return []
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.5))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, top):
        ax.hist(train[col].dropna(), bins=30, alpha=0.5, label="train", color="#5b8def")
        ax.hist(test[col].dropna(), bins=30, alpha=0.5, label="test", color="#f0883e")
        ax.set_title(col, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("Train vs test distributions")
    p = os.path.join(out_dir, "train_test_shift.png")
    fig.savefig(p); plt.close(fig)
    return [p]


def print_findings(f):
    print("Train/test numeric shift (PSI; >0.25 large):")
    for s in f["numeric_shift"][:10]:
        print(f"  {s['column']}: PSI {s['psi']} ({s['shift']})")
    if f["unseen_categories"]:
        print("Categories in test but not train:")
        for c, vals in f["unseen_categories"].items():
            print(f"  {c}: {vals}")
    if f["any_large_shift"]:
        print("-> Large shift detected: use robust CV; a single validation score may mislead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tr, te = load_data(args.train), load_data(args.test)
    f = analyze_train_test(tr, te, target=args.target)
    plot_train_test(tr, te, f, args.out)
    if args.json:
        print(json.dumps(f, indent=2))
    else:
        print_findings(f)


if __name__ == "__main__":
    main()
