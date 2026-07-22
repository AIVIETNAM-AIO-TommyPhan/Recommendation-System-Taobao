"""Run the EDA phases on the Taobao data with correct column roles.

Mirrors ``eda_report.run`` but keeps the prepared DataFrame in memory instead of
round-tripping through CSV — writing to CSV and reloading would let pandas
re-infer every string-cast ID back to an int, undoing ``prep_taobao.prep``.

Usage:
    python eda_taobao.py --train ../dataset/train.csv --test ../dataset/test.csv --out ../eda_out
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from eda_utils import ensure_dir

import overview as m_overview
import missing as m_missing
import numerical as m_numerical
import categorical as m_categorical
import relationships as m_relationships
import correlation as m_correlation
import train_test as m_train_test
import transform as m_transform
import eda_report

from prep_taobao import TARGET, prep


def run(train_path, test_path=None, out="eda_out"):
    ensure_dir(out)
    plots_dir = ensure_dir(os.path.join(out, "plots"))

    df = prep(pd.read_csv(train_path))

    findings = {"dataset": os.path.basename(train_path), "target": TARGET}
    findings["overview"] = m_overview.analyze_overview(df, target=TARGET)
    findings["missing"] = m_missing.analyze_missing(df, target=TARGET)
    m_missing.plot_missing(findings["missing"], plots_dir)
    findings["numerical"] = m_numerical.analyze_numerical(df, target=TARGET)
    m_numerical.plot_numerical(df, findings["numerical"], plots_dir)
    findings["categorical"] = m_categorical.analyze_categorical(df, target=TARGET)
    m_categorical.plot_categorical(df, findings["categorical"], plots_dir)

    rel = m_relationships.analyze_relationships(df, TARGET)
    findings["relationships"] = rel
    if "error" not in rel:
        m_relationships.plot_relationships(df, TARGET, rel, plots_dir)

    findings["correlation"] = m_correlation.analyze_correlation(df, target=TARGET)
    m_correlation.plot_correlation(findings["correlation"], plots_dir)

    if test_path:
        te = prep(pd.read_csv(test_path))
        tt = m_train_test.analyze_train_test(df, te, target=TARGET)
        findings["train_test"] = tt
        m_train_test.plot_train_test(df, te, tt, plots_dir)

    findings["transform_plan"] = m_transform.build_plan(df, target=TARGET)

    with open(os.path.join(out, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, default=str)
    with open(os.path.join(out, "report_generated.md"), "w", encoding="utf-8") as fh:
        fh.write(eda_report.build_report(findings, plots_dir))
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="dataset/train.csv")
    ap.add_argument("--test", default="dataset/test.csv")
    ap.add_argument("--out", default="eda_out")
    args = ap.parse_args()
    f = run(args.train, args.test, args.out)
    k = f["overview"]["column_kinds"]
    print(f"numeric={k['numeric']}")
    print(f"categorical={len(k['categorical'])} cols, id_like={k['id_like']}")
    print(f"Done. See {args.out}/report_generated.md")


if __name__ == "__main__":
    main()
