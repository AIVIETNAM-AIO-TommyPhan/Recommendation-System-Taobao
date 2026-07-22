"""Orchestrator: run all EDA phases, save plots, emit findings.json + report.md."""
from __future__ import annotations

import argparse
import json
import os

from eda_utils import ensure_dir, load_data

import overview as m_overview
import missing as m_missing
import numerical as m_numerical
import categorical as m_categorical
import relationships as m_relationships
import correlation as m_correlation
import transform as m_transform


def run(data, target=None, test=None, out="eda_out"):
    ensure_dir(out)
    plots_dir = ensure_dir(os.path.join(out, "plots"))
    df = load_data(data)

    findings = {"dataset": os.path.basename(data), "target": target}

    findings["overview"] = m_overview.analyze_overview(df, target=target)

    findings["missing"] = m_missing.analyze_missing(df, target=target)
    m_missing.plot_missing(findings["missing"], plots_dir)

    findings["numerical"] = m_numerical.analyze_numerical(df, target=target)
    m_numerical.plot_numerical(df, findings["numerical"], plots_dir)

    findings["categorical"] = m_categorical.analyze_categorical(df, target=target)
    m_categorical.plot_categorical(df, findings["categorical"], plots_dir)

    if target and target in df.columns:
        rel = m_relationships.analyze_relationships(df, target)
        findings["relationships"] = rel
        if "error" not in rel:
            m_relationships.plot_relationships(df, target, rel, plots_dir)

    findings["correlation"] = m_correlation.analyze_correlation(df, target=target)
    m_correlation.plot_correlation(findings["correlation"], plots_dir)

    if test:
        import train_test as m_tt
        te = load_data(test)
        tt = m_tt.analyze_train_test(df, te, target=target)
        findings["train_test"] = tt
        m_tt.plot_train_test(df, te, tt, plots_dir)

    findings["transform_plan"] = m_transform.build_plan(df, target=target)

    with open(os.path.join(out, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, default=str)

    report = build_report(findings, plots_dir)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    return findings


def _rel(plots_dir, name):
    return os.path.join("plots", name)


def build_report(f, plots_dir):
    ov = f["overview"]
    lines = []
    A = lines.append
    A(f"# EDA Report — {f['dataset']}\n")
    A(f"**Shape:** {ov['n_rows']} rows × {ov['n_cols']} columns  ")
    A(f"**Target:** {f['target'] or 'none (exploratory)'}\n")

    A("## 1. Overview & data types\n")
    k = ov["column_kinds"]
    A(f"- Numeric: {len(k['numeric'])} · Categorical: {len(k['categorical'])}"
      f" · ID/text: {len(k['id_like'])} · Constant: {len(k['constant'])}")
    A(f"- Duplicate rows: {ov['duplicate_rows']}")
    if k["constant"]:
        A(f"- **Drop (constant):** {k['constant']}")
    if k["id_like"]:
        A(f"- **Exclude from modeling (ID/text):** {k['id_like']}")
    if ov["possibly_mistyped_numeric"]:
        A(f"- **Possibly numeric stored as text:** {ov['possibly_mistyped_numeric']}")
    A("")

    if "target" in ov:
        t = ov["target"]
        A("## 2. Target balance\n")
        if t["task"] == "classification":
            A(f"Classification target. Imbalance ratio **{t['imbalance_ratio']}** "
              f"({'imbalanced → stratified CV, class_weight, per-class metrics' if t['imbalanced'] else 'reasonably balanced'}).")
            A(f"Class counts: {t['class_counts']}\n")
        else:
            A(f"Regression target, skew {t['skew']} ({t['skew_label']}). "
              f"{'Consider modeling log(target).' if abs(t['skew'])>1 else ''}\n")

    A("## 3. Missing values\n")
    mi = f["missing"]
    if not mi["any_missing"]:
        A("No missing values.\n")
    else:
        A("| Column | % missing | Verdict |")
        A("|---|---|---|")
        cat_cols = set(ov["column_kinds"]["categorical"])
        for c in mi["columns_with_missing"]:
            if c.get("informative"):
                verdict = "impute + indicator (informative)"
            elif c.get("high_missing"):
                if c["column"] in cat_cols:
                    verdict = "keep as 'Missing' level + indicator (>50%)"
                else:
                    verdict = "drop (>50%, keep indicator if presence predictive)"
            else:
                verdict = "simple impute"
            A(f"| {c['column']} | {c['fraction']*100:.1f}% | {verdict} |")
        A(f"\n![missing]({_rel(plots_dir,'missing_values.png')})\n")
        A("*Key question: is missingness random or informative? Informative columns keep "
          "a `_is_missing` indicator so the signal isn't erased by imputation.*\n")

    A("## 4. Numerical features\n")
    nu = f["numerical"]
    A(f"Scale ratio across features: **{nu['scale_ratio']}** "
      f"({'very different → StandardScaler for scale-sensitive models' if nu['very_different_scales'] else 'comparable'}).")
    A("| Feature | Skew | Outliers (IQR) | Suggested |")
    A("|---|---|---|---|")
    for x in nu["features"]:
        sugg = []
        if abs(x["skew"]) > 1:
            sugg.append("log/Yeo-Johnson")
        if x["outliers_iqr"] > 0:
            sugg.append("RobustScaler/clip")
        A(f"| {x['column']} | {x['skew']} | {x['outliers_iqr']} | {', '.join(sugg) or '—'} |")
    A(f"\n![hist]({_rel(plots_dir,'numerical_histograms.png')})")
    A(f"![box]({_rel(plots_dir,'numerical_boxplots.png')})\n")

    A("## 5. Categorical features\n")
    ca = f["categorical"]
    if ca["features"]:
        A("| Feature | Cardinality | Rare cats | Encoding |")
        A("|---|---|---|---|")
        for x in ca["features"]:
            A(f"| {x['column']} | {x['cardinality']} | {len(x['rare_categories'])} | {x['suggested_encoding']} |")
        A(f"\n![cat]({_rel(plots_dir,'categorical_counts.png')})\n")
        A("*Categoricals are not numbers — label-encoding nominal values invents a false "
          "order. Use one-hot (low cardinality) or target/frequency encoding (high). "
          "Group rare categories into 'Other'.*\n")
    else:
        A("No categorical features.\n")

    if "relationships" in f and "error" not in f["relationships"]:
        rel = f["relationships"]
        A("## 6. Feature ↔ target relationships\n")
        A("Numeric features ranked by how well they separate the target:")
        for x in rel["numeric_vs_target"][:8]:
            if "separation" in x:
                A(f"- {x['column']}: class-median separation {x['separation']} std")
            else:
                A(f"- {x['column']}: target corr {x['target_corr']}")
        if os.path.exists(os.path.join(plots_dir, "numeric_by_target.png")):
            A(f"\n![num_target]({_rel(plots_dir,'numeric_by_target.png')})")
        if os.path.exists(os.path.join(plots_dir, "categorical_vs_target.png")):
            A(f"![cat_target]({_rel(plots_dir,'categorical_vs_target.png')})")
        A("\n*Heavy overlap doesn't mean useless — a weak feature can still help combined "
          "with others.*\n")

    A("## 7. Correlation & redundancy\n")
    co = f["correlation"]
    if "error" in co:
        A(co["error"] + "\n")
    else:
        if co["strong_pairs"]:
            A("Strongly correlated pairs (candidates to drop/combine, esp. for linear models):")
            for p in co["strong_pairs"]:
                A(f"- {p['a']} ~ {p['b']}: r = {p['corr']}")
        else:
            A("No strongly correlated pairs (|r| ≥ 0.8).")
        A(f"\n![corr]({_rel(plots_dir,'correlation_heatmap.png')})\n")
        A("*Correlation is linear co-movement, not causation.*\n")

    if "train_test" in f:
        tt = f["train_test"]
        A("## 8. Train/test consistency\n")
        for s in tt["numeric_shift"][:8]:
            A(f"- {s['column']}: PSI {s['psi']} ({s['shift']} shift)")
        if tt["unseen_categories"]:
            A(f"- Categories in test but not train: {tt['unseen_categories']}")
        if tt["any_large_shift"]:
            A("\n**Large shift → rely on robust CV; a single validation score may mislead.**")
        A(f"\n![shift]({_rel(plots_dir,'train_test_shift.png')})\n")

    A("## Cleaning & transformation plan\n")
    plan = f["transform_plan"]
    A("Ordered, leak-free (fit on train only, then apply to val/test):\n")
    if plan["drop"]:
        A(f"1. **Drop:** {sorted(set(plan['drop']))}")
    if plan["missing_indicator"]:
        A(f"2. **Add `_is_missing` indicators:** {plan['missing_indicator']}")
    if plan["numeric_impute"]:
        A(f"3. **Impute numeric (median):** {plan['numeric_impute']}")
    if plan["numeric_log"]:
        A(f"4. **Log/Yeo-Johnson:** {plan['numeric_log']}")
    if plan["scale"]:
        A("5. **Scale** numeric (RobustScaler default; StandardScaler if no strong outliers).")
    if plan["onehot"]:
        A(f"6. **One-hot encode:** {plan['onehot']}")
    if plan["target_encode"]:
        A(f"7. **Target/frequency encode (high cardinality):** {plan['target_encode']}")
    A("\nBuild these as an sklearn `ColumnTransformer` (see `transform.py`). Validate each "
      "change by checking whether a held-out metric improves.\n")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run full EDA and produce report.md + findings.json")
    ap.add_argument("data")
    ap.add_argument("--target", default=None)
    ap.add_argument("--test", default=None, help="optional separate test set for shift check")
    ap.add_argument("--out", default="eda_out")
    args = ap.parse_args()
    run(args.data, target=args.target, test=args.test, out=args.out)
    print(f"Done. See {args.out}/report.md, {args.out}/findings.json, {args.out}/plots/")


if __name__ == "__main__":
    main()
