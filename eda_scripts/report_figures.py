"""Purpose-built figures for EDA_REPORT.md.

Unlike the generic phase plots in ``eda_out/plots/``, each figure here illustrates
one specific claim in the written report, so a reader can see the finding instead of
parsing a table. Output goes to ``eda_out/report_figures/``.

Palette: dataviz categorical slots 1-4 (blue/orange/aqua/yellow), validated
colorblind-safe; every bar carries a direct value label (the relief rule, since the
fills sit below 3:1 contrast on white).

Usage:
    python report_figures.py --train ../dataset/train.csv --test ../dataset/test.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

SENTINEL = ["cms_segid", "pvalue_level", "brand", "new_user_class_level"]
SIGNAL_COLS = [
    "adgroup_id", "campaign_id", "customer", "brand", "cate_id", "cms_group_id",
    "age_level", "cms_segid", "pid", "hour", "occupation", "weekday",
    "new_user_class_level", "pvalue_level", "final_gender_code", "shopping_level",
]
AD_SIDE = {"adgroup_id", "campaign_id", "customer", "brand", "cate_id"}


def style_ax(ax):
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_column_roles(tr, out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})

    # Panel A: adgroup_id treated as a magnitude — IQR fence flags "outliers"
    # that are really just ads with a low/high ID number, unrelated to CTR.
    g = tr.groupby("adgroup_id").click.agg(["size", "mean"])
    q1, q3 = tr.adgroup_id.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = (g.index >= lo) & (g.index <= hi)

    a0.axvspan(g.index.min() - 5000, lo, color=GRID, zorder=0)
    a0.axvspan(hi, g.index.max() + 5000, color=GRID, zorder=0)
    a0.scatter(g.index[inside], g["mean"][inside] * 100, s=14, color=BLUE,
               zorder=3, label="inside fence")
    a0.scatter(g.index[~inside], g["mean"][~inside] * 100, s=14, color=ORANGE,
               zorder=3, label="\"outlier\" by ID magnitude")
    for v, name in [(lo, "lower fence"), (hi, "upper fence")]:
        a0.axvline(v, color=MUTED, ls="--", lw=1, zorder=2)
    a0.set_xlim(g.index.min() - 5000, g.index.max() + 5000)
    a0.set_ylim(0, 50)
    a0.set_xlabel("adgroup_id (the number itself)")
    a0.set_ylabel("this ad's CTR, %")
    a0.set_title("adgroup_id: \"6,728 IQR outliers\" — same CTR spread\n"
                  "inside the fence (blue) and outside it (orange)", fontsize=10.5, loc="left")
    a0.legend(frameon=False, fontsize=8.5, loc="upper right")
    style_ax(a0)
    a0.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    # Panel B: cms_segid -> cms_group_id is a deterministic step function,
    # not a linear relationship — correlation is the wrong instrument for it.
    sub = tr[tr.cms_segid != 0]
    pairs = sub[["cms_segid", "cms_group_id"]].drop_duplicates().sort_values("cms_segid")
    a1.scatter(pairs.cms_segid, pairs.cms_group_id, s=22, color=BLUE, zorder=3)
    a1.set_xlabel("cms_segid (96 non-zero levels)")
    a1.set_ylabel("cms_group_id (13 buckets)")
    a1.set_title("cms_segid → cms_group_id: every level maps\nto exactly one bucket (r = 0.453 all rows,\n"
                  "0.984 only after dropping the 0-sentinel)", fontsize=10.5, loc="left")
    style_ax(a1)
    a1.grid(axis="y", color=GRID, lw=0.8, zorder=0)

    fig.suptitle("§1  Integer dtype ≠ quantity: IDs and codes aren't numbers to do arithmetic on",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(out, "fig_column_roles.png"), bbox_inches="tight")
    plt.close(fig)


def fig_target_balance(tr, te, out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})

    n1 = int(tr.click.sum())
    n0 = len(tr) - n1
    a0.barh(["clicks\n(1)", "non-clicks\n(0)"], [n1, n0], color=[ORANGE, BLUE], zorder=3, height=0.6)
    for y, v in zip([0, 1], [n1, n0]):
        a0.text(v - max(n0, n1) * 0.01, y, f"{v:,}", va="center", ha="right",
                color="white", fontweight="bold")
    a0.set_title(f"Class balance — {n0:,} : {n1:,}  ≈  3.95 : 1", fontsize=11, loc="left")
    a0.set_xlim(0, n0 * 1.05)
    style_ax(a0)

    ctrs = [tr.click.mean() * 100, te.click.mean() * 100]
    bars = a1.bar(["train", "test"], ctrs, color=[BLUE, AQUA], zorder=3, width=0.5)
    for b, v in zip(bars, ctrs):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center",
                fontweight="bold")
    a1.set_ylim(0, 25)
    a1.set_title("CTR (positive rate)", fontsize=11, loc="left")
    a1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    a1.set_axisbelow(True)
    a1.tick_params(length=0)

    fig.suptitle("§2  Target: mildly imbalanced, ~20% CTR", fontsize=12,
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out, "fig_target_balance.png"), bbox_inches="tight")
    plt.close(fig)


def fig_missing(tr, out):
    rates = [(c, (tr[c] == 0).mean() * 100) for c in SENTINEL]
    occ = ("occupation", (tr.occupation == 0).mean() * 100)
    rates_sorted = sorted(rates, key=lambda x: x[1])
    labels = [c for c, _ in rates_sorted] + ["occupation"]
    vals = [v for _, v in rates_sorted] + [occ[1]]
    colors = [ORANGE] * len(rates_sorted) + [BLUE]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(labels, vals, color=colors, zorder=3, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center",
                color=MUTED, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of rows where value == 0")
    ax.set_title("§3  The 0-sentinel: 4 columns encode 'unknown' as 0 (orange),\n"
                 "occupation's 0 = 'not a student' is a real level (blue)",
                 fontsize=11, loc="left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_missing_sentinels.png"), bbox_inches="tight")
    plt.close(fig)


def fig_price(tr, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    a0, a1, a2 = axes

    a0.hist(tr.price, bins=50, color=BLUE, zorder=3)
    a0.set_title(f"price (raw) — skew {tr.price.skew():.2f}", fontsize=11, loc="left")
    a0.set_xlabel("¥")

    lp = np.log1p(tr.price)
    a1.hist(lp, bins=50, color=AQUA, zorder=3)
    a1.set_title(f"log1p(price) — skew {lp.skew():.2f}", fontsize=11, loc="left")
    a1.set_xlabel("log1p ¥")

    dec = pd.qcut(tr.price, 10, duplicates="drop")
    ctr = tr.groupby(dec, observed=True).click.mean() * 100
    x = range(len(ctr))
    a2.plot(x, ctr.values, color=ORANGE, lw=2, marker="o", ms=6, zorder=3)
    a2.axhline(tr.click.mean() * 100, color=MUTED, ls="--", lw=1, zorder=2)
    a2.text(len(ctr) - 1, tr.click.mean() * 100 + 0.4, "base 20.19%", ha="right",
            color=MUTED, fontsize=9)
    a2.set_ylim(0, 30)
    a2.set_xlabel("price decile (cheap → expensive)")
    a2.set_title("CTR by price decile — flat, no trend", fontsize=11, loc="left")

    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(length=0)

    fig.suptitle("§4  price: right-skewed, log1p fixes it — but it barely moves CTR",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out, "fig_price.png"), bbox_inches="tight")
    plt.close(fig)


def fig_feature_signal(tr, out):
    base = tr.click.mean()
    rows = []
    for c in SIGNAL_COLS:
        g = tr.groupby(c).click.agg(["size", "mean"])
        g = g[g["size"] >= 200]
        w = g["size"] / g["size"].sum()
        sd = np.sqrt((w * (g["mean"] - base) ** 2).sum())
        rows.append((c, sd))
    rows.sort(key=lambda x: x[1])
    labels = [c for c, _ in rows]
    vals = [v for _, v in rows]
    colors = [ORANGE if c in AD_SIDE else BLUE for c, _ in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(labels, vals, color=colors, zorder=3, height=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.0008, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center",
                color=MUTED, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("impression-weighted SD of per-level CTR around 20.19% base")
    ax.set_title("§5  Ad identity carries ~10× the signal of any user attribute\n"
                 "orange = ad-side feature · blue = user / context feature",
                 fontsize=11, loc="left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_feature_signal.png"), bbox_inches="tight")
    plt.close(fig)


def fig_shift(tr, te, out):
    def tvd(col):
        a = tr[col].value_counts(normalize=True)
        b = te[col].value_counts(normalize=True).reindex(a.index).fillna(0)
        return 0.5 * (a - b).abs().sum()

    feats = ["adgroup_id", "cate_id", "age_level", "pid", "final_gender_code"]
    tvds = [(f, tvd(f)) for f in feats]
    tvds.sort(key=lambda x: x[1])

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})

    labels = [f for f, _ in tvds]
    vals = [v for _, v in tvds]
    colors = [ORANGE if v >= 0.05 else BLUE for _, v in tvds]
    bars = a0.barh(labels, vals, color=colors, zorder=3, height=0.6)
    for b, v in zip(bars, vals):
        a0.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                color=MUTED, fontsize=9)
    a0.set_xlim(0, max(vals) * 1.2)
    a0.set_xlabel("total variation distance, train vs test")
    a0.set_title("Distribution shift by feature", fontsize=11, loc="left")
    style_ax(a0)

    ptr = tr.adgroup_id.value_counts(normalize=True)
    pte = te.adgroup_id.value_counts(normalize=True)
    top = (ptr.head(6).index.union(pte.head(6).index))
    diff = (pte.reindex(top).fillna(0) - ptr.reindex(top).fillna(0)).sort_values()
    y = np.arange(len(diff))
    a1.barh(y, ptr.reindex(diff.index).fillna(0) * 100, color=BLUE, height=0.38,
            label="train", zorder=3)
    a1.barh(y + 0.42, pte.reindex(diff.index).fillna(0) * 100, color=ORANGE, height=0.38,
            label="test", zorder=3)
    a1.set_yticks(y + 0.21)
    a1.set_yticklabels([str(i) for i in diff.index], fontsize=8)
    a1.set_xlabel("% of impressions")
    a1.set_title("Which ads dominate changes (top adgroup_ids)", fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    a1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    a1.set_axisbelow(True)
    a1.tick_params(length=0)

    fig.suptitle("§8  Catalog is fixed, but the ad mix shifts (adgroup TVD 0.28)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(out, "fig_train_test_shift.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="dataset/train.csv")
    ap.add_argument("--test", default="dataset/test.csv")
    ap.add_argument("--out", default="eda_out/report_figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)

    fig_column_roles(tr, args.out)
    fig_target_balance(tr, te, args.out)
    fig_missing(tr, args.out)
    fig_price(tr, args.out)
    fig_feature_signal(tr, args.out)
    fig_shift(tr, te, args.out)
    print(f"6 figures written to {args.out}")


if __name__ == "__main__":
    main()
