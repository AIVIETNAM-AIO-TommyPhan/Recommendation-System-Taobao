"""Recompute every hard number claimed in README.md and 'Data Foundation.md'.

The generic EDA phases don't emit these dataset-specific claims (functional
dependencies, shared attribute triples, per-pid CTR, user overlap), so they are
checked here explicitly. Run after any dataset refresh so the docs can be
reconciled against the data they describe.

Usage:
    python verify_docs.py --train ../dataset/train.csv --test ../dataset/test.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

ID_COLS = ["adgroup_id", "cate_id", "campaign_id", "customer", "brand", "pid"]
ZERO_CHECK = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
    "brand",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time_stamp"])
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def basics(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] rows={len(df):,} cols={df.shape[1]}")
    print(f"  columns: {list(df.columns)}")
    print(f"  period : {df.time_stamp.min()} -> {df.time_stamp.max()}")
    print(f"  CTR    : {df.click.mean():.4%}  (clicks={int(df.click.sum()):,})")
    print(f"  users  : {df.userid.nunique():,} unique")
    na = df.isna().sum()
    print(f"  NaN    : {na[na > 0].to_dict() or 'none'}")
    dup = df.duplicated().sum()
    print(f"  exact duplicate rows: {dup:,}")


def cardinalities(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] cardinalities")
    for c in ID_COLS:
        vals = df[c].dropna().unique()
        print(f"  {c:<13} {len(vals):>6}"
              + (f"  values={sorted(vals)}" if len(vals) <= 3 else ""))
    print(f"  price         min={df.price.min()} max={df.price.max()} "
          f"mean={df.price.mean():.2f} median={df.price.median()}")


def zero_rates(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] share of rows where value == 0 (missing-sentinel check)")
    for c in ZERO_CHECK:
        rate = (df[c] == 0).mean()
        print(f"  {c:<22} {rate:8.4%}")


def hierarchy(df: pd.DataFrame) -> None:
    section("AD HIERARCHY / FUNCTIONAL DEPENDENCIES (train)")
    for child, parents in [("adgroup_id", ["cate_id", "brand", "price"]),
                           ("adgroup_id", ["campaign_id", "customer"]),
                           ("campaign_id", ["customer"])]:
        for p in parents:
            n = df.groupby(child)[p].nunique(dropna=False)
            bad = n[n > 1]
            status = "OK 1:1" if bad.empty else f"VIOLATED ({len(bad)} {child}s map to >1 {p})"
            print(f"  {child} -> {p:<12} {status}")

    triples = df.groupby("adgroup_id")[["cate_id", "brand", "price"]].first()
    n_ag = len(triples)
    n_tri = len(triples.drop_duplicates())
    dupes = triples[triples.duplicated(keep=False)]
    shared = dupes.groupby(list(triples.columns), dropna=False).size()
    print(f"\n  {n_ag} ad groups -> {n_tri} distinct (cate_id, brand, price) triples")
    print(f"  {len(shared)} triples shared by more than one ad group "
          f"({len(dupes)} ad groups involved)")

    cust = df.groupby("adgroup_id")["customer"].first()
    print("\n  examples of shared triples:")
    for key, _ in list(shared.items())[:6]:
        ags = triples[(triples.cate_id == key[0])
                      & (triples.brand.fillna(-1) == (key[1] if pd.notna(key[1]) else -1))
                      & (triples.price == key[2])].index.tolist()
        customers = [int(cust[a]) for a in ags]
        same = "same customer" if len(set(customers)) == 1 else "different customers"
        print(f"    cate={key[0]} brand={key[1]} price={key[2]} -> "
              f"adgroups {ags} customers {customers} ({same})")


def pid_table(df: pd.DataFrame) -> None:
    section("PID (ad slot) IMPRESSIONS & CTR (train)")
    g = df.groupby("pid").agg(impressions=("click", "size"), ctr=("click", "mean"))
    for pid, row in g.iterrows():
        print(f"  {pid:<13} impressions={int(row.impressions):>8,}  CTR={row.ctr:.2%}")


def split_checks(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    section("TRAIN / TEST SPLIT")
    print(f"  train max time_stamp : {tr.time_stamp.max()}")
    print(f"  test  min time_stamp : {te.time_stamp.min()}")
    print(f"  strictly chronological: {tr.time_stamp.max() < te.time_stamp.min()}")
    print(f"  split ratio          : {len(tr) / (len(tr) + len(te)):.2%} train")

    tu, eu = set(tr.userid), set(te.userid)
    print(f"\n  test users           : {len(eu):,}")
    print(f"  test users seen in train: {len(tu & eu):,} "
          f"({len(tu & eu) / len(eu):.1%})")
    for c in ID_COLS:
        unseen = set(te[c].dropna()) - set(tr[c].dropna())
        print(f"  unseen {c:<13} in test: {len(unseen)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="dataset/train.csv")
    ap.add_argument("--test", default="dataset/test.csv")
    args = ap.parse_args()

    tr, te = load(args.train), load(args.test)

    section("BASICS")
    basics("train", tr)
    basics("test", te)

    section("CARDINALITIES")
    cardinalities("train", tr)
    cardinalities("test", te)

    section("ZERO-SENTINEL RATES")
    zero_rates("train", tr)
    zero_rates("test", te)

    hierarchy(tr)
    pid_table(tr)
    split_checks(tr, te)


if __name__ == "__main__":
    main()
