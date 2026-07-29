"""Gold layer: leak-safe feature engineering -> training-ready table.

Ports the feature engineering validated in Full_Training_Model_v5.ipynb
into reusable functions: target encoding (with raw-id drop), domain
(historical CTR) features, and user-based/item-based CF features. Every
"before" feature only ever uses data strictly earlier than its own row's
timestamp - the point-in-time-correctness rule this whole project is
built around.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from medallion.common.config import GOLD_DIR, ID_COLS, RANDOM_SEED, SILVER_DIR, SMOOTHING, TARGET


def load_silver():
    parts = sorted((SILVER_DIR / "fact_events").glob("dt=*/part.parquet"))
    fact = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dim_ad = pd.read_parquet(SILVER_DIR / "dim_ad.parquet")
    dim_user = pd.read_parquet(SILVER_DIR / "dim_user.parquet")
    df = fact.merge(dim_ad, on="adgroup_id", how="left").merge(dim_user, on="userid", how="left")
    return df.sort_values("time_stamp").reset_index(drop=True)


def kfold_target_encode(train_df, test_df, col, target, global_mean, n_splits=5, smoothing=SMOOTHING, seed=RANDOM_SEED):
    oof = pd.Series(index=train_df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in kf.split(train_df):
        fit_stats = train_df.iloc[fit_idx].groupby(col)[target].agg(["mean", "count"])
        smoothed = (fit_stats["mean"] * fit_stats["count"] + global_mean * smoothing) / (fit_stats["count"] + smoothing)
        oof.iloc[hold_idx] = train_df.iloc[hold_idx][col].map(smoothed).fillna(global_mean).values
    full_stats = train_df.groupby(col)[target].agg(["mean", "count"])
    full_smoothed = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
    test_encoded = test_df[col].map(full_smoothed).fillna(global_mean)
    return oof.values, test_encoded.values


def add_target_encodings(combined, train_mask, global_mean):
    """Stash userid/adgroup_id as plain key columns (used by every later
    step and by the Gold snapshots), then target-encode and drop the raw,
    high-cardinality id columns so no model ever splits on the id itself."""
    combined["userid_key"] = combined["userid"]
    combined["adgroup_id_key"] = combined["adgroup_id"]

    for col in ID_COLS:
        te_train, te_test = kfold_target_encode(combined[train_mask], combined[~train_mask], col, TARGET, global_mean)
        combined.loc[train_mask, f"{col}_te"] = te_train
        combined.loc[~train_mask, f"{col}_te"] = te_test

    return combined.drop(columns=ID_COLS)


def add_domain_features(combined, global_mean):
    combined["_rev"] = combined["price"] * combined["click"]
    for col, prefix in [("adgroup_id_key", "ag"), ("cate_id", "cate"), ("userid_key", "user")]:
        combined[f"{prefix}_clicks_before"] = combined.groupby(col)["click"].cumsum() - combined["click"]
        combined[f"{prefix}_impressions_before"] = combined.groupby(col).cumcount()
        combined[f"{prefix}_ctr_before"] = (
            combined[f"{prefix}_clicks_before"] / combined[f"{prefix}_impressions_before"]
        ).replace([np.inf, -np.inf], np.nan).fillna(global_mean)
        if prefix == "ag":
            combined[f"{prefix}_revenue_before"] = combined.groupby(col)["_rev"].cumsum() - combined["_rev"]

    combined_ti = combined.set_index("time_stamp")

    def rolling_before(g, window):
        cnt = g["click"].rolling(window).count() - 1
        clk = g["click"].rolling(window).sum() - g["click"]
        return pd.DataFrame({"cnt": cnt, "clk": clk}, index=g.index)

    for window, suffix in [("1D", "1d"), ("3D", "3d")]:
        res = combined_ti.groupby("adgroup_id_key", group_keys=False).apply(
            lambda g, w=window: rolling_before(g, w), include_groups=False
        ).reset_index(drop=True)
        combined[f"ag_impressions_last_{suffix}"] = res["cnt"].values
        combined[f"ag_ctr_last_{suffix}"] = (
            (res["clk"] / res["cnt"]).replace([np.inf, -np.inf], np.nan).fillna(global_mean).values
        )
    combined["ag_ctr_trend_1d_vs_3d"] = combined["ag_ctr_last_1d"] - combined["ag_ctr_last_3d"]
    return combined.drop(columns=["_rev"])


def add_item_coclick_affinity(combined, train_end_idx):
    """Sequential co-click affinity: P(click this item | last click was item X),
    built as a single causal pass - a row's feature only reflects clicks that
    happened strictly before it; the running counters update *after* scoring."""
    combined["_clicked_item"] = np.where(combined["click"] == 1, combined["adgroup_id_key"], np.nan)
    ffilled = combined.groupby("userid_key")["_clicked_item"].ffill()
    combined["user_prev_clicked_item"] = ffilled.groupby(combined["userid_key"]).shift(1)

    # plain dicts + .get(), NOT defaultdict - a defaultdict auto-creates a zero-valued
    # entry on every *read* (the scoring lookup below runs on every row, not just
    # clicks), which would silently pollute the persisted snapshot with thousands of
    # phantom zero entries that were never a real co-click count.
    coclick_counts = {}
    prev_item_totals = {}
    scores = np.zeros(len(combined))
    prev_items = combined["user_prev_clicked_item"].values
    cur_items = combined["adgroup_id_key"].values
    clicks = combined["click"].values

    snapshot = None
    for i in range(len(combined)):
        prev = prev_items[i]
        if not pd.isna(prev):
            total = prev_item_totals.get(prev, 0)
            if total > 0:
                scores[i] = coclick_counts.get((prev, cur_items[i]), 0) / total
        if clicks[i] == 1 and not pd.isna(prev):
            key = (prev, cur_items[i])
            coclick_counts[key] = coclick_counts.get(key, 0) + 1
            prev_item_totals[prev] = prev_item_totals.get(prev, 0) + 1
        if i == train_end_idx - 1:
            # frozen "as of end of training window" snapshot, used later for scoring
            snapshot = {"coclick_counts": dict(coclick_counts), "prev_item_totals": dict(prev_item_totals)}

    combined["item_coclick_affinity"] = scores
    return combined.drop(columns=["_clicked_item"]), snapshot


def add_user_affinity_features(combined, global_mean):
    for col, prefix in [("cate_id", "user_cate"), ("brand", "user_brand")]:
        grp = combined.groupby(["userid_key", col])
        combined[f"{prefix}_clicks_before"] = grp["click"].cumsum() - combined["click"]
        combined[f"{prefix}_impressions_before"] = grp.cumcount()
        combined[f"{prefix}_affinity_before"] = (
            (combined[f"{prefix}_clicks_before"] + global_mean * SMOOTHING)
            / (combined[f"{prefix}_impressions_before"] + SMOOTHING)
        )

    combined["_clicked_price"] = np.where(combined["click"] == 1, combined["price"], np.nan)
    expanding_avg = combined.groupby("userid_key")["_clicked_price"].expanding().mean().reset_index(level=0, drop=True)
    prev_avg = expanding_avg.groupby(combined["userid_key"]).shift(1).reindex(combined.index)
    combined["user_avg_clicked_price_before"] = prev_avg.fillna(combined["price"].mean())
    combined["user_price_gap"] = combined["price"] - combined["user_avg_clicked_price_before"]
    return combined.drop(columns=["_clicked_price"])


def add_interactions(combined, train_mask):
    for a, b in [("pid", "cate_id"), ("cate_id", "age_level"), ("brand", "cms_group_id")]:
        key = f"{a}_x_{b}"
        joint_key = combined[a].astype(str) + "_" + combined[b].astype(str)
        counts = joint_key[train_mask].value_counts()
        freq = counts / train_mask.sum()
        combined[f"{key}_freq"] = joint_key.map(freq).fillna(0).values
    return combined


CATEGORICAL_FEATURES = [
    "cate_id", "brand", "pid", "cms_segid", "cms_group_id", "final_gender_code",
    "age_level", "pvalue_level", "shopping_level", "occupation",
    "new_user_class_level", "hour", "weekday",
]
NUMERIC_FEATURES = ["price"] + [f"{c}_te" for c in ID_COLS] + [
    "ag_clicks_before", "ag_impressions_before", "ag_ctr_before", "ag_revenue_before",
    "ag_impressions_last_1d", "ag_ctr_last_1d", "ag_impressions_last_3d", "ag_ctr_last_3d", "ag_ctr_trend_1d_vs_3d",
    "cate_clicks_before", "cate_impressions_before", "cate_ctr_before",
    "user_clicks_before", "user_impressions_before", "user_ctr_before",
    "pid_x_cate_id_freq", "cate_id_x_age_level_freq", "brand_x_cms_group_id_freq",
    "item_coclick_affinity", "user_cate_affinity_before", "user_brand_affinity_before", "user_price_gap",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def run():
    t0 = time.time()
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_silver()
    train_mask = combined["_split"] == "train"
    global_mean = combined.loc[train_mask, TARGET].mean()
    train_end_idx = int(train_mask.sum())

    combined = add_target_encodings(combined, train_mask, global_mean)
    combined = add_domain_features(combined, global_mean)
    combined, coclick_snapshot = add_item_coclick_affinity(combined, train_end_idx)
    combined = add_user_affinity_features(combined, global_mean)
    combined = add_interactions(combined, train_mask)

    for c in CATEGORICAL_FEATURES:
        combined[c] = combined[c].fillna(-1).astype("Int64").astype(str) if combined[c].dtype != object else combined[c].fillna("-1")

    out_cols = ["userid_key", "adgroup_id_key", "user_prev_clicked_item", "time_stamp", "_split", TARGET] + FEATURES
    combined[out_cols].to_parquet(GOLD_DIR / "training_features.parquet", index=False)
    with open(GOLD_DIR / "item_coclick_snapshot.pkl", "wb") as f:
        pickle.dump(coclick_snapshot, f)

    print("gold/training_features.parquet:", combined[out_cols].shape, f"{time.time() - t0:.1f}s")
    print("gold/item_coclick_snapshot.pkl:", len(coclick_snapshot["coclick_counts"]), "item pairs")
    return GOLD_DIR


if __name__ == "__main__":
    run()
