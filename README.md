# Recommendation-System-Taobao
CTR prediction estimates the probability that a user clicks an advertisement, directly influencing ad ranking and revenue. The task is challenging due to massive-scale datasets, high-dimensional sparse features, and complex nonlinear interactions that cannot be effectively captured by traditional linear models such as Logistic Regression.

# About Dataset

`dataset/` holds a single flat table, already joined from the original Taobao
ad-display tables (`raw_sample` + `ad_feature` + `user_profile`) and split into
`train.csv` and `test.csv`. Produced by `Sample_Dataset.ipynb`, which restricts
the ad catalog to the **300 most-frequent products**, so `adgroup_id` is a
low-cardinality, learnable feature instead of a near-unique ID.

| file | rows | period | CTR |
|---|---|---|---|
| `train.csv` | 240,000 | 2017-05-05 16:00:03 → 2017-05-12 04:24:15 | 20.19% |
| `test.csv` | 60,000 | 2017-05-12 04:24:18 → 2017-05-13 15:59:26 | 19.26% |

The split is chronological (80/20) — every test impression happens strictly
after every training impression. Users may appear in both (20,508 of the
47,608 test users are also in train); ads always do, since the catalog is
fixed at 300 — test draws on 286 of them, so it has **zero** unseen levels for
any ad-side column. The `behavior_log` table from the original release is
**not** included. Both files share the same 21 columns and contain no `NaN`
(but see the `0`-sentinel note below — some fields encode "unknown" as `0`).

Note that the *mix* of ads shifts between the two windows even though the
catalog does not: total variation distance on `adgroup_id` is 0.28, and test
prices run higher (mean ¥230.7 / median ¥162 vs train's ¥204.8 / ¥138). See
`EDA_REPORT.md` §8.

## Impression fields (from `raw_sample`)
(1) userid: user ID (int);
(2) time_stamp: impression time, formatted as `YYYY-MM-DD HH:MM:SS` (the raw
    dataset stores this as a Unix timestamp);
(3) adgroup_id: ad group ID (int), 300 unique values;
(4) pid: ad slot / scenario, 2 values (`430539_1007`, `430548_1007`);
(5) click: **1 for click, 0 for no click** — note this is the inverse of the
    `noclk` field in the original release;

## Ad fields (from `ad_feature`)
(1) cate_id: category ID, 44 unique values;
(2) campaign_id: campaign ID, 248 unique values;
(3) customer: advertiser ID, 213 unique values (named `customer_id` upstream);
(4) brand: brand ID, 116 unique values — but one of those is the `0`
    sentinel, so there are **115 real brands** and 131 of the 300 ads are
    brandless;
(5) price: item price (float), 2.4 – 999.0, mean ≈ 204.8;

One ad ID corresponds to an item, an item belongs to a category, an item
belongs to a brand.

## User fields (from `user_profile`)
(1) cms_segid: micro group ID;
(2) cms_group_id: cms group ID, 0–12;
(3) final_gender_code: gender, 1 for male, 2 for female;
(4) age_level: age level, 0–6;
(5) pvalue_level: consumption grade, 1: low, 2: mid, 3: high;
(6) shopping_level: shopping depth, 1: shallow, 2: moderate, 3: deep;
(7) occupation: is the user a college student, 1: yes, 0: no;
(8) new_user_class_level: city level, 1–4;

## The `0` sentinel

Four columns use `0` to mean "unknown" rather than as a real level — the
values were missing in the original upstream tables and were filled with `0`
during preprocessing. Share of training rows affected:

| column | `0` share | why it's a sentinel |
|---|---|---|
| `cms_segid` | 55.6% | micro-group unknown |
| `pvalue_level` | 49.9% | grades are 1–3 |
| `brand` | 48.0% | 131 of the 300 ads carry no brand |
| `new_user_class_level` | 25.7% | levels are 1–4 |

Three columns look similar but are **not** sentinels: `occupation` (`0` = "not
a student", 92.4% of rows, a genuine level), `final_gender_code` (coded 1/2,
no `0` present), and `cms_group_id` / `age_level` (`0` occurs in 0.03% of rows
— too rare for a sentinel, consistent with a small real bucket).

Decode these to a distinct "unknown" category rather than imputing; the
missingness is **not** informative (every gap in CTR between missing and
present is under 0.3pp against a 20.19% base rate), so no missing-indicator is
needed. See `EDA_REPORT.md` §3.

## Derived fields
(1) date: calendar date of `time_stamp`;
(2) hour: hour of day, 0–23;
(3) weekday: day of week, 0–6;
