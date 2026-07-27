import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")
TARGET = "click"

NUMERIC_FEATURES = ["price"]
CATEGORICAL_FEATURES = [
    "adgroup_id","cate_id","cms_group_id","final_gender_code","age_level",
    "pvalue_level","shopping_level","occupation","new_user_class_level",
    "pid","hour","weekday",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
])
model.fit(train[FEATURES], train[TARGET])

y_score = model.predict_proba(test[FEATURES])[:, 1]
print(f"BASELINE  AUC={roc_auc_score(test[TARGET], y_score):.4f}  "
      f"LogLoss={log_loss(test[TARGET], y_score):.4f}")



    # ===== TIEN XU LY =====
for d in (train, test):
    d["ts"]  = pd.to_datetime(d["time_stamp"])
    d["day"] = d["ts"].dt.floor("D")
    d["price_clip"] = d["price"].clip(upper=d["price"].quantile(0.999))
    d["price_log"]  = np.log1p(d["price_clip"])
    d["revenue"]    = d["price_clip"] * d[TARGET]

full = pd.concat([train.assign(_s="tr"), test.assign(_s="te")], ignore_index=True)

# ===== BUILD DOMAIN + TIME SERIES (causal) =====
def build_domain_ts(full, key, prefix):
    PRIOR = full[TARGET].mean()
    daily = (full.groupby([key, "day"])
                  .agg(clicks=(TARGET,"sum"), impr=(TARGET,"size"),
                       rev=("revenue","sum"), users=("userid","nunique"),
                       avgp=("price_clip","mean"))
                  .reset_index())
    daily["ctr"]  = (daily["clicks"] + 20*PRIOR) / (daily["impr"] + 20)
    daily["rpm"]  = daily["rev"]  / (daily["impr"]   + 1)
    daily["rpc"]  = daily["rev"]  / (daily["clicks"] + 1)
    daily["freq"] = daily["impr"] / (daily["users"]  + 1)
    daily = daily.sort_values([key, "day"])

    METRICS = ["clicks","impr","rev","users","avgp","ctr","rpm","rpc","freq"]
    for L in (1,2,3):
        for m in METRICS:
            daily[f"{prefix}_{m}_lag{L}"] = daily.groupby(key)[m].shift(L)
    for m in ("ctr","rpm","rpc","freq"):
        daily[f"{prefix}_{m}_d12"] = daily[f"{prefix}_{m}_lag1"] - daily[f"{prefix}_{m}_lag2"]
        daily[f"{prefix}_{m}_d13"] = daily[f"{prefix}_{m}_lag1"] - daily[f"{prefix}_{m}_lag3"]
    for m in ("impr","clicks","rev","users"):
        daily[f"{prefix}_{m}_r12"] = daily[f"{prefix}_{m}_lag1"] / (daily[f"{prefix}_{m}_lag2"] + 1)

    feat = [c for c in daily.columns if c.startswith(prefix+"_")
            and ("_lag" in c or "_d12" in c or "_d13" in c or "_r12" in c)]
    return daily[[key,"day"]+feat], feat, PRIOR

dom_adg, feat_adg, PRIOR = build_domain_ts(full, "adgroup_id", "adg")
dom_cat, feat_cat, _     = build_domain_ts(full, "cate_id",    "cate")
full = full.merge(dom_adg, on=["adgroup_id","day"], how="left")
full = full.merge(dom_cat, on=["cate_id","day"],   how="left")

ALL_TS = feat_adg + feat_cat

# ===== XU LY MISSING (ngay dau chua co lich su) =====
for c in ALL_TS:
    if c.endswith(("_d12","_d13")):   full[c] = full[c].fillna(0.0)
    elif c.endswith("_r12"):          full[c] = full[c].fillna(1.0)
    elif "_ctr_" in c:                full[c] = full[c].fillna(PRIOR)
    else:                             full[c] = full[c].fillna(0.0)

TR = full[full._s=="tr"].copy()
TE = full[full._s=="te"].copy()
print("ALL_TS:", len(ALL_TS), "features  |  TR:", TR.shape, "TE:", TE.shape)