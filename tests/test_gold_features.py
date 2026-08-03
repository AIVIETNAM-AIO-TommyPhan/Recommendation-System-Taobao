import numpy as np
import pandas as pd
import pytest

from medallion.gold import features as gold_features


# Tiêu chí: kfold_target_encode cho kết quả ổn định khi seed cố định (không phụ thuộc random state ngoài).
def test_kfold_target_encode_is_deterministic_for_fixed_seed():
    train = pd.DataFrame({"cate": ["a", "a", "b", "b", "a", "b"], "click": [1, 0, 0, 1, 1, 0]})
    test = pd.DataFrame({"cate": ["a", "b"]})

    oof1, enc1 = gold_features.kfold_target_encode(train, test, "cate", "click", global_mean=0.5, n_splits=3, seed=42)
    oof2, enc2 = gold_features.kfold_target_encode(train, test, "cate", "click", global_mean=0.5, n_splits=3, seed=42)

    assert np.array_equal(oof1, oof2)
    assert np.array_equal(enc1, enc2)


# Tiêu chí: Category chưa từng xuất hiện ở train được fallback về global_mean trên tập test.
def test_kfold_target_encode_falls_back_to_global_mean_for_unseen_category():
    train = pd.DataFrame({"cate": ["a", "a", "a"], "click": [1, 0, 1]})
    test = pd.DataFrame({"cate": ["never_seen"]})

    _, test_encoded = gold_features.kfold_target_encode(train, test, "cate", "click", global_mean=0.37, n_splits=2, seed=0)

    assert test_encoded[0] == pytest.approx(0.37)


# Tiêu chí: add_target_encodings bỏ hẳn cột id thô, chỉ giữ userid_key/adgroup_id_key và các cột *_te.
def test_add_target_encodings_drops_raw_id_columns_and_keeps_keys():
    combined = pd.DataFrame(
        {
            "userid": [1, 2, 3, 4, 5, 6, 7, 8],
            "adgroup_id": [10, 20, 10, 20, 10, 20, 10, 20],
            "campaign_id": [100, 200, 100, 200, 100, 200, 100, 200],
            "customer": [1000, 2000, 1000, 2000, 1000, 2000, 1000, 2000],
            "click": [1, 0, 0, 1, 1, 0, 1, 0],
        }
    )
    # default kfold_target_encode uses n_splits=5, so the train side needs >= 5 rows
    train_mask = pd.Series([True, True, True, True, True, False, False, False])

    out = gold_features.add_target_encodings(combined.copy(), train_mask, global_mean=0.5)

    assert not set(gold_features.ID_COLS) & set(out.columns)
    assert {"userid_key", "adgroup_id_key"} <= set(out.columns)
    assert {f"{c}_te" for c in gold_features.ID_COLS} <= set(out.columns)


# Tiêu chí: ag_clicks_before/impressions_before chỉ tính trên các dòng TRƯỚC dòng hiện tại (point-in-time correctness).
def test_add_domain_features_excludes_current_row_from_before_counts():
    combined = pd.DataFrame(
        {
            "adgroup_id_key": ["ag1", "ag1", "ag1"],
            "cate_id": ["c1", "c1", "c1"],
            "userid_key": ["u1", "u1", "u1"],
            "price": [10.0, 10.0, 10.0],
            "click": [1, 0, 1],
            "time_stamp": pd.to_datetime(
                ["2017-05-05 08:00:00", "2017-05-05 09:00:00", "2017-05-05 10:00:00"]
            ),
        }
    )

    out = gold_features.add_domain_features(combined, global_mean=0.4)

    assert out["ag_clicks_before"].tolist() == [0, 1, 1]
    assert out["ag_impressions_before"].tolist() == [0, 1, 2]
    # first row has 0 impressions-before -> ctr_before falls back to global_mean, not NaN/inf
    assert out.loc[0, "ag_ctr_before"] == pytest.approx(0.4)
    assert out.loc[1, "ag_ctr_before"] == pytest.approx(1.0)


# Tiêu chí: coclick_counts/prev_item_totals chỉ được cập nhật khi có click thật, các lượt scoring
# thuần túy (.get trên plain dict) không được phép "tạo" entry ma trong bộ đếm.
def test_add_item_coclick_affinity_ignores_pure_scoring_lookups():
    combined = pd.DataFrame(
        {
            "userid_key": ["u1"] * 5,
            "adgroup_id_key": ["A", "B", "C", "A", "B"],
            "click": [1, 1, 0, 1, 0],
        }
    )

    out, snapshot = gold_features.add_item_coclick_affinity(combined, train_end_idx=4)

    # row4 (item B, prev=A): A->B was seen once as a real click transition (row1) -> score 1.0
    assert out.loc[4, "item_coclick_affinity"] == pytest.approx(1.0)
    # row2 (item C, prev=B, pure scoring, click=0) must NOT have created a "B" entry
    assert snapshot["prev_item_totals"] == {"A": 1, "B": 1}
    assert snapshot["coclick_counts"] == {("A", "B"): 1, ("B", "A"): 1}


# Tiêu chí: user_cate_affinity_before dùng đúng công thức smoothing và chỉ tính trên lịch sử trước dòng hiện tại.
def test_add_user_affinity_features_applies_smoothing_formula():
    combined = pd.DataFrame(
        {
            "userid_key": ["u1", "u1", "u1"],
            "cate_id": ["c1", "c1", "c1"],
            "brand": ["b1", "b1", "b1"],
            "price": [10.0, 20.0, 30.0],
            "click": [1, 0, 1],
        }
    )
    smoothing = gold_features.SMOOTHING
    global_mean = 0.3

    out = gold_features.add_user_affinity_features(combined, global_mean)

    # row 2: clicks_before=1 (rows 0,1 -> clicks 1,0), impressions_before=2 (cumcount)
    expected_row2 = (1 + global_mean * smoothing) / (2 + smoothing)
    assert out.loc[2, "user_cate_affinity_before"] == pytest.approx(expected_row2)
    # first row has no prior clicked price -> falls back to overall price mean
    assert out.loc[0, "user_avg_clicked_price_before"] == pytest.approx(combined["price"].mean())


# Tiêu chí: Tần suất tương tác (pid x cate_id) chỉ học từ train, tổ hợp chỉ xuất hiện ở test phải bằng 0.
def test_add_interactions_computes_freq_from_train_only():
    combined = pd.DataFrame(
        {
            "pid": ["p1", "p1", "p1", "p2"],
            "cate_id": ["c1", "c1", "c2", "c1"],
            "age_level": [1, 1, 1, 1],
            "brand": ["b1", "b1", "b1", "b1"],
            "cms_group_id": [1, 1, 1, 1],
        }
    )
    train_mask = pd.Series([True, True, True, False])

    out = gold_features.add_interactions(combined, train_mask)

    assert out.loc[0, "pid_x_cate_id_freq"] == pytest.approx(2 / 3)
    assert out.loc[3, "pid_x_cate_id_freq"] == pytest.approx(0.0)
