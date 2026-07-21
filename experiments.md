# Experiment Log

Personal tracking table per team lead's request. Update after every run.

| STT | Tên exp | Thay đổi gì (feature / model / data setup) | AUC | GAUC | GINI (2·AUC−1) | Precision@10 | Recall@10 | Recall@50 | Recall@100 |
|-----|---------|---------------------------------------------|-----|------|-----------------|--------------|-----------|-----------|------------|
| 1 | Baseline (LogReg) | Features: price, adgroup_id, cate_id, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level, pid, hour, weekday. Model: LogisticRegression. Data: top-300 product catalog subsample (train CTR 20.2%, test CTR 19.3% — not the real ~5% imbalance). | 0.5759 | 0.5543 | 0.1518 | 0.0106 | 0.1065 | 0.3439 | 0.5391 |
| 2 | + campaign_id/customer/brand | Baseline + 3 feature còn thiếu (campaign_id, customer, brand), cùng model LogReg, cùng data setup | 0.5755 | 0.5546 | 0.1510 | 0.0097 | 0.0966 | 0.3498 | 0.5380 |

## Nhận xét nhanh
- **GAUC (0.5543) < AUC (0.5543 vs 0.5759)**: model xếp hạng "ai hay click hơn ai" tốt hơn là xếp hạng "trong các lựa chọn của riêng 1 user, cái nào họ thích hơn"
- **Thêm campaign_id/customer/brand không giúp ích** — AUC giảm nhẹ (0.5759 → 0.5755), GAUC gần như không đổi, Recall@10 giảm (0.1065 → 0.0966).