BASELINE (one-hot ad)  AUC=0.5759

+adg_clicks_lag1        AUC=0.5754  d=-0.0006
+adg_impr_lag1          AUC=0.5740  d=-0.0019
+adg_rev_lag1           AUC=0.5750  d=-0.0009
+adg_users_lag1         AUC=0.5740  d=-0.0019
+adg_avgp_lag1          AUC=0.5756  d=-0.0004
+adg_ctr_lag1           AUC=0.5759  d=-0.0001
+adg_rpm_lag1           AUC=0.5754  d=-0.0006
+adg_rpc_lag1           AUC=0.5757  d=-0.0002
+adg_freq_lag1          AUC=0.5756  d=-0.0004
+adg_clicks_lag2        AUC=0.5749  d=-0.0010
+adg_impr_lag2          AUC=0.5744  d=-0.0015
+adg_rev_lag2           AUC=0.5716  d=-0.0043
+adg_users_lag2         AUC=0.5744  d=-0.0016
+adg_avgp_lag2          AUC=0.5758  d=-0.0002 
Vấn đề: bị trùng thông tin (redundant)

Baseline đã one-hot adgroup_id (300 cột) → LogReg đã học sẵn CTR lịch sử của từng sản phẩm rồi. Giờ bạn thêm adg_ctr_lag1 vào = đưa lại đúng thông tin model đã biết → không thêm gì mới → d ≈ 0 (thậm chí hơi âm vì thêm nhiễu).

Nhớ 2 bảng này chứng minh cùng 1 chuyện:

	USER-ONLY (bỏ one-hot ad)	Baseline (có one-hot ad)adg_ctr_lag1	+0.0506 ✅ mạnh	−0.0001 ❌ vô dụng

→ Cùng 1 feature, lúc mạnh lúc vô dụng. Khác biệt duy nhất = có one-hot ad hay không. Điều này chứng minh: adg_ctr_lag1 và one-hot adgroup_id làm y hệt nhau → có 1 cái là đủ.

Kết luận (đây mới là finding quan trọng)

Feature CTR-lag của bạn KHÔNG vô dụng — nó chỉ trùng với one-hot ad. Nên có 2 hướng dùng, không phải "thêm vào":

Hướng A — THAY THẾ (nên làm): Bỏ one-hot adgroup_id (300 cột), thay bằng vài cột CTR-lag:

Python
1
run(["price","adg_ctr_lag1","adg_ctr_lag2","adg_ctr_lag3","cate_ctr_lag1"], USER_CAT)
Show more lines

→ Nếu ra ≈ 0.5759 (bằng baseline) mà chỉ dùng 4 cột thay vì 300 → thắng lớn về sự gọn nhẹ + khái quát hoá. Đây là điểm bán được.

Hướng B — chồng thêm (bạn vừa thử): Giữ one-hot + thêm CTR-lag → d≈0 → vô nghĩa, bỏ hướng này.

Vì sao đây thực ra là kết quả TỐT cho báo cáo

Bạn chứng minh được: one-hot ad (tĩnh, 300 cột) ↔ CTR-lag (động, 4 cột) tương đương nhau. CTR-lag còn có ưu thế:

Gọn hơn 75 lần
Xử lý được sản phẩm mới (one-hot chịu thua sản phẩm chưa thấy, CTR-lag thì fallback về prior)
Mang chiều thời gian




SAU KHI BỎ ONE HOT ID
USER-ONLY  AUC=0.5197

+adg_clicks_lag1        AUC=0.5386  d=+0.0188
+adg_impr_lag1          AUC=0.5307  d=+0.0110
+adg_rev_lag1           AUC=0.5373  d=+0.0176
+adg_users_lag1         AUC=0.5307  d=+0.0110
+adg_avgp_lag1          AUC=0.5198  d=+0.0000
+adg_ctr_lag1           AUC=0.5704  d=+0.0506
+adg_rpm_lag1           AUC=0.5538  d=+0.0341
+adg_rpc_lag1           AUC=0.5204  d=+0.0007
+adg_freq_lag1          AUC=0.5200  d=+0.0003
+adg_clicks_lag2        AUC=0.5298  d=+0.0101
+adg_impr_lag2          AUC=0.5238  d=+0.0041
+adg_rev_lag2           AUC=0.5322  d=+0.0124
+adg_users_lag2         AUC=0.5238  d=+0.0041
+adg_avgp_lag2          AUC=0.5187  d=-0.0011
+adg_ctr_lag2           AUC=0.5652  d=+0.0455
+adg_rpm_lag2           AUC=0.5364  d=+0.0166
+adg_rpc_lag2           AUC=0.5191  d=-0.0007
+adg_freq_lag2          AUC=0.5153  d=-0.0045
+adg_clicks_lag3        AUC=0.5234  d=+0.0036
+adg_impr_lag3          AUC=0.5189  d=-0.0008
+adg_rev_lag3           AUC=0.5245  d=+0.0047
+adg_users_lag3         AUC=0.5189  d=-0.0008
+adg_avgp_lag3          AUC=0.5198  d=+0.0001
+adg_ctr_lag3           AUC=0.5664  d=+0.0467
+adg_rpm_lag3           AUC=0.5341  d=+0.0144
+adg_rpc_lag3           AUC=0.5198  d=+0.0000
+adg_freq_lag3          AUC=0.5190  d=-0.0008
+adg_ctr_d12            AUC=0.5200  d=+0.0003
+adg_ctr_d13            AUC=0.5144  d=-0.0054
+adg_rpm_d12            AUC=0.5201  d=+0.0004
+adg_rpm_d13            AUC=0.5193  d=-0.0005
+adg_rpc_d12            AUC=0.5197  d=-0.0000
+adg_rpc_d13            AUC=0.5201  d=+0.0004
+adg_freq_d12           AUC=0.5198  d=+0.0001
+adg_freq_d13           AUC=0.5194  d=-0.0004
+adg_impr_r12           AUC=0.5197  d=-0.0001
+adg_clicks_r12         AUC=0.5198  d=+0.0000
+adg_rev_r12            AUC=0.5197  d=-0.0000
+adg_users_r12          AUC=0.5197  d=-0.0001
+cate_clicks_lag1       AUC=0.5191  d=-0.0007
+cate_impr_lag1         AUC=0.5196  d=-0.0002
+cate_rev_lag1          AUC=0.5197  d=-0.0001
+cate_users_lag1        AUC=0.5196  d=-0.0001
+cate_avgp_lag1         AUC=0.5204  d=+0.0007
+cate_ctr_lag1          AUC=0.5432  d=+0.0235
+cate_rpm_lag1          AUC=0.5267  d=+0.0069
+cate_rpc_lag1          AUC=0.5207  d=+0.0010
+cate_freq_lag1         AUC=0.5177  d=-0.0020
+cate_clicks_lag2       AUC=0.5189  d=-0.0009
+cate_impr_lag2         AUC=0.5199  d=+0.0002
+cate_rev_lag2          AUC=0.5193  d=-0.0004
+cate_users_lag2        AUC=0.5201  d=+0.0004
+cate_avgp_lag2         AUC=0.5197  d=-0.0000
+cate_ctr_lag2          AUC=0.5360  d=+0.0163
+cate_rpm_lag2          AUC=0.5219  d=+0.0021
+cate_rpc_lag2          AUC=0.5195  d=-0.0002
+cate_freq_lag2         AUC=0.5214  d=+0.0016
+cate_clicks_lag3       AUC=0.5185  d=-0.0012
+cate_impr_lag3         AUC=0.5200  d=+0.0003
+cate_rev_lag3          AUC=0.5182  d=-0.0015
+cate_users_lag3        AUC=0.5202  d=+0.0005
+cate_avgp_lag3         AUC=0.5198  d=+0.0001
+cate_ctr_lag3          AUC=0.5347  d=+0.0149
+cate_rpm_lag3          AUC=0.5226  d=+0.0029
+cate_rpc_lag3          AUC=0.5199  d=+0.0002
+cate_freq_lag3         AUC=0.5213  d=+0.0015
+cate_ctr_d12           AUC=0.5219  d=+0.0022
+cate_ctr_d13           AUC=0.5198  d=+0.0001
+cate_rpm_d12           AUC=0.5197  d=-0.0000
+cate_rpm_d13           AUC=0.5171  d=-0.0026
+cate_rpc_d12           AUC=0.5203  d=+0.0006
+cate_rpc_d13           AUC=0.5204  d=+0.0006
+cate_freq_d12          AUC=0.5195  d=-0.0003
+cate_freq_d13          AUC=0.5193  d=-0.0005
+cate_impr_r12          AUC=0.5196  d=-0.0001
+cate_clicks_r12        AUC=0.5198  d=+0.0000
+cate_rev_r12           AUC=0.5198  d=+0.0000
+cate_users_r12         AUC=0.5196  d=-0.0002

