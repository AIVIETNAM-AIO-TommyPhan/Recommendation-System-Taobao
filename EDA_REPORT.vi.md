# Báo cáo EDA — Bộ dữ liệu CTR Taobao

`dataset/train.csv` (240,000 × 21) và `dataset/test.csv` (60,000 × 21), biến mục tiêu `click`.

Được tạo bởi `eda_scripts/eda_taobao.py`; bản nháp tự động sinh nằm ở
`eda_out/report_generated.md`, các biểu đồ ở `eda_out/plots/`. File này là bản đã được
rà soát lại — nó sửa hai chỗ mà các quy tắc tổng quát (generic heuristics) đưa ra lời
khuyên sai đối với bộ dữ liệu này, và bổ sung những phát hiện mà các pha phân tích
tổng quát không thể thấy được.

Các hình minh họa nhúng bên dưới được xây dựng riêng cho báo cáo này bởi
`eda_scripts/report_figures.py` (kết quả xuất ra tại `eda_out/report_figures/`), mỗi
phát hiện chính có một hình. Màu cam đánh dấu tín hiệu phía quảng cáo (ad-side)
xuyên suốt; màu xanh dương đánh dấu phía người dùng/ngữ cảnh (user/context).

> **📘 Cách đọc báo cáo này.** Các ô 📘 giải thích một thuật ngữ thống kê ở lần đầu
> nó xuất hiện — bỏ qua nếu bạn đã quen; phát hiện nằm phía trên ô vẫn tự đứng vững
> mà không cần đến chúng.

---

## 0. Đã xử lý: `train.csv` bị Excel làm hỏng đã được thay thế

Một phiên bản trước đó của `train.csv` đã bị round-trip qua Excel — timestamp dạng
ISO bị viết lại thành định dạng ngắn theo locale (`5/5/2017 16:00`) và bị cắt mất phần
giây, khiến 177,259 timestamp khác biệt bị dồn lại chỉ còn 9,268. File đó đã được nạp
lại và phần hư hỏng không còn nữa:

| Kiểm tra | Giá trị | Kỳ vọng |
|---|---|---|
| Định dạng `time_stamp` | `2017-05-05 16:00:03` | ISO có giây ✔ |
| Số timestamp khác biệt | 177,259 | ~177k ✔ |
| Số giá trị `.dt.second` khác biệt | 60 | không phải hằng số ✔ |
| Định dạng `date` | `2017-05-05` | ISO, khớp với `test.csv` ✔ |
| `hour` / `weekday` / `date` so với `time_stamp` | khớp nhau tất cả | ✔ |
| Số dòng trùng lặp hoàn toàn | 0 | ✔ |

Cả hai file giờ đều parse được chỉ với một `format=` duy nhất, và mốc chia
`04:24:15` trong README có thể kiểm chứng lại được từ dữ liệu. Dù vậy vẫn nên giữ
guard tại thời điểm load, vì lỗi này âm thầm không báo:

```python
df["time_stamp"] = pd.to_datetime(df["time_stamp"])
assert df["time_stamp"].dt.second.nunique() > 1, "train.csv lost seconds — regenerate"
```

**Về trùng lặp.** 11,213 cặp `(userid, time_stamp)` xuất hiện nhiều hơn một lần, bao
phủ tổng cộng 22,999 dòng (trong đó 11,786 dòng là bản sao thứ-hai-trở-đi của một
cặp). Nhưng **không có dòng nào** trùng lặp toàn bộ (full-row) — đó là trường hợp một người dùng
thấy nhiều quảng cáo trong cùng một giây, là hiện tượng thật. Số lượng trùng lặp
trong báo cáo tổng quát được tính sau khi đã bỏ `userid`/`time_stamp`, nên đây cũng
không phải vấn đề trùng lặp dữ liệu. Không nên khử trùng (de-duplicate) các dòng này.

Mọi phát hiện bên dưới đều đã được tính lại trên các file đã nạp lại và cho ra kết
quả giống hệt.

> **📘 Vì sao mục này được đánh số 0.** Đây là một cổng kiểm tra tính toàn vẹn dữ
> liệu (data-integrity gate), không phải một bước phân tích. Không mục nào trong §1–§8
> đáng tin cậy chừng nào file đầu vào chưa được xác nhận là tốt, vì vậy mục này được
> đánh số 0: nó tồn tại để (1) *chứng minh* file hiện tại là ổn bằng một bảng kiểm tra
> thay vì chỉ khẳng định suông, (2) để lại một guard chống lỗi âm thầm để sự cố Excel
> không thể tái diễn mà không ai hay biết, và (3) ngăn người đọc khử trùng 22,999 dòng
> cùng-giây chỉ *trông giống* trùng lặp.

---

## 1. Vai trò của các cột — một cột số nguyên không đương nhiên là một đại lượng

**Quan sát.** 16 trong số 19 cột dùng để mô hình hóa được lưu dưới dạng số nguyên,
nên `infer_column_kinds` gọi tất cả là "numeric" và đưa vào pha phân tích số, và pha
này ngoan ngoãn tính skew, biên ngoại lai IQR và hệ số Pearson *r* trên từng cột.
Nhưng kiểu `int` chỉ là *cách lưu trữ* — nó không nói gì về **ý nghĩa** của con số. 16
số nguyên này thật ra là bốn loại khác nhau:

- một **đại lượng** thật sự — `price` (¥200 đúng là nhiều hơn ¥100 một khoản ¥100);
- các **mã ID** — `adgroup_id`, `cate_id`, `brand`, `customer`, `cms_segid` — những
  cái tên tình cờ được viết bằng chữ số, mà độ lớn thì vô nghĩa: quảng cáo #660,710
  không "nhiều" hơn quảng cáo #85,419, cũng như `SELECT AVG(primary_key)` chẳng có
  nghĩa gì;
- các **mã có thứ tự** — `age_level`, `pvalue_level`, `cms_group_id`, … — nơi *thứ
  tự* là thật nhưng khoảng cách giữa các mức thì không;
- các **cột tuần hoàn** — `hour`, `weekday` — nơi 23:00 nằm sát 00:00.

Sai lầm duy nhất của công cụ là đối xử với cả 16 cột như thể chúng đều thuộc loại
đầu tiên. Dưới đây: một dòng cụ thể để thấy các cột, rồi hai kiểu mà sai lầm đó gây họa.

> **📘 Đại lượng và nhãn.** Một **đại lượng** (quantity) là con số mà *độ lớn* và *hiệu
> số* đều có thật — bạn có thể lấy trung bình và trừ nó. Một **nhãn** (label) là một
> danh tính chỉ mượn chữ số để viết ra (một ID, một mã danh mục). Cả hai đều hiện lên
> dưới dạng `int`; chỉ ý nghĩa là khác — và mean / skew / IQR / tương quan chỉ có nghĩa
> trên đại lượng.

**Một dòng, nhìn gần.** Bốn lượt hiển thị của cùng một quảng cáo (`adgroup_id` 433864):

| adgroup_id | cate_id | campaign_id | customer | brand | price | cms_segid | cms_group_id | click |
|---|---|---|---|---|---|---|---|---|
| 433864 | 6185 | 12546 | 4001 | 275122 | ¥75.00 | 0 | 4 | 0 |
| 433864 | 6185 | 12546 | 4001 | 275122 | ¥75.00 | 0 | 3 | 1 |
| 433864 | 6185 | 12546 | 4001 | 275122 | ¥75.00 | 0 | 5 | 1 |
| 433864 | 6185 | 12546 | 4001 | 275122 | ¥75.00 | 30 | 4 | 0 |

Đọc ngang một dòng: mọi cột phía quảng cáo (`cate_id`, `campaign_id`, `customer`,
`brand`, `price`) đều **giống hệt nhau trên cả bốn dòng** — đây không phải bốn sự kiện
khác nhau, mà là thuộc tính của một quảng cáo được sao chép lên bốn lượt hiển thị
riêng biệt (quan hệ hàm mà §6 nói chi tiết). Chỉ `cms_segid`, `cms_group_id` và
`click` thay đổi, vì chúng mô tả *người dùng và thời điểm*, không phải quảng cáo.

**Sai lầm gây họa ở đâu — hai ví dụ.**

*(1) Lấy trung bình một ID.* Công cụ gắn cờ "`adgroup_id` skew −2.121, 6,728 điểm
ngoại lai IQR, đề xuất biến đổi log." Mọi con số ở đó đều là phép tính số học trên các
mã ID. Các tứ phân vị là Q1 = 619,783 và Q3 = 715,187, nên biên ngoại lai IQR rơi vào
[476,677, 858,293] và 6,728 lượt hiển thị nằm ngoài biên — nhưng các "điểm ngoại lai"
đó chỉ là những quảng cáo có chữ số ID tình cờ nhỏ hoặc lớn. Quảng cáo 433864 ở trên bị
gắn cờ, nhưng nó là một quảng cáo hoàn toàn bình thường (388 lượt hiển thị, CTR 15.5%).
Giá trị trung bình 660,710 và độ lệch −2.121 mô tả cách Taobao *cấp phát* số ID, chứ
không nói gì về các quảng cáo; biến đổi log chỉ nén lại không gian ID chứ không sửa
được gì.

*(2) Tương quan giữa hai ID.* Công cụ gắn cờ "`cms_segid` ~ `cms_group_id`, r =
0.984," một liên hệ đường-thẳng gần như hoàn hảo. Nó sai hai lần. **Thứ nhất**, con số
0.984 chỉ xuất hiện sau khi âm thầm loại bỏ 55.6% số dòng mang sentinel `cms_segid = 0`
(một giá trị giữ chỗ đại diện cho *missing*, xem §3); trên toàn bộ 240,000 dòng nó là
0.453. **Thứ hai, và cơ bản hơn:** tương quan hỏi *"hai cột này có tăng cùng nhau theo
một đường thẳng không?"* — câu hỏi sai cho các mã ID. Liên hệ thật sự là chính xác
nhưng không tuyến tính — mỗi mức trong **96** mức `cms_segid` khác 0 ánh xạ vào đúng
một `cms_group_id`. `cms_segid` *xác định* `cms_group_id`, theo cách một khóa ngoại
(foreign key) trỏ tới đúng dòng của nó, và tương quan không thấy được điều đó: nó báo
một con số 0.453 tầm thường cho một quan hệ thực chất chặt 100%.

> **📘 *r*, hệ số tương quan.** Pearson **r** chạy từ −1 đến +1 và chỉ đo mức độ một
> **đường thẳng** duy nhất khớp với hai cột: +1 là đường thẳng đi lên hoàn hảo, 0 là
> không có liên hệ tuyến tính, −1 là đường thẳng đi xuống hoàn hảo. Một quan hệ có thể
> dự đoán được hoàn hảo mà *vẫn* nhận điểm *r* thấp nếu hình dạng của nó không phải
> đường thẳng — đúng như những gì xảy ra với `cms_segid`.

![Vai trò cột: độ lớn ID so với CTR, và hàm bậc thang segid→group](eda_out/report_figures/fig_column_roles.png)

**Biểu đồ bên trái** vẽ CTR của từng quảng cáo theo số `adgroup_id` thô. Dải màu cam
là toàn bộ phần bị quy tắc ngoại lai gắn cờ (dưới biên 476,677 — không quảng cáo nào
chạm biên trên 858,293); vậy mà CTR của chúng (9%–31%) nằm ngay trong khoảng với các
điểm xanh mà quy tắc bỏ qua. Quy tắc chỉ phản ứng với việc số ID nhỏ, điều này không
cho ta thông tin gì. (So sánh với `price` ở §4, nơi phần đuôi bị gắn cờ — 12,194 dòng
trên ¥566.5 — thực sự là các mặt hàng đắt tiền thật.)

**Biểu đồ bên phải** vẽ mọi cặp `(cms_segid, cms_group_id)` quan sát được. Quan hệ
tuyến tính sẽ rải rác quanh một đường dốc lên; thay vào đó các điểm rơi trên các bậc
thang nằm ngang phẳng — mỗi `cms_segid` chỉ ứng với một, và chỉ một, `cms_group_id` —
một cầu thang 96-sang-13, không phải một đường thẳng. Cầu thang đó *chính là* quan hệ
hàm, và đó là lý do tương quan cho ra hai đáp số khác nhau, cả hai đều sai (0.453 /
0.984): không đường thẳng đơn nào khớp được với một cầu thang.

> **📘 Quy tắc mang theo.** Kiểu `int` **không** khiến một cột trở thành đại lượng. Nếu
> các chữ số là một *nhãn* (một ID hay một mã cấp độ), độ lớn của chúng là tùy ý —
> mean, skew, IQR và Pearson *r* đều chỉ là "phép tính số học trên số ID" và sẽ gây
> hiểu lầm. Quan hệ giữa các cột như vậy là **quan hệ hàm** (giá trị A có ghim chặt giá
> trị B không? — tìm bằng kiểm tra ánh xạ 1-1), chứ không phải **quan hệ tuyến tính**
> (chúng có tăng cùng nhau không? — tìm bằng tương quan). Hãy xác định vai trò của từng
> cột trước; thống kê phù hợp sẽ theo sau.

**Thay đổi.** `eda_scripts/prep_taobao.py` gán vai trò thực sự trước khi phân tích:

| Vai trò | Các cột |
|---|---|
| Liên tục (Continuous) | `price` — **cột duy nhất** |
| ID định danh (Nominal ID) | `adgroup_id`, `cate_id`, `campaign_id`, `customer`, `brand`, `cms_segid` |
| Mã có thứ tự (Ordinal code) | `cms_group_id`, `age_level`, `pvalue_level`, `shopping_level`, `new_user_class_level` |
| Nhị phân (Binary) | `final_gender_code`, `occupation` |
| Chu kỳ (Cyclic) | `hour`, `weekday` |
| Ngữ cảnh (Context) | `pid` |
| Bị loại bỏ (Dropped) | `userid` (147k giá trị, chỉ là row ID), `time_stamp` (đã được thay thế bởi `date`/`hour`/`weekday`) |

**Cách kiểm chứng:** pha phân tích số giờ chỉ còn lại một mình `price`, và bản đồ
nhiệt tương quan (correlation heatmap) không còn gì để bị diễn giải sai quá mức nữa.
Cả hai điều này đều đúng khi chạy lại.

---

## 2. Cân bằng của biến mục tiêu

`click` chiếm 48,446 / 240,000 = **20.19%** dương tính (test: 19.26%).

![Target balance and CTR](eda_out/report_figures/fig_target_balance.png)

**Tỷ lệ mất cân bằng** (imbalance ratio) đơn giản là số lượng mẫu âm trên mỗi mẫu
dương — lớp đa số chia cho lớp thiểu số:

```
191,554 non-clicks / 48,446 clicks = 3.954
```

Ở đây `p` là **tỷ lệ mẫu dương** — tức tỷ lệ click — bằng số click chia cho tổng số
dòng: `p = 48,446 / 240,000 = 0.201858`, chính là con số **20.19%** ở đầu mục. Phần
còn lại, `1 − p = 191,554 / 240,000 = 0.798142`, là tỷ lệ non-click. Vậy nên nếu chia
cả tử và mẫu của phép chia trên cho cùng tổng N = 240,000, hai số đếm biến thành đúng
hai tỷ lệ này, và `191,554 / 48,446` được viết lại gọn thành `(1 − p) / p` (chia cả tử
và mẫu cho cùng một số không bao giờ làm đổi giá trị phân số). Cả hai cách đều cho
3.954 — vậy cứ mỗi lượt click thì có ~4 lượt không click. Tỷ lệ 1.0 là cân bằng hoàn
hảo; quy tắc kinh nghiệm thông thường coi tỷ lệ >10 là mất cân bằng nghiêm trọng, nên ở
mức 3.95 bộ dữ liệu này chỉ mất cân bằng **nhẹ** — đủ để khiến accuracy trở nên vô
dụng, nhưng chưa cần đến resampling hay class weights.

> **📘 Khái niệm — vì sao tỷ lệ này quan trọng.** Ở mức 20% dương tính, một mô hình
> luôn dự đoán "không click" cho *tất cả mọi người* đã đạt 80% accuracy trong khi
> hoàn toàn vô dụng — đó chính là ý nghĩa của việc "accuracy không có giá trị thông
> tin dưới điều kiện mất cân bằng". Các chỉ số xếp hạng và xác suất (**AUC**,
> **log loss**) không mắc phải điểm mù này vì chúng đánh giá *xác suất* dự đoán, chứ
> không phải một ngưỡng 0.5 duy nhất. Ở mức 3.95, độ lệch là nhẹ, nên chưa cần
> resampling/class-weights; các kỹ thuật đó chỉ cần thiết khi tỷ lệ tiến gần mức >10.

Từ đây suy ra hai điều. Thứ nhất, accuracy tại ngưỡng 0.5 không có giá trị thông tin
— nên dùng AUC và log loss, điều mà baseline đã làm. Thứ hai, **CTR 20% cao gấp
khoảng 4 lần mức ~5% thường thấy trong log quảng cáo thực tế của Taobao**, điều này
phù hợp với giả thuyết rằng các mẫu âm đã bị downsample khi tạo bộ dữ liệu mẫu. Đây
chỉ là một *giả thuyết*: `Sample_Dataset.ipynb` không có trong repo (nó nằm trong
`.gitignore`), nên không thể xác nhận điều này tại đây.

Nếu giả thuyết trên đúng, hệ quả thực tiễn là **xác suất dự đoán sẽ không chuyển đổi
được** — một mô hình huấn luyện ở base rate 20% sẽ hệ thống hóa việc dự đoán quá cao
so với luồng dữ liệu thực tế có base rate 5%. Các chỉ số xếp hạng không bị ảnh hưởng;
nhưng ngưỡng quyết định và các phép tính giá trị kỳ vọng (expected value) thì có.
Cần hiệu chỉnh lại (recalibrate) trước khi chọn bất kỳ ngưỡng nào.

Liên quan: ô markdown "Standard classification sanity checks" (ô số 10) trong
`Baseline_Model.ipynb` viết "CTR is heavily imbalanced (~5% positive)". Nhận xét đó
đã lỗi thời — tỷ lệ thực tế là 20%.

---

## 3. Giá trị thiếu — cả bốn trường hợp đều là giá trị sentinel, không trường hợp nào mang thông tin

**Quan sát.** Các file không chứa giá trị `NaN` nào; giá trị thiếu được mã hóa dưới
dạng `0`. Việc giải mã điều này chính là thứ khiến pha phân tích này có ý nghĩa:

| Cột | Tỷ lệ giá trị `0` | `0` có thực sự là missing không? |
|---|---|---|
| `cms_segid` | 55.6% | có — đã giải mã |
| `pvalue_level` | 49.9% | có — đã giải mã (các mức là 1–3) |
| `brand` | 48.0% | có — đã giải mã (131 trong 300 quảng cáo không có brand) |
| `new_user_class_level` | 25.7% | có — đã giải mã (các mức là 1–4) |
| `occupation` | 92.4% | **không** — 0 nghĩa là "không phải sinh viên", một mức thực |
| `final_gender_code` | — | **không** — mã hóa 1/2, không tồn tại giá trị 0 |
| `cms_group_id`, `age_level` | 0.03% | **không** — quá hiếm để là sentinel; chỉ là một nhóm nhỏ có thật |

![0-sentinel share by column](eda_out/report_figures/fig_missing_sentinels.png)

**Việc thiếu dữ liệu có mang thông tin không?** Không — và đây chính là kết quả hữu
ích:

| Cột | CTR khi thiếu | CTR khi có giá trị |
|---|---|---|
| `brand` | 0.2015 | 0.2022 |
| `cms_segid` | 0.2015 | 0.2022 |
| `pvalue_level` | 0.2022 | 0.2015 |
| `new_user_class_level` | 0.2034 | 0.2013 |

Mọi khoảng chênh lệch đều dưới 0.3 điểm phần trăm so với base rate 20.19%. Ngưỡng để
coi việc thiếu dữ liệu là "có mang thông tin" theo phương pháp luận là chênh lệch
10 điểm phần trăm giữa các lớp; không trường hợp nào ở đây gần đạt đến mức đó.

**Thay đổi.** Coi "unknown" là một hạng mục (category) riêng và bỏ qua các cột chỉ
báo (indicator) `_is_missing`. Với one-hot encoding, việc này không tốn thêm chi phí
— "unknown" chỉ đơn giản trở thành một mức khác. Kế hoạch tổng quát đề xuất một
indicator cho `cms_segid` chỉ vì tỷ lệ vượt 50%; quy tắc đó không còn áp dụng khi ta
đã biết việc thiếu dữ liệu không mang tín hiệu gì. **Không** nên impute bằng median —
median của một tập các brand ID là vô nghĩa.

**Cách kiểm chứng:** thêm các indicator này thì AUC trên tập held-out phải không đổi.
Nếu nó thay đổi, kết luận này là sai.

> **📘 Khái niệm — hạng mục "unknown" so với chỉ báo missing.** Hai cách này giải
> quyết hai việc khác nhau. Biến "unknown" thành một mức one-hot riêng
> (`pvalue_unknown`) cho phép mô hình *biểu diễn* được sự thiếu hụt — cột đó đã đánh
> dấu mọi dòng bị thiếu, miễn phí. Một chỉ báo `_is_missing` riêng biệt là một cột
> *thứ hai*, chỉ có giá trị khi bản thân **hành vi bị thiếu** dự đoán được biến mục
> tiêu (ví dụ: người dùng giấu thu nhập lại click nhiều hơn hẳn). Bảng trên cho thấy
> điều ngược lại: CTR chênh lệch dưới 0.3 điểm phần trăm giữa thiếu và có, nên chỉ báo
> đó chỉ trùng lặp với `*_unknown` với hệ số gần như bằng không. Giữ hạng mục, bỏ chỉ
> báo.

---

## 4. `price` — biến số liên tục thực sự duy nhất

Skew **1.964** — một con số duy nhất cho biết một phân phối *bất đối xứng* đến mức
nào: dương nghĩa là có một đuôi dài về bên phải (ở đây là các mặt hàng đắt, kéo đến
¥999) và số càng lớn thì độ lệch càng mạnh, nên đây là lệch phải mạnh. `log1p` đưa về
−0.530. 12,194 dòng (5.1%) là điểm ngoại lai theo IQR, tất cả đều trên ¥566.5 — là các
mặt hàng đắt tiền thật sự, không phải lỗi dữ liệu. (Cả hai con số skew đều được diễn
giải trong các ô 📘 bên dưới.)

![price distribution and CTR by decile](eda_out/report_figures/fig_price.png)

Biểu đồ hộp (box plot) thể hiện rõ độ lệch và các "điểm ngoại lai": hộp (Q1 ¥99 – Q3
¥286, trung vị ¥138) bị dồn về bên trái trong khi một dải dài các điểm kéo dài đến
¥999. Mọi điểm vượt qua biên trên (¥566.5) đều bị đánh dấu là ngoại lai — 12,194
điểm, chiếm 5.1% — nhưng chúng là các mặt hàng đắt tiền có thật, không phải lỗi.
`log1p` (bên phải) kéo hộp trở lại về giữa và phần đuôi gần như biến mất.

![price box plot, raw vs log1p](eda_out/report_figures/fig_price_boxplot.png)

| Điểm mốc trên box-plot | Giá trị | Diễn giải |
|---|---|---|
| min | ¥2.4 | mặt hàng rẻ nhất |
| Q1 (percentile thứ 25) | ¥99 | 25% quảng cáo có giá ≤ 99 |
| trung vị (percentile thứ 50) | ¥138 | một nửa có giá ≤ 138 |
| Q3 (percentile thứ 75) | ¥286 | 75% có giá ≤ 286 |
| max | ¥999 | mặt hàng đắt nhất |
| IQR = Q3 − Q1 | ¥187 | độ rộng của hộp (50% giữa) |
| biên trên = Q3 + 1.5·IQR | ¥566.5 | mọi điểm vượt qua đây được vẽ là ngoại lai |

> **📘 Khái niệm — Độ lệch/Skewness (vì sao là 1.964).** Skew là một con số duy nhất,
> không đơn vị, đo mức độ *bất đối xứng* của một phân phối:
> `skew = mean((x−x̄)³) / [mean((x−x̄)²)]^1.5`. Lũy thừa bậc ba ở tử số **giữ nguyên
> dấu**, nên một dải đuôi phải dài gồm các mặt hàng đắt tiền (¥999 → độ lệch³ ≈
> +500 triệu) lấn át phía rẻ, và tổng cho ra kết quả **dương** → lệch phải. Chia cho
> phương sai^1.5 để triệt tiêu đơn vị ¥³, còn lại 1.964. Quy tắc kinh nghiệm:
> |skew| < 0.5 ≈ đối xứng, > 1 = lệch mạnh. Ở đây, mean ¥204.84 > trung vị ¥138 chính
> là câu chuyện tương tự thể hiện qua hai con số khác — mean bị kéo lên bởi phần
> đuôi, còn trung vị thì không.

> **📘 Khái niệm — vì sao log1p đảo dấu thành −0.53.** `log1p(x) = log(1+x)` nén các
> giá trị lớn mạnh hơn nhiều so với giá trị nhỏ (¥138→4.93, ¥999→6.91 — khoảng cách
> 861 đơn vị co lại còn ~2), nên nó kéo phần đuôi phải vào gần và triệt tiêu phần lớn
> độ lệch. Nó hơi *quá tay* ở đầu thấp (¥2.4 → 1.22 nằm khá xa bên trái so với trung
> vị 4.93), tạo ra một đuôi **trái** mờ nhạt — đó là lý do skew rơi vào phía âm ngay
> sau điểm 0. Dấu chỉ cho biết nó nghiêng về phía nào; |−0.53| ≪ |1.964|, nên hình
> dạng đối xứng hơn nhiều so với trước.

Nhưng quan sát quan trọng hơn là: **`price` chỉ có 159 giá trị khác biệt trên 300
quảng cáo**, vì nó là một thuộc tính của quảng cáo, không phải của lượt hiển thị
(impression). Đây là một thuộc tính cardinality thấp phía quảng cáo đang khoác kiểu
dữ liệu liên tục.

Và nó hầu như không làm thay đổi biến mục tiêu. CTR theo decile của giá vẫn nằm trong
khoảng 0.174–0.229 mà không có xu hướng đơn điệu — decile rẻ nhất (0.2222) và decile
đắt nhất (0.2051) nằm ở hai bên các decile giữa. Mức độ phân tách theo trung vị lớp
là 0.0 độ lệch chuẩn.

**Thay đổi.** Dùng `log1p` + `StandardScaler` cho mô hình baseline tuyến tính (rẻ,
xử lý được độ lệch, không có lý do gì để không làm). Không nên clip phần đuôi. Đừng
kỳ vọng nhiều từ biến này khi đứng một mình — giá trị của nó nằm ở tương tác với các
đặc trưng người dùng (độ nhạy cảm về giá theo `pvalue_level`), điều mà một mô hình
tuyến tính không thể biểu diễn được nếu không có một điều khoản tương tác (crossed
term) tường minh.

> **📘 Khái niệm — StandardScaler.** Nó chuẩn hóa lại một cột về **mean 0, std 1**
> qua công thức z-score `z = (x − mean) / std`. Trên `log1p(price)` (mean 5.03, std
> 0.80): ¥138 → z = −0.12 (gần đúng mean), ¥999 → z = +2.33 (2.3 độ lệch chuẩn trên
> mean), ¥2.4 → z = −4.73. Các mô hình tuyến tính/dựa trên khoảng cách (logistic,
> SVM, kNN) **nhạy cảm với tỷ lệ** — nếu không chuẩn hóa, một đặc trưng có khoảng
> giá trị 1–7 sẽ lấn át các cột one-hot 0/1 chỉ vì độ lớn, làm lệch gradient. Các mô
> hình Tree/GBM không cần bước này (chúng so sánh ngưỡng, không phải tổng có trọng
> số), đó là lý do bước này chỉ dành riêng cho *baseline tuyến tính*. **Lưu ý quan
> trọng:** chỉ fit `mean`/`std` trên **tập train**, sau đó áp dụng cho val/test; tính
> chúng trên toàn bộ dữ liệu sẽ làm rò rỉ thông tin của tập test. Đó là lý do phần
> §Cleaning bọc `log1p → StandardScaler` trong một `ColumnTransformer`.

---

## 5. Đặc trưng nào thực sự mang tín hiệu

**Bảng này đo gì.** Với một đặc trưng như `age_level`, hãy nhìn từng giá trị của nó
(mỗi giá trị là một **mức**) và tính CTR của mức đó. Nếu mọi mức đều bấm ở tỷ lệ gần
như nhau, đặc trưng chẳng cho ta thông tin gì; nếu các mức bấm ở những tỷ lệ rất khác
nhau, đặc trưng phân tách click tốt. Ta đo **độ trải rộng** đó bằng một **độ lệch
chuẩn (SD)** — trung bình các CTR-từng-mức nằm cách mốc chung 20.19% bao xa — kèm hai
điều chỉnh: mỗi mức được **cân theo số lượt hiển thị** (mức phổ biến tính nặng hơn mức
hiếm), và các mức **dưới 200 lượt hiển thị bị loại** (một mức chỉ thấy vài lần có thể
cho CTR ăn may — là nhiễu chứ không phải tín hiệu). **Càng cao = càng nhiều tín hiệu.**

**"Mức" là gì — một ví dụ tính tay.** Một **mức** là một giá trị riêng biệt mà một cột
có thể nhận: `age_level` nhận các giá trị 1–6, nên nó có 6 mức, và mỗi mức có CTR
riêng:

| Mức (`age_level`) | Lượt hiển thị | CTR | Trọng số wᵢ | wᵢ·(CTRᵢ − 0.2019)² |
|---|---|---|---|---|
| 1 | 16,880 | 0.226 | 0.070 | 0.000042 |
| 2 | 60,109 | 0.212 | 0.250 | 0.000028 |
| 3 | 80,704 | 0.198 | 0.336 | 0.000004 |
| 4 | 50,279 | 0.188 | 0.210 | 0.000039 |
| 5 | 30,047 | 0.198 | 0.125 | 0.000002 |
| 6 | 1,913 | 0.219 | 0.008 | 0.000002 |

> **📘 Cột trọng số, nói gọn.** **Trọng số** của một mức đơn giản là *phần nó chiếm
> trong tổng lượt hiển thị* — mức đó là bao nhiêu phần của dữ liệu. Tính bằng lượt hiển
> thị của mức ÷ tổng của mọi mức được giữ (16,880 + 60,109 + … + 1,913 = 239,932): mức
> 3 là 80,704 / 239,932 ≈ **0.336** (chiếm ~34% dữ liệu), còn mức 6 chỉ 1,913 / 239,932
> ≈ **0.008** (0.8%). Cân trọng số để mức phổ biến có tiếng nói lớn hơn — giống bỏ
> phiếu theo dân số — nên một mức hiếm có CTR bất thường không thể chi phối. Cả sáu
> trọng số cộng lại bằng 1 (toàn bộ dữ liệu).

Sáu mức bấm ở những tỷ lệ thực sự khác nhau — từ 18.8% (mức 4) đến 22.6% (mức 1). Để
ra SD, cộng **cột cuối** — các đóng góp `wᵢ·(CTRᵢ − 0.2019)²`, **không phải** cột trọng số:

```
0.000042 + 0.000028 + 0.000004 + 0.000039 + 0.000002 + 0.000002 = 0.000117
√0.000117 = 0.0108
```

Con số **0.0108** đó đúng bằng dòng `age_level` trong bảng xếp hạng bên dưới. (Nếu cộng
cột *trọng số* thì chỉ ra 1 — dấu hiệu các tỷ trọng đã đúng, không phải SD.) Ở đây độ trải rộng là thật nhưng nhỏ, nên
`age_level` xếp hạng thấp; còn `adgroup_id`, với ~300 mức trải từ 7.7% đến 45.2%, mới
là nơi độ trải rộng — và tín hiệu — lớn. (Bốn bước số học được trình bày chi tiết cho
đặc trưng đơn giản nhất là `pid` trong ô 📘 sau bảng.)

| Đặc trưng | Số mức | Khoảng CTR | SD có trọng số |
|---|---|---|---|
| `adgroup_id` | 297 | 0.077 – 0.452 | **0.0532** |
| `campaign_id` | 248 | 0.077 – 0.452 | 0.0505 |
| `customer` | 213 | 0.077 – 0.452 | 0.0484 |
| `brand` | 116 | 0.115 – 0.362 | 0.0334 |
| `cate_id` | 44 | 0.129 – 0.356 | 0.0238 |
| `cms_group_id` | 12 | 0.185 – 0.239 | 0.0113 |
| `age_level` | 6 | 0.188 – 0.226 | 0.0108 |
| `cms_segid` | 52 | 0.158 – 0.250 | 0.0096 |
| `pid` | 2 | 0.195 – 0.213 | 0.0086 |
| `hour` | 24 | 0.185 – 0.223 | 0.0060 |
| `occupation` | 2 | 0.200 – 0.219 | 0.0049 |
| `weekday` | 7 | 0.195 – 0.211 | 0.0048 |
| `new_user_class_level` | 5 | 0.196 – 0.206 | 0.0037 |
| `pvalue_level` | 4 | 0.185 – 0.204 | 0.0030 |
| `final_gender_code` | 2 | 0.195 – 0.203 | 0.0027 |
| `shopping_level` | 3 | 0.201 – 0.203 | **0.0004** |

![Feature signal strength](eda_out/report_figures/fig_feature_signal.png)

> **📘 Khái niệm — Weighted SD (cách tính ra thứ hạng).** Với mỗi đặc trưng, nó hỏi
> đúng một câu: *các mức của nó có CTR khác nhau thật sự không?* Công thức:
> `WSD = sqrt( Σ_mức  w_i · (CTR_i − 0.2019)² )`, với `w_i` là tỷ trọng lượt hiển thị
> của mức đó. Đọc theo 4 bước — (1) mỗi mức lệch bao xa khỏi nền 20.19%, (2) **bình
> phương** (bỏ dấu, phạt nặng lệch lớn — cùng ý tưởng với phương sai), (3) **nhân
> trọng số theo lượt hiển thị** để một mức hiếm có CTR bất thường không chi phối,
> (4) **căn bậc hai** để đưa về đơn vị CTR cho dễ đọc. Ví dụ tính tay cho `pid`
> (2 mức). **Mức A** — CTR 0.2125, trọng số 0.39: lệch 0.2125 − 0.2019 = +0.0106,
> bình phương 0.000112, ×0.39 = 0.000044. **Mức B** — CTR 0.1949, trọng số 0.61:
> lệch −0.0070, bình phương 0.000049, ×0.61 = 0.000030. Cộng = 0.000074, và
> `sqrt(0.000074) = 0.0086` — đúng giá trị trong bảng. Chính việc nhân trọng số theo
> lượt hiển thị —
> cộng với việc loại các mức dưới 200 lượt — biến nó thành ước lượng *tín hiệu thật*
> chứ không phải nhiễu mẫu nhỏ: một quảng cáo chỉ hiện 10 lần với CTR 60% **không**
> thổi phồng điểm số.

**Danh tính của quảng cáo mang lượng tín hiệu lớn hơn cả bậc độ lớn so với bất kỳ
thuộc tính người dùng nào.** Riêng `adgroup_id` đã trải dài từ CTR 7.7% đến 45.2%.
Mọi đặc trưng nhân khẩu học của người dùng đều dưới 0.012 — `shopping_level` gần như
là hằng số (0.201/0.202/0.203) và là đặc trưng yếu nhất trong toàn bộ bộ dữ liệu.

Điều này là dễ hiểu đối với CTR: *quảng cáo nào* chi phối hơn *ai đã xem nó*, và các
đặc trưng người dùng chỉ phát huy giá trị thông qua tương tác (người dùng này ×
danh mục này), chứ không phải như hiệu ứng chính (main effect). Một mô hình tuyến
tính chỉ dùng hiệu ứng chính sẽ gần như không nắm bắt được điều đó — cơ hội cải thiện
mô hình lớn nhất có sẵn ở đây là các đặc trưng tương tác chéo (crossed features)
hoặc một mô hình cây (tree) hay factorization-machine (FM).

**Thay đổi.** Giữ lại các đặc trưng người dùng yếu (chúng vẫn có giá trị trong tương
tác) nhưng đừng kỳ vọng hiệu ứng chính từ chúng. Ưu tiên các tương tác chéo giữa
`adgroup_id`/`cate_id` với phân khúc người dùng.
**Cách kiểm chứng:** xem liệu việc thêm các tương tác chéo có vượt qua baseline chỉ
dùng hiệu ứng chính về AUC hay không.

---

## 6. Sự dư thừa mang tính hàm số, không phải tuyến tính

> **📘 Nói cho dễ hiểu — ví von với mã bưu chính (ZIP).** "Phụ thuộc hàm số" chỉ có
> nghĩa là *một cột quyết định cột khác*. Hãy nghĩ tới mã bưu chính: khi đã biết mã
> ZIP, thì thành phố và tỉnh đã cố định — ghi thêm chúng vào các cột riêng chẳng thêm
> thông tin gì mới. Ở đây **`adgroup_id` chính là mã ZIP**: biết *đó là quảng cáo nào*
> thì tự động khóa luôn category, brand, giá, campaign và nhà quảng cáo của nó. Vậy
> năm cột đó không phải năm manh mối độc lập — chúng là các thuộc tính của cùng một
> quảng cáo được viết lại năm lần. Hai hệ quả: (1) đưa cả năm cột vào mô hình tuyến
> tính là lặp lại, không phải thêm tín hiệu; (2) tương quan không phát hiện được điều
> này, vì tương quan chỉ đo *các con số cùng tăng giảm với nhau*, mà đây là **nhãn,
> không phải đại lượng** (§1). Cách phát hiện đúng là hỏi "mỗi giá trị của A có ánh xạ
> về đúng một giá trị của B không?" — và ở đây câu trả lời là có, không ngoại lệ.

Bản đồ nhiệt tương quan giờ không tìm thấy cặp nào mạnh — điều này đúng, vì Pearson
chỉ tính trên một mình `price` thì không có gì để so sánh. Nhưng sự dư thừa thực sự
vẫn tồn tại và vô hình với tương quan, vì đó là quan hệ phụ thuộc hàm số
(functional dependence) giữa các biến phân loại (categorical):

```
ad side:    adgroup_id ─→ cate_id, brand, price   (toàn bộ 300 ánh xạ 1:1 — không ngoại lệ)
            adgroup_id ─→ campaign_id ─→ customer  (mỗi bước đều 1:1)
user side:  cms_segid  ─→ cms_group_id             (toàn bộ 96 mức khác 0 đều 1:1)
```

`adgroup_id` **quyết định** năm cột còn lại. Khi đã biết `adgroup_id`, chúng không
thêm bất kỳ thông tin nào; bảng độ phân tán CTR ở trên cho thấy chính xác điều này
(0.0532 / 0.0505 / 0.0484 cho `adgroup_id` / `campaign_id` / `customer` — cùng một
tín hiệu, nhưng ngày càng thô hơn).

Cặp phía người dùng cũng hoạt động tương tự: mỗi trong số 96 nhóm nhỏ `cms_segid`
thực sự nằm gọn trong đúng một trong 13 nhóm `cms_group_id`, nên `cms_group_id` là
một phiên bản thô hơn của `cms_segid` và không thêm gì một khi cột chi tiết hơn đã
được encode. Điểm cần lưu ý duy nhất là giá trị sentinel — 55.6% số dòng có
`cms_segid = 0` (unknown) nhưng vẫn mang một `cms_group_id` thật, nên cột thô hơn là
tín hiệu phân khúc người dùng *duy nhất* có sẵn trên đa số các dòng. Giữ cả hai:
`cms_group_id` đóng vai trò dự phòng (backoff) đúng cho những dòng mà `cms_segid`
bị thiếu.

Tệ hơn, `campaign_id` gần như vô dụng ở đây: bộ lọc top-300 để lại 248 campaign cho
300 quảng cáo, và **217 campaign chỉ chứa đúng một quảng cáo**. One-hot encode cả hai
cột gần như là mã hóa cùng một cột hai lần.

**Thay đổi.** Với một mô hình tuyến tính, encode `adgroup_id` và chỉ giữ
`cate_id`/`brand` như phương án dự phòng cho các quảng cáo có lượt hiển thị thấp;
bỏ `campaign_id` và `customer`. Lưu ý notebook baseline loại các cột này với một lý
do nay đã lỗi thời — nó nói rằng đây là "các ID gần như duy nhất (hàng chục nghìn
danh mục)", nhưng sau khi lọc top-300 thì chỉ còn 248 và 213. Lý do đúng là sự dư
thừa, không phải cardinality.

---

## 7. Encoding

Kế hoạch tổng quát nói nên target/frequency-encode mọi cột có trên 20 mức. Có hai
ngoại lệ:

- **`hour` (24 mức) → không phải cardinality cao.** Nên one-hot, hoặc tốt hơn, encode
  theo chu kỳ (`sin`/`cos` của `2πh/24`) để giờ 23 nằm cạnh giờ 0. Target-encode một
  đặc trưng có tính chu kỳ sẽ loại bỏ thứ tự mà không mang lại lợi ích gì.
- **`adgroup_id` (300 mức) → one-hot là ổn và đó cũng là cách baseline đang làm.** Bộ
  lọc top-300 được áp dụng chính xác là để việc này khả thi: mỗi quảng cáo có trung
  bình ~800 lượt hiển thị, đủ để học hệ số riêng cho nó. Target-encode nó sẽ cần
  fitting out-of-fold (mã hóa của mỗi dòng chỉ được tính từ các dòng ở fold *khác*,
  nên nó không bao giờ thấy nhãn của chính mình) để tránh rò rỉ dữ liệu — thêm phức
  tạp mà không có lợi ích rõ ràng ở mức cardinality này.

Các ứng viên thực sự phù hợp cho target/frequency-encoding: không còn cột nào, sau
khi `campaign_id` và `customer` đã bị loại vì dư thừa.

Số liệu "hạng mục hiếm" (rare categories) trong bảng tự động sinh (ví dụ:
"`adgroup_id`: 293 rare") chỉ là hệ quả của ngưỡng tỷ lệ <1% — với 300 mức có phân bố
tương đối cân bằng, gần như mọi mức đều dưới 1%. Đây không phải là các hạng mục hiếm
và không nên bị gộp vào nhóm "Other".

---

## 8. Tính nhất quán giữa train/test

**Không có mức nào chưa từng thấy (unseen levels).** Tất cả các cột `adgroup_id`,
`cate_id`, `campaign_id`, `customer`, `brand`, `pid`, `cms_segid` đều xuất hiện trong
test với **không** một mức nào chưa từng thấy — điều này được đảm bảo bởi danh mục
cố định 300 quảng cáo. (Test chỉ khai thác 286 trong số 300 quảng cáo; 14 quảng cáo
còn lại đơn giản là không nhận được lượt hiển thị nào trong cửa sổ 1.5 ngày.)

**Nhưng cơ cấu quảng cáo (ad mix) có thay đổi, và pha phân tích tổng quát bỏ lỡ điều
này.** Pha đó chỉ tính PSI trên các biến số, nên nó chỉ thấy `price` có PSI 0.0486 và
báo cáo "nhỏ" (small). Sự dịch chuyển thực sự nằm ở các biến phân loại. Khoảng cách
biến thiên toàn phần (Total Variation Distance – TVD), so sánh train và test:

| Đặc trưng | TVD | Mức độ |
|---|---|---|
| `adgroup_id` | **0.277** | lớn |
| `cate_id` | 0.087 | vừa phải |
| `age_level` | 0.031 | không đáng kể |
| `pid` | 0.024 | không đáng kể |
| `final_gender_code` | 0.006 | không đáng kể |

![Train/test distribution shift](eda_out/report_figures/fig_train_test_shift.png)

> **📘 Khái niệm — TVD và PSI (hai cách đo độ dịch chuyển).** Cả hai trả lời cùng một
> câu hỏi: *phân phối của một cột dịch chuyển bao xa giữa train và test?* **Total
> variation distance (TVD)** chạy từ 0 → 1: đó là tỷ lệ khối xác suất bạn phải dịch đi
> để biến histogram của train thành của test — 0 nghĩa là giống hệt, 1 nghĩa là không
> chồng lấn. Vậy con số **0.277** của `adgroup_id` cho biết ~28% lượt hiển thị rơi vào
> một cơ cấu quảng cáo khác với những gì train dự đoán — một mức dịch chuyển lớn.
> **PSI** (population stability index) hỏi đúng câu đó cho một cột *dạng số* qua tỷ số
> log theo từng khoảng (bin); pha phân tích tổng quát chỉ chạy PSI trên các biến số,
> nên nó gắn cờ `price` nhưng bỏ lỡ sự dịch chuyển lớn hơn nhiều ở cơ cấu quảng cáo
> dạng phân loại.

Quảng cáo nào được phục vụ (served) thay đổi đáng kể giữa hai cửa sổ thời gian —
quảng cáo được phục vụ nhiều nhất trong train (`710164`, 4.0% số lượt hiển thị) giảm
xuống còn 2.0% trong test, trong khi `836889` tăng lên 3.9%. `price` cũng biến động
theo: train có mean ¥204.84 / trung vị ¥138 so với test có mean ¥230.71 / trung vị
¥162. Vì §5 cho thấy `adgroup_id` là đặc trưng mạnh nhất, sự dịch chuyển về *quảng
cáo nào* chiếm ưu thế chính là sự dịch chuyển quan trọng nhất ở đây.

**Hệ quả.** Nên kỳ vọng điểm số trên test thấp hơn trên validation ngay cả khi phép
chia thời gian là đúng, và nên đọc khoảng cách đó như là dấu hiệu của sự dịch chuyển
cơ cấu (mix shift) trước khi nghĩ đến overfitting. Phía người dùng ổn định, nên đây
là một thay đổi trong cách phục vụ quảng cáo, không phải thay đổi về dân số người
dùng.
**Cách kiểm chứng:** tính điểm theo từng `adgroup_id` và xác nhận rằng loss tập
trung ở các quảng cáo có tỷ lệ lượt hiển thị thay đổi, chứ không dàn trải đều.

Còn hai vấn đề nữa:

**`date` không thể dùng làm đặc trưng.** Train bao phủ 05-05…05-12, test bao phủ
05-12…05-13. `2017-05-13` không bao giờ xuất hiện trong train, nên cột one-hot của
nó luôn bằng 0 tại thời điểm fit và mô hình không có hệ số nào cho phần lớn các ngày
trong test. Bỏ `date`; `hour` và `weekday` đã mang đủ tín hiệu thời gian hữu ích.
(Baseline đã làm đúng điều này.) Lưu ý `date` là cột *duy nhất* mà pha phân tích
train/test gắn cờ có mức chưa từng thấy — mọi hạng mục phía quảng cáo và phía người
dùng đều được bao phủ đầy đủ.

**Việc chia tập validation phải theo thời gian, không phải ngẫu nhiên.** Tập test là
một holdout tương lai nghiêm ngặt (mọi lượt hiển thị trong test đều xảy ra sau mọi
lượt hiển thị trong train), nên một `KFold` ngẫu nhiên trên train sẽ để mô hình nhìn
thấy các lượt hiển thị tương lai của cùng một quảng cáo và cho ra điểm số lạc quan
giả tạo. Nên dùng `TimeSeriesSplit` hoặc một mốc cắt cố định — ví dụ giữ lại
05-11…05-12 làm validation — để việc validate phản ánh đúng phép chia thực tế.

Lưu ý rằng 20,508 trong số 47,608 người dùng ở test cũng xuất hiện trong train. Đây
không phải là rò rỉ dữ liệu (leakage) — đó là cùng những người dùng quay lại vào một
ngày sau, đúng như bối cảnh thực tế khi triển khai production.

---

## 9. Encoding đặc trưng & lựa chọn đặc trưng — kiểm chứng thực nghiệm (`Nguyen-tasks.ipynb`)

§7 lập luận chỉ dựa trên cardinality và sự dư thừa rằng nên bỏ `campaign_id`/
`customer`, và một khi đã bỏ thì không còn ứng viên thực sự nào cho target/frequency
encoding. `Nguyen-tasks.ipynb` kiểm chứng lập luận đó trực tiếp: nó encode cả sáu
cột **ID định danh (Nominal ID)** ở §1 (`adgroup_id`, `cate_id`, `campaign_id`,
`customer`, `brand`, `cms_segid`) theo bốn cách — one-hot, count, frequency, target
(out-of-fold, có làm mượt/smoothing) — fit cùng một mô hình logistic regression
trên mỗi cách, và chạy một quy trình chọn đặc trưng tham lam tiến (greedy forward
selection) dựa trên AUC trên tập test. **Base** trong cả hai bảng dưới đây = `price`
(đã scale) + 10 biến phân loại không phải ID (`cms_group_id`, `final_gender_code`,
`age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level`,
`pid`, `hour`, `weekday`), tất cả one-hot — giữ cố định ở mọi dòng.

**Bảng A — cách encode nào thắng.**

| Đặc trưng | AUC train | AUC test | Ghi chú |
|---|---|---|---|
| `adgroup_id` + `cate_id` (định danh sản phẩm, one-hot) + base | 0.5929 | 0.5759 | baseline |
| count-encoded: cả 6 cột ID + base | 0.5385 | 0.5291 | kém hơn baseline |
| freq-encoded: cả 6 cột ID + base | 0.5385 | 0.5291 | giống hệt count (r = 1.0) |
| target-encoded: cả 6 cột ID + base | 0.5853 | 0.5753 | ≈ baseline |
| target-encoded: cả 6 cột ID + tương tác `adgroup_id × cms_group_id` + base | 0.5859 | 0.5760 | ≈ baseline, tốt nhất trong 5 |

Count và frequency encoding là *cùng một con số trên hai thang đo khác nhau*
(`freq = count / len(train)`) — sau `StandardScaler` chúng khớp mô hình giống hệt
nhau, đó là lý do AUC của chúng bằng nhau tuyệt đối. Cả hai đạt điểm rõ ràng thấp
hơn baseline: một count chỉ encode "quảng cáo này được phục vụ bao nhiêu lần," và
§8 đã cho thấy đó chính là thứ dịch chuyển giữa train và test (`adgroup_id` TVD
0.277) — count/frequency encoding vô tình nướng luôn sự dịch chuyển đó vào đặc
trưng. Target encoding, vốn encode "hạng mục này có CTR bao nhiêu" thay vì "hạng
mục này lớn cỡ nào," lại rơi vào trong biên độ nhiễu so với baseline one-hot.

**Bảng B — đặc trưng riêng lẻ nào thực sự xứng đáng.** Chọn đặc trưng tham lam
tiến, dùng target encoding, thử các ứng viên theo thứ tự độ mạnh tín hiệu ở §5, chỉ
giữ lại một đặc trưng nếu AUC trên test cải thiện so với mức tốt nhất hiện tại:

| Bước | Đặc trưng thêm vào | AUC train | AUC test | Kết luận |
|---|---|---|---|---|
| start | *(không có gì — chỉ base, chưa có định danh ad/sản phẩm)* | 0.5313 | 0.5197 | mốc tham chiếu |
| 1 | `adgroup_id` (định danh quảng cáo) | 0.5853 | 0.5752 | **GIỮ** (+0.0554) |
| 2 | `campaign_id` (chiến dịch của quảng cáo) | 0.5853 | 0.5753 | GIỮ (+0.0001) |
| 3 | `customer` (nhà quảng cáo) | 0.5853 | 0.5753 | GIỮ (+0.00005) |
| 4 | `brand` | 0.5853 | 0.5753 | loại (−0.0001) |
| 5 | `cate_id` (danh mục quảng cáo) | 0.5853 | 0.5753 | loại (≈0) |
| 6 | `cms_segid` (phân khúc nhỏ người dùng) | 0.5853 | 0.5756 | GIỮ (+0.0003) |
| 7 | tương tác `adgroup_id × cms_group_id` | 0.5859 | 0.5761 | GIỮ (+0.0005) |
| 8 | tương tác `adgroup_id × cate_id` | 0.5859 | 0.5762 | GIỮ (+0.00001) |

Tập đặc trưng ở mỗi bước là tích lũy — bao gồm mọi đặc trưng đã được **GIỮ** trước
đó cộng với đặc trưng đang được thử ở dòng đó. "start" không phải một lần thử; đó
là mốc tham chiếu mà mọi dòng sau được đo so với nó (0.5197 của nó là mức mà
`adgroup_id` một mình phải vượt qua ở bước 1; một khi `adgroup_id` được giữ, chính
0.5752 của nó trở thành mốc tham chiếu mới cho bước 2, và cứ thế tiếp diễn).

Độ lớn các con số tái hiện lại §5 và §6 theo cách định lượng, không chỉ định tính:
một mình `adgroup_id` (+0.0554) áp đảo mọi đóng góp khác cộng lại, khớp với vị trí
dẫn đầu 0.0532 SD-có-trọng-số của nó ở §5. Độ chênh lệch của `cate_id` gần 0 nhất
trong cả 8 ứng viên — đúng như kỳ vọng, vì §6 đã cho thấy `adgroup_id` quyết định
`cate_id` một cách chính xác (được kiểm chứng lại ở đây: mã hóa target của chúng
tương quan ở mức 1.0). `campaign_id` và `customer` về mặt kỹ thuật được giữ nhưng
chênh lệch nhỏ hơn cả sai số làm tròn, phù hợp với nhận định "gần như một bản sao
của `adgroup_id`" ở §6 chứ không mâu thuẫn với nó. Tương tác `adgroup_id ×
cms_group_id` là mức tăng lớn thứ ba trong toàn bộ chuỗi thử (chỉ sau `adgroup_id`)
— bằng chứng trực tiếp cho khẳng định ở §5 rằng dư địa còn lại nằm ở các tương tác
quảng cáo × phân khúc người dùng, không phải thêm cột ID.

**Lưu ý.** Các bước 2, 3, và 8 giữ lại một đặc trưng dựa trên ngưỡng `delta > 0` đo
trên một lần chia train/test duy nhất — ở quy mô đó (0.00001–0.0001), nhiễu lấy mẫu
của chính chỉ số này có thể quyết định kết quả. Chỉ `adgroup_id` và tương tác
`cms_group_id` của nó đạt mức tăng đủ lớn để tin tưởng mà không cần cross-validation.

---

## Kế hoạch làm sạch & biến đổi dữ liệu

Fit trên train, sau đó áp dụng cho validation/test.

1. **Nạp dữ liệu kèm guard kiểm tra timestamp** (§0) — lỗi Excel-mangling là lỗi âm
   thầm.
2. **Bỏ** `userid`, `time_stamp`, `date` (§1, §8); **bỏ** `campaign_id`,
   `customer` vì dư thừa về mặt hàm số so với `adgroup_id` (§6).
3. **Giải mã `0` → "unknown"** ở các cột `brand`, `cms_segid`, `pvalue_level`,
   `new_user_class_level`, và giữ nó như một hạng mục. Không impute, không thêm chỉ
   báo missing (§3).
4. **`price`**: `log1p`, sau đó `StandardScaler` (§4).
5. **One-hot**: `adgroup_id`, `cate_id`, `brand`, `pid`, `cms_segid`, `cms_group_id`,
   `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`,
   `new_user_class_level`, `weekday`, với `handle_unknown="ignore"` (§7).
6. **`hour`**: encode theo chu kỳ bằng `sin`/`cos` (§7).
7. **Validation**: chia theo thời gian, không dùng `KFold` ngẫu nhiên (§8).
8. **Sau đó** thêm các tương tác chéo giữa `adgroup_id`/`cate_id` với phân khúc
   người dùng, hoặc chuyển sang mô hình tree/FM — §5 chỉ ra đây mới là nơi còn dư
   địa cải thiện, không phải ở việc tiền xử lý thêm.

Xây dựng các bước 3–6 dưới dạng một `ColumnTransformer` của sklearn để đảm bảo không
rò rỉ dữ liệu và có thể tái lập; `eda_scripts/transform.py` có một cài đặt có thể
cấu hình được.

---

## Cách tái lập (Reproducing)

```bash
.venv/Scripts/python.exe eda_scripts/eda_taobao.py \
    --train dataset/train.csv --test dataset/test.csv --out eda_out
# inline report figures:
.venv/Scripts/python.exe eda_scripts/report_figures.py \
    --train dataset/train.csv --test dataset/test.csv
```

| File | Mục đích |
|---|---|
| `eda_scripts/prep_taobao.py` | Vai trò các cột + giải mã sentinel cho bộ dữ liệu này |
| `eda_scripts/eda_taobao.py` | Chạy các pha phân tích in-process trên khung dữ liệu đã chuẩn bị |
| `eda_scripts/report_figures.py` | Các hình minh họa trong báo cáo này (§2–§8) |
| `eda_scripts/verify_docs.py` | Tính lại mọi con số quan trọng trong README / Data Foundation |
| `eda_scripts/*.py` | Các script pha phân tích tổng quát từ skill eda-analysis |
| `eda_out/report_generated.md` | Bản nháp tự động sinh |
| `eda_out/findings.json` | Các phát hiện có cấu trúc |
| `eda_out/report_figures/` | Các hình minh họa nhúng cho báo cáo này |
| `eda_out/plots/` | Các hình minh họa của pha phân tích tổng quát |
| `Nguyen-tasks.ipynb` | So sánh các cách encoding + chọn đặc trưng tham lam (§9) |

Cần hai chỉnh sửa trên các script tổng quát của skill để chạy được trên
matplotlib 3.11 / Windows: `boxplot(labels=)` → `tick_labels=` trong `numerical.py`
và `relationships.py`, và mã hóa UTF-8 khi ghi báo cáo trong `eda_report.py`.
