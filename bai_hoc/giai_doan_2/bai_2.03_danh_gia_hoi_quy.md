# Bài 2.3 – Đánh giá hồi quy: MAE, MSE, RMSE, R²

> **Giai đoạn 2 – Machine Learning cổ điển** · Thời lượng: 1 buổi (2–3 giờ) · Hình thức mở đầu: *Mini-game đoán kết quả*

## 🎯 Sau bài này bạn sẽ

- Tự tính tay được MAE, MSE, RMSE, R² và hiểu mỗi con số trả lời câu hỏi gì.
- Biết chỉ số nào **nhạy với outlier**, chỉ số nào dễ giải thích cho sếp.
- Hiểu vì sao R² có thể **âm**.
- Đánh giá đúng cách: so với mốc, so train với test, nhìn cả phân bố sai số.

---

## 1. Mini-game: đoán kết quả 🎲

Hai mô hình dự đoán giá 5 căn hộ (tỷ đồng):

| Căn | Giá thật | Mô hình A | Mô hình B |
|---|---|---|---|
| 1 | 3,0 | 2,5 | 3,0 |
| 2 | 5,0 | 5,0 | 5,0 |
| 3 | 2,5 | 3,0 | 2,5 |
| 4 | 7,0 | 8,0 | 7,0 |
| 5 | 4,5 | 4,0 | **9,5** |

- Mô hình A: sai **một chút ở hầu hết** các căn.
- Mô hình B: đoán **đúng tuyệt đối 4 căn**, nhưng sai **5 tỷ** ở căn cuối.

**Đoán:**
1. Tính theo "sai trung bình mỗi căn bao nhiêu tỷ", mô hình nào tốt hơn?
2. Có cách đo nào khiến mô hình B trông **tệ hơn cả việc đoán bừa** không?

Ghi đáp án. Câu trả lời ở mục 4.

---

## 2. Bốn chỉ số

Ký hiệu chung: yᵢ là giá thật, ŷᵢ là giá dự đoán, n là số mẫu, eᵢ = ŷᵢ − yᵢ là sai số.

### 2.1. MAE – Mean Absolute Error (Sai số tuyệt đối trung bình)

- **Định nghĩa:** Trung bình độ lớn sai số, không quan tâm sai lên hay sai xuống.
- **Ví dụ đời thực:** "Trung bình mỗi lần dự đoán, app giao hàng báo lệch giờ đến khoảng 7 phút."

```
         1   n
  MAE = ───  Σ  |yᵢ − ŷᵢ|
         n  i=1
```

### 2.2. MSE – Mean Squared Error (Sai số bình phương trung bình)

- **Định nghĩa:** Trung bình **bình phương** sai số. Chính là loss của Linear Regression (bài 2.2).
- **Đơn vị:** bình phương đơn vị gốc (tỷ²), khó hình dung.

```
         1   n
  MSE = ───  Σ  (yᵢ − ŷᵢ)²
         n  i=1
```

### 2.3. RMSE – Root Mean Squared Error (Căn bậc hai của MSE)

- **Định nghĩa:** √MSE. Đưa đơn vị về như gốc (tỷ), nhưng vẫn **phạt nặng lỗi lớn**.

```
  RMSE = √MSE
```

### 2.4. R² – Coefficient of Determination (Hệ số xác định)

- **Định nghĩa:** Mô hình giải thích được bao nhiêu phần trăm sự **biến động** của y, so với cách đoán ngây thơ "luôn đoán trung bình".
- **Ví dụ đời thực:** "Thu nhập giải thích được 40% sự khác biệt giá nhà giữa các khu. 60% còn lại do yếu tố khác."

```
              SS_res          Σ (yᵢ − ŷᵢ)²       ← tổng bình phương sai số của MÔ HÌNH
  R² = 1 − ─────────  = 1 − ──────────────
              SS_tot          Σ (yᵢ − ȳ)²        ← tổng bình phương sai số của MỐC "đoán trung bình"
```

| R² | Nghĩa |
|---|---|
| 1 | Hoàn hảo |
| 0 | Chỉ ngang việc luôn đoán trung bình |
| **< 0** | **Tệ hơn** cả việc luôn đoán trung bình |

---

## 3. Tính tay với mô hình A

| Căn | y | ŷ (A) | e = ŷ − y | \|e\| | e² | y − ȳ | (y − ȳ)² |
|---|---|---|---|---|---|---|---|
| 1 | 3,0 | 2,5 | −0,5 | 0,5 | 0,25 | −1,4 | 1,96 |
| 2 | 5,0 | 5,0 | 0 | 0 | 0 | 0,6 | 0,36 |
| 3 | 2,5 | 3,0 | 0,5 | 0,5 | 0,25 | −1,9 | 3,61 |
| 4 | 7,0 | 8,0 | 1,0 | 1,0 | 1,00 | 2,6 | 6,76 |
| 5 | 4,5 | 4,0 | −0,5 | 0,5 | 0,25 | 0,1 | 0,01 |
| | ȳ = 4,4 | | **Tổng** | **2,5** | **1,75** | | **12,7** |

- MAE = 2,5 / 5 = **0,5 tỷ**
- MSE = 1,75 / 5 = **0,35 tỷ²**
- RMSE = √0,35 ≈ **0,5916 tỷ**
- R² = 1 − 1,75 / 12,7 = 1 − 0,1378 ≈ **0,8622**

```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)

y_that = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
du_doan_A = np.array([2.5, 5.0, 3.0, 8.0, 4.0])

print(mean_absolute_error(y_that, du_doan_A),               # MAE
      mean_squared_error(y_that, du_doan_A),                # MSE
      round(root_mean_squared_error(y_that, du_doan_A), 4), # RMSE
      round(r2_score(y_that, du_doan_A), 4))                # R²
```

**Kết quả mẫu:**

```
0.5 0.35 0.5916 0.8622
```

> Lưu ý: `root_mean_squared_error` có từ Scikit-learn 1.4. Bản cũ hơn dùng `np.sqrt(mean_squared_error(...))`.

---

## 4. Giải mini-game

```python
du_doan_B = np.array([3.0, 5.0, 2.5, 7.0, 9.5])
print(mean_absolute_error(y_that, du_doan_B),
      round(root_mean_squared_error(y_that, du_doan_B), 4),
      round(r2_score(y_that, du_doan_B), 4))
```

**Kết quả mẫu:**

```
1.0 2.2361 -0.9685
```

| Chỉ số | Mô hình A | Mô hình B | Ai tốt hơn? |
|---|---|---|---|
| MAE | 0,5 | 1,0 | A (gấp đôi) |
| RMSE | 0,59 | 2,24 | A (gấp gần **4 lần**) |
| R² | 0,86 | **−0,97** | B **tệ hơn cả đoán trung bình**! |

**Câu 1:** Theo MAE, A tốt hơn (0,5 so với 1,0).

**Câu 2:** Có! R² của B **âm**. Một lỗi 5 tỷ, khi bình phương thành 25, lớn hơn cả tổng SS_tot = 12,7 của cách đoán trung bình.

**Bài học:**
- **MAE** đối xử mọi lỗi như nhau. Sai 5 tỷ một lần = sai 1 tỷ năm lần.
- **RMSE / MSE / R²** phạt **cực nặng** lỗi lớn.
- **Khoảng cách giữa RMSE và MAE** cho biết có lỗi lớn hay không. RMSE ≈ MAE: lỗi đều đặn. RMSE ≫ MAE: có vài lỗi rất lớn.

### Kiểm tra định nghĩa R² = 0 và R² âm

```python
print(r2_score(y_that, np.full(5, y_that.mean())),         # luôn đoán trung bình 4.4
      round(r2_score(y_that, np.full(5, 10.0)), 3))        # luôn đoán 10 tỷ
```

**Kết quả mẫu:**

```
0.0 -12.346
```

---

## 5. Chọn chỉ số nào?

| Chỉ số | Đơn vị | Nhạy outlier | Dùng khi |
|---|---|---|---|
| **MAE** | Như gốc | Ít | Báo cáo cho người không chuyên; khi mọi lỗi đáng sợ như nhau |
| **MSE** | Gốc² | Rất nhạy | Làm **loss** để huấn luyện (đạo hàm đẹp), ít dùng để báo cáo |
| **RMSE** | Như gốc | Nhạy | Khi lỗi lớn **đặc biệt tai hại** (dự báo nhu cầu điện: thiếu nhiều gây mất điện) |
| **R²** | Không đơn vị | Nhạy | So sánh mô hình **trên cùng dữ liệu**; nói "giải thích được bao nhiêu %" |
| **MAPE** | % | | Khi sai số **tương đối** quan trọng hơn tuyệt đối |

### MAPE – Mean Absolute Percentage Error (Sai số phần trăm tuyệt đối trung bình)

- **Định nghĩa:** Trung bình của |sai số| / |giá thật|. Sai 100 triệu với căn 1 tỷ (10%) khác hẳn sai 100 triệu với căn 10 tỷ (1%).
- **Cảnh báo:** Nếu có y thật bằng hoặc gần 0, MAPE bùng nổ.

```python
from sklearn.metrics import mean_absolute_percentage_error
print(round(mean_absolute_percentage_error(y_that, du_doan_A), 4))   # 0.1241 = 12,41%
```

**Kết quả mẫu:**

```
0.1241
```

> 💡 **R² không so sánh được giữa các dataset khác nhau.** R² = 0,9 trên dữ liệu A và 0,5 trên dữ liệu B không có nghĩa mô hình trên A "tốt hơn": có thể y của B vốn khó đoán hơn nhiều.

---

## 6. Đánh giá mô hình giá nhà một cách đầy đủ

Dùng ô `nap_du_lieu_nha()` ở bài 2.1.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

df = nap_du_lieu_nha()
cot = ["thu_nhap", "km_tp_lon", "phong_moi_ho", "ti_le_phong_ngu", "nguoi_moi_ho", "tuoi_nha"]
Xtr, Xte, ytr, yte = train_test_split(df[cot], df["gia_nha"], test_size=0.2, random_state=42)

def danh_gia(ten, y, p):
    """In 4 chỉ số trên một dòng để dễ so sánh."""
    print(f"{ten:<12} MAE={mean_absolute_error(y, p):6.1f}  RMSE={root_mean_squared_error(y, p):6.1f}"
          f"  R²={r2_score(y, p):6.3f}  MAPE={mean_absolute_percentage_error(y, p):.1%}")

danh_gia("Mốc", yte, DummyRegressor().fit(Xtr, ytr).predict(Xte))
danh_gia("Thu nhập", yte, LinearRegression().fit(Xtr[["thu_nhap"]], ytr).predict(Xte[["thu_nhap"]]))
m = LinearRegression().fit(Xtr, ytr)
danh_gia("6 đặc trưng", yte, m.predict(Xte))
danh_gia("Train", ytr, m.predict(Xtr))           # đánh giá trên chính dữ liệu đã học
```

**Kết quả mẫu:**

```
Mốc          MAE=  77.6  RMSE=  96.8  R²=-0.000  MAPE=57.5%
Thu nhập     MAE=  56.8  RMSE=  74.9  R²= 0.401  MAPE=38.0%
6 đặc trưng  MAE=  48.7  RMSE=  65.9  R²= 0.537  MAPE=30.4%
Train        MAE=  49.1  RMSE=  65.9  R²= 0.550  MAPE=30.3%
```

**Đọc kết quả, theo 3 câu hỏi:**

**① Có hơn mốc không?** Có. MAE giảm từ 77,6 xuống 48,7 (giảm 37%).

**② Train và test có chênh lệch lớn không?** R² train 0,550, test 0,537: rất gần nhau. Mô hình **không học vẹt**. Nhưng cả hai đều chỉ khoảng 0,55: mô hình **quá đơn giản**, chưa nắm được quy luật (gọi là **underfitting**, sẽ học kỹ ở bài 2.6).

**③ Giải thích cho sếp thế nào?** "Trung bình mỗi dự đoán lệch khoảng **49.000 USD**, tức khoảng **30%** giá nhà."

### Nhìn sâu hơn con số trung bình

```python
p = m.predict(Xte)
sai = np.abs(yte - p)                                  # sai số tuyệt đối từng khu
print(round(np.median(sai), 1), round(np.percentile(sai, 90), 1))
```

**Kết quả mẫu:**

```
38.4 97.6
```

- Một nửa số khu bị lệch dưới **38.400 USD**.
- Nhưng 10% số khu bị lệch trên **97.600 USD**.

Một con số trung bình che giấu chuyện mô hình **rất tệ với một nhóm khu nhất định**. Tìm ra nhóm đó là gì (ven biển? vùng núi?) là cách hay nhất để cải thiện mô hình.

---

## 7. Bài tập

1. **Tính tay** MAE, RMSE, R² cho: y thật = [10, 20, 30], ŷ = [12, 18, 33].
2. Một mô hình dự báo số ly cà phê bán mỗi ngày có MAE = 20 ly, RMSE = 60 ly. Bạn đoán điều gì về các lỗi của mô hình?
3. Với mô hình 6 đặc trưng ở mục 6, tìm 5 khu có sai số tuyệt đối lớn nhất. Chúng có điểm chung gì? (Gợi ý: in thêm `km_tp_lon`, `thu_nhap`, giá thật và giá dự đoán.)
4. Vì sao không nên dùng MAPE cho bài toán dự đoán **lợi nhuận** của cửa hàng?

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" → `Turn into → Toggle heading 3`.

1. Sai số: 2, −2, 3. MAE = (2 + 2 + 3) / 3 ≈ **2,333**. MSE = (4 + 4 + 9) / 3 ≈ 5,667 → RMSE ≈ **2,380**. ȳ = 20, SS_tot = 100 + 0 + 100 = 200, SS_res = 17 → R² = 1 − 17/200 = **0,915**.
2. RMSE gấp 3 lần MAE → phần lớn ngày dự đoán khá sát, nhưng có **vài ngày sai rất nhiều** (có thể là ngày lễ, ngày mưa bão, ngày có sự kiện). Nên tìm những ngày đó và thêm đặc trưng "ngày lễ", "thời tiết".
3. Code:
```python
ket_qua = Xte.copy()
ket_qua["gia_that"] = yte
ket_qua["du_doan"] = p.round(1)
ket_qua["sai"] = np.abs(yte - p).round(1)
print(ket_qua.sort_values("sai", ascending=False).head(5)[["thu_nhap", "km_tp_lon", "gia_that", "du_doan", "sai"]])
```
   Thường là các khu **giá thật rất cao** (gần 500) nhưng mô hình đoán thấp, hoặc ngược lại, các khu có đặc trưng bất thường (ví dụ `phong_moi_ho`, `nguoi_moi_ho` cực lớn, bài 1.12). Mô hình đường thẳng không xử lý được những trường hợp ngoài quy luật chung.
4. Lợi nhuận có thể **bằng 0 hoặc âm**. Chia cho 0 hoặc số rất gần 0 làm MAPE bùng nổ thành con số vô nghĩa.

---

## 8. Flashcard

| Mặt trước | Mặt sau |
|---|---|
| MAE là gì? | Trung bình \|sai số\|. Cùng đơn vị, dễ giải thích. |
| MSE vs RMSE? | RMSE = √MSE, đưa về đơn vị gốc. Cả hai phạt nặng lỗi lớn. |
| RMSE ≫ MAE nói lên điều gì? | Có một số lỗi rất lớn. |
| Công thức R²? | 1 − SS_res / SS_tot |
| R² = 0 nghĩa là? | Ngang với luôn đoán trung bình. |
| R² âm được không? | Được: tệ hơn cả luôn đoán trung bình. |
| Khi nào tránh MAPE? | Khi y thật có thể bằng 0 hoặc gần 0. |
| 3 câu hỏi khi đánh giá? | Hơn mốc chưa? Train vs test chênh không? Giải thích cho người dùng thế nào? |
| R² train ≈ R² test nhưng đều thấp nghĩa là? | Mô hình quá đơn giản (underfitting). |

**Bài tiếp theo:** 2.4 – Logistic Regression.
