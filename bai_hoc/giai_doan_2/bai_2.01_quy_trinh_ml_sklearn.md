# Bài 2.1 – Supervised vs Unsupervised, quy trình dự án ML, nhập môn Scikit-learn

> **Giai đoạn 2 – Machine Learning cổ điển** · Thời lượng: 1 buổi (2–3 giờ) · Hình thức mở đầu: *Câu chuyện*

## 🎯 Sau bài này bạn sẽ

- Phân biệt chắc chắn học có giám sát và không giám sát, biết bài toán nào thuộc loại nào.
- Nắm **quy trình 7 bước** của một dự án ML, dùng cho mọi bài sau.
- Hiểu "ngôn ngữ chung" của Scikit-learn: `fit`, `predict`, `transform`, `score`.
- Huấn luyện mô hình dự đoán giá nhà đầu tiên, và biết **so với mốc** để đánh giá.

---

## 1. Câu chuyện: Hai nhiệm vụ của chị Lan

Chị Lan là trưởng phòng dữ liệu của một chuỗi siêu thị. Sáng thứ Hai, chị nhận hai nhiệm vụ:

**Nhiệm vụ 1 (từ phòng Tín dụng):** *"Khách đăng ký thẻ thành viên trả góp. Dự đoán giúp anh khách nào sẽ trả nợ đúng hạn."*
→ Chị có lịch sử 10.000 khách cũ, **mỗi khách đã biết kết quả** (trả đúng hạn hay không).

**Nhiệm vụ 2 (từ phòng Marketing):** *"Chia khách hàng thành vài nhóm để làm khuyến mãi riêng. Em tự xem nên chia mấy nhóm."*
→ Chị có dữ liệu mua sắm của 50.000 khách, **không có "đáp án" nào cả**. Không ai biết trước có những nhóm nào.

Chị Lan mỉm cười: hai nhiệm vụ này thuộc hai "thế giới" khác nhau của ML. Nhiệm vụ 1 là **học có giám sát**. Nhiệm vụ 2 là **học không giám sát**. Công cụ, cách đánh giá, cách nghĩ đều khác.

---

## 2. Hai thế giới của ML

### 2.1. Supervised Learning (Học có giám sát)

- **Định nghĩa:** Học từ dữ liệu **có nhãn** (có đáp án), để dự đoán nhãn cho dữ liệu mới.
- **Ví dụ đời thực:** Học sinh luyện đề có đáp án.

```
 Dữ liệu đặc trưng X          Nhãn y (đáp án)
 [tuổi, thu nhập, nợ...]  →   trả đúng hạn / không
           │                         │
           └───────► MÔ HÌNH ◄───────┘  (học quan hệ X → y)
                        │
 Khách mới [X] ─────────┴────► dự đoán y
```

Hai dạng (đã gặp ở bài 0.1):

| Dạng | Nhãn y là | Ví dụ | Học ở bài |
|---|---|---|---|
| **Regression (Hồi quy)** | Con số liên tục | Giá nhà, doanh thu, nhiệt độ | 2.2, 2.3 |
| **Classification (Phân loại)** | Một nhóm trong số ít nhóm | Spam/không, trả nợ/không, loại hoa | 2.4, 2.5 |

### 2.2. Unsupervised Learning (Học không giám sát)

- **Định nghĩa:** Tìm cấu trúc ẩn trong dữ liệu **không có nhãn**.
- **Ví dụ đời thực:** Đổ một túi Lego lẫn lộn ra bàn, tự xếp thành đống theo hình dạng, màu sắc.

| Dạng | Làm gì | Ví dụ | Học ở bài |
|---|---|---|---|
| **Clustering (Phân cụm)** | Gom điểm giống nhau thành nhóm | Phân nhóm khách hàng | 2.13, 2.14 |
| **Dimensionality Reduction (Giảm chiều)** | Nén nhiều cột thành ít cột mà giữ thông tin | Vẽ dữ liệu 30 chiều lên mặt phẳng | 2.15 |

### 2.3. Bảng nhận diện nhanh

| Câu hỏi bạn tự hỏi | Có → | Không → |
|---|---|---|
| Có cột "đáp án" đã biết cho dữ liệu cũ không? | Có giám sát | Không giám sát |
| (Nếu có giám sát) Đáp án là con số liên tục? | Hồi quy | Phân loại |

---

## 3. Quy trình 7 bước của một dự án ML

```
 ① Xác định bài toán ─► ② Thu thập dữ liệu ─► ③ EDA & làm sạch
                                                     │
 ⑦ Triển khai & theo dõi ◄─ ⑥ Đánh giá trên test ◄─ ④ Chia train/val/test
                                  ▲                  │
                                  └── ⑤ Huấn luyện & tinh chỉnh (lặp nhiều lần trên val)
```

| Bước | Câu hỏi chính | Bạn đã/sẽ học ở |
|---|---|---|
| ① Xác định bài toán | Dự đoán cái gì? Hồi quy hay phân loại? **Thế nào là "tốt"?** | Bài này |
| ② Thu thập dữ liệu | Có đủ dữ liệu? Nhãn có đáng tin? | |
| ③ EDA & làm sạch | Dữ liệu có vấn đề gì? | Bài 1.4–1.7, 1.12 |
| ④ Chia dữ liệu | Có xáo trộn? Có rò rỉ? | Bài 1.11 |
| ⑤ Huấn luyện & tinh chỉnh | Mô hình nào? Tham số nào? | Bài 2.2–2.18 |
| ⑥ Đánh giá | Trên test, **một lần**, so với mốc | Bài 2.3, 2.5 |
| ⑦ Triển khai | Đưa vào sản phẩm, theo dõi chất lượng theo thời gian | Phần C |

> 💡 Ngoài đời, bước ① và ③ chiếm phần lớn thời gian. Rất nhiều dự án thất bại vì **giải đúng một bài toán sai**, chứ không phải vì chọn sai thuật toán.

---

## 4. Scikit-learn: một "ngôn ngữ" cho mọi mô hình

### 4.1. Khái niệm

- **Scikit-learn (sklearn):** Thư viện ML cổ điển phổ biến nhất của Python. Có sẵn hàng chục thuật toán, công cụ chia dữ liệu, chuẩn hóa, đánh giá.
- **Điểm mạnh nhất:** mọi thuật toán đều dùng **chung một cách gọi**. Học một lần, dùng được cho tất cả.

### 4.2. Ba loại đối tượng

| Loại | Phương thức | Ví dụ | Đã gặp |
|---|---|---|---|
| **Estimator (Mô hình dự đoán)** | `fit(X, y)` → `predict(X)` | `LinearRegression`, `RandomForestClassifier` | Bài 1.2 (tự viết) |
| **Transformer (Bộ biến đổi)** | `fit(X)` → `transform(X)` | `StandardScaler`, `OneHotEncoder` | Bài 1.11 |
| **Đánh giá** | Hàm `metric(y_that, y_du_doan)` | `mean_absolute_error`, `accuracy_score` | Bài 2.3, 2.5 |

```
   mo_hinh = TenMoHinh(tham_so=...)   # 1. Tạo (chọn thuật toán + cấu hình)
   mo_hinh.fit(X_train, y_train)      # 2. Học từ dữ liệu huấn luyện
   y_du_doan = mo_hinh.predict(X_test)# 3. Dự đoán trên dữ liệu mới
   mo_hinh.score(X_test, y_test)      # 4. Chấm điểm nhanh (R² cho hồi quy, accuracy cho phân loại)
```

Đổi thuật toán = **chỉ đổi dòng 1**. Ba dòng còn lại giữ nguyên.

---

## 5. Dự án ML đầu tiên: dự đoán giá nhà

### 5.1. Chuẩn bị dữ liệu (như bài 1.12)

Ô code này sẽ được dùng lại ở nhiều bài sau. Hãy lưu nó vào Notion.

```python
import numpy as np
import pandas as pd

def nap_du_lieu_nha():
    """Đọc và chuẩn bị dữ liệu giá nhà California giống bài 1.12."""
    url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
    df = pd.read_csv(url)
    df.columns = ["kinh_do", "vi_do", "tuoi_nha", "tong_phong", "tong_phong_ngu",
                  "dan_so", "so_ho", "thu_nhap", "gia_nha"]
    df = df[df["gia_nha"] < 500001].copy()             # bỏ các khu bị giới hạn trần giá
    df["gia_nha"] = df["gia_nha"] / 1000               # đổi sang NGHÌN USD cho số gọn
    df["phong_moi_ho"] = df["tong_phong"] / df["so_ho"]
    df["ti_le_phong_ngu"] = df["tong_phong_ngu"] / df["tong_phong"]
    df["nguoi_moi_ho"] = df["dan_so"] / df["so_ho"]

    def km(v1, k1, v2, k2):                             # khoảng cách xấp xỉ (km), bài 1.12
        return np.sqrt(((k1 - k2) * 111 * np.cos(np.radians(v1))) ** 2 + ((v1 - v2) * 111) ** 2)
    df["km_tp_lon"] = np.minimum(km(df["vi_do"], df["kinh_do"], 34.05, -118.24),    # tới LA
                                 km(df["vi_do"], df["kinh_do"], 37.77, -122.42))    # tới SF
    return df.reset_index(drop=True)

df = nap_du_lieu_nha()
print(df.shape)
```

**Kết quả mẫu:**

```
(16186, 13)
```

### 5.2. Bước ①: xác định bài toán

- Dự đoán `gia_nha` (nghìn USD) → **hồi quy có giám sát**.
- Thế nào là "tốt"? Tạm dùng **MAE (sai số tuyệt đối trung bình)**: trung bình mỗi dự đoán lệch bao nhiêu nghìn USD (sẽ học kỹ ở bài 2.3).

### 5.3. Bước ④: chia dữ liệu

```python
from sklearn.model_selection import train_test_split

X = df[["thu_nhap", "km_tp_lon", "phong_moi_ho", "tuoi_nha"]]   # chọn 4 đặc trưng
y = df["gia_nha"]                                                # cột đích
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
```

**Kết quả mẫu:**

```
(12948, 4) (3238, 4)
```

### 5.4. Mốc so sánh (baseline): luôn có trước mô hình

- **Baseline (Mô hình mốc):** Cách dự đoán "ngây thơ" nhất, không cần thông minh. Mô hình thật phải **tốt hơn mốc** mới có giá trị.
- **Ví dụ đời thực:** Dự báo thời tiết "ngày mai giống hôm nay". Đài khí tượng tốn hàng tỷ mà không hơn được cách này thì vô ích.

Mốc cho hồi quy: **luôn đoán giá trung bình** của tập train.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

moc = DummyRegressor(strategy="mean").fit(X_train, y_train)   # "mô hình" luôn đoán giá trung bình
print("Mốc:", round(mean_absolute_error(y_test, moc.predict(X_test)), 1))
```

**Kết quả mẫu:**

```
Mốc: 77.6
```

Đoán bừa giá trung bình thì sai trung bình **77.600 USD** mỗi khu.

### 5.5. Bước ⑤–⑥: huấn luyện và đánh giá

```python
from sklearn.linear_model import LinearRegression

mo_hinh = LinearRegression()              # 1. tạo mô hình (sẽ học kỹ ở bài 2.2)
mo_hinh.fit(X_train, y_train)             # 2. học
du_doan = mo_hinh.predict(X_test)         # 3. dự đoán
print(du_doan[:3].round(1), y_test.values[:3])   # so 3 dự đoán đầu với giá thật
print("Linear Regression:", round(mean_absolute_error(y_test, du_doan), 1))
print("Điểm R²:", round(mo_hinh.score(X_test, y_test), 3))      # 4. chấm nhanh
```

**Kết quả mẫu:**

```
[ 78.9 133.  208.2] [ 52.6 178.4 212.5]
Linear Regression: 50.4
Điểm R²: 0.513
```

**Đọc kết quả:**
- Sai số giảm từ 77,6 xuống **50,4 nghìn USD**: tốt hơn mốc 35%.
- Dự đoán thứ 3 rất sát (208 so với 212), dự đoán thứ 2 lệch nhiều (133 so với 178). Vẫn còn nhiều chỗ để cải thiện. Cả Giai đoạn 2 sẽ dạy bạn cách làm.
- R² = 0,513 nghĩa là mô hình "giải thích" được khoảng 51% sự khác biệt về giá giữa các khu (sẽ học kỹ ở bài 2.3).

---

## 6. Xem trước: học không giám sát

Cùng dữ liệu đó, nhưng **không dùng giá nhà**. Chỉ đưa kinh độ, vĩ độ cho thuật toán K-Means và bảo nó: "chia thành 3 nhóm".

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, random_state=42)   # sẽ học kỹ ở bài 2.13
nhom = km.fit_predict(df[["kinh_do", "vi_do"]])         # KHÔNG có y: không giám sát
print(pd.Series(nhom).value_counts().sort_index().tolist())   # số khu mỗi nhóm
print(df.groupby(nhom)[["vi_do", "kinh_do", "gia_nha"]].mean().round(1))
```

**Kết quả mẫu:**

```
[8628, 6081, 1477]
   vi_do  kinh_do  gia_nha
0   33.8   -117.9    204.2
1   38.1   -121.9    194.4
2   36.0   -119.7    117.3
```

Không ai dạy, máy **tự** chia California thành: miền Nam (quanh LA, nhóm 0), miền Bắc (quanh SF, nhóm 1), và vùng trung tâm (nhóm 2), nơi giá nhà thấp hơn hẳn. Để ý: ta dùng `fit_predict(X)`, **không có y**.

---

## 7. Bài tập

1. Phân loại mỗi bài toán: **hồi quy**, **phân loại**, **phân cụm**, hay **giảm chiều**?
   1. Dự đoán số phút giao hàng của đơn GrabFood.
   2. Phát hiện bình luận tiêu cực trên fanpage.
   3. Gom các bài báo cùng chủ đề mà không có nhãn chủ đề.
   4. Dự đoán khách có rời bỏ (churn) gói cước trong tháng tới.
   5. Nén ảnh 1.000 điểm ảnh thành 50 số để vẽ lên biểu đồ.
2. Trong mục 5, thêm đặc trưng `ti_le_phong_ngu` và `nguoi_moi_ho` vào X. MAE thay đổi thế nào?
3. Thử `DummyRegressor(strategy="median")`. MAE của mốc này so với mốc "mean" thế nào? Vì sao? (Gợi ý: bài 1.7.)
4. Đổi `n_clusters=5` ở mục 6. In bảng trung bình. Các nhóm mới có ý nghĩa địa lý không?

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" → `Turn into → Toggle heading 3`.

1. (1) Hồi quy. (2) Phân loại. (3) Phân cụm. (4) Phân loại (rời bỏ / không). (5) Giảm chiều.
2. MAE giảm còn khoảng **48,7** (sẽ thấy lại ở bài 2.3).
3. MAE của mốc "median" **thấp hơn** một chút (75,9 so với 77,6). Vì MAE là sai số tuyệt đối, và trung vị là giá trị làm **tổng sai số tuyệt đối** nhỏ nhất (trung bình thì làm tổng **bình phương** sai số nhỏ nhất). Giá nhà lệch phải nên hai giá trị này khác nhau.
4. Các nhóm thường tách thêm quanh San Diego, Sacramento, vùng núi phía bắc... Máy chỉ dựa vào mật độ điểm trên bản đồ, nhưng kết quả trùng khớp với các vùng đô thị thật.

---

## 8. Flashcard

| Mặt trước | Mặt sau |
|---|---|
| Có giám sát vs không giám sát? | Có giám sát: có nhãn đáp án. Không giám sát: không có nhãn, tìm cấu trúc. |
| Hồi quy vs phân loại? | Hồi quy đoán con số. Phân loại đoán nhóm. |
| 7 bước dự án ML? | Bài toán → Dữ liệu → EDA → Chia → Huấn luyện/tinh chỉnh → Đánh giá test → Triển khai. |
| Estimator dùng phương thức nào? | `fit(X, y)`, `predict(X)` |
| Transformer dùng phương thức nào? | `fit(X)`, `transform(X)` |
| Baseline là gì? | Cách dự đoán ngây thơ nhất. Mô hình phải tốt hơn nó. |
| Baseline cho hồi quy? | Luôn đoán trung bình (hoặc trung vị) của train. |
| Phân cụm dùng `fit` có y không? | Không. Chỉ `fit(X)` / `fit_predict(X)`. |

**Bài tiếp theo:** 2.2 – Linear Regression & Loss Function.
