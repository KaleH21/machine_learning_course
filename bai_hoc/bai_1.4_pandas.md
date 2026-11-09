# Bài 1.4 – Pandas: đọc, lọc, nhóm, nối dữ liệu

> **Giai đoạn 1 – Python cho ML & Data Science** · Thời lượng: 3 buổi · Hình thức mở đầu: *So sánh ví von*

## 🎯 Sau bài này bạn sẽ

- Đọc một file dữ liệu thật 17.000 dòng chỉ bằng 1 dòng code.
- "Khám sức khỏe" nhanh cho dataset: kích thước, kiểu dữ liệu, ô trống, thống kê.
- Chọn cột, chọn dòng, lọc theo điều kiện, tạo cột mới, sắp xếp.
- Nhóm dữ liệu để trả lời câu hỏi kiểu "giá trung bình theo từng nhóm".
- Nối hai bảng lại với nhau.

**Chia buổi:**
- **Buổi 1:** Mục 1 → 4 (Series, DataFrame, đọc và khám dữ liệu).
- **Buổi 2:** Mục 5 → 7 (chọn, lọc, tạo cột, sắp xếp).
- **Buổi 3:** Mục 8 → 12 (nhóm, nối bảng, lưu file, bài tập).

---

## 1. So sánh ví von: Pandas là "Excel có siêu năng lực"

Nếu bạn từng dùng Excel, bạn đã biết 80% tư duy của Pandas.

| Việc trong Excel | Trong Pandas | Ưu điểm của Pandas |
|---|---|---|
| Mở file .xlsx/.csv | `pd.read_csv("file.csv")` | Mở file hàng triệu dòng không bị treo |
| Bấm Filter, chọn điều kiện | `df[df["gia"] > 3]` | Lưu lại thành code, chạy lại được |
| Gõ công thức `=B2/C2` rồi kéo xuống | `df["moi"] = df["B"] / df["C"]` | Không cần kéo, áp dụng cả cột |
| Pivot Table | `df.groupby("quan")["gia"].mean()` | Linh hoạt hơn nhiều |
| VLOOKUP | `pd.merge(bang1, bang2, on="ma")` | Không lỗi khi chèn thêm cột |
| Sort | `df.sort_values("gia")` | Giống hệt |

Khác biệt lớn nhất: trong Excel, bạn **bấm chuột** và quên mất mình đã làm gì. Trong Pandas, mọi bước là **code**. Tháng sau có dữ liệu mới, chạy lại là xong.

---

## 2. Hai kiểu dữ liệu chính

### 2.1. Series (Chuỗi dữ liệu)

- **Định nghĩa:** Một **cột** dữ liệu có nhãn cho từng dòng (gọi là **index**).
- **Ví dụ đời thực:** Cột "giá" trong bảng Excel, kèm số thứ tự dòng bên trái.

```python
import pandas as pd       # quy ước đặt tên tắt là pd
import numpy as np

# Series = mảng NumPy + nhãn cho từng phần tử
gia = pd.Series([2.5, 3.1, 4.0], index=["Căn A", "Căn B", "Căn C"])
print(gia)
print(gia["Căn B"])       # lấy giá trị theo nhãn
```

**Kết quả mẫu:**

```
Căn A    2.5
Căn B    3.1
Căn C    4.0
dtype: float64
3.1
```

### 2.2. DataFrame (Bảng dữ liệu)

- **Định nghĩa:** Một **bảng** gồm nhiều cột, mỗi cột là một Series. Đây là thứ bạn dùng 95% thời gian.
- **Ví dụ đời thực:** Một sheet Excel.

```python
# Tạo DataFrame từ dict: mỗi khóa là tên cột, mỗi giá trị là list dữ liệu của cột đó
nha = pd.DataFrame({
    "quan": ["Quận 1", "Bình Thạnh", "Thủ Đức", "Quận 7"],
    "dien_tich": [50, 65, 80, 100],
    "gia_ty": [6.5, 4.2, 3.8, 7.0],
})
print(nha)
```

**Kết quả mẫu:**

```
         quan  dien_tich  gia_ty
0      Quận 1         50     6.5
1  Bình Thạnh         65     4.2
2     Thủ Đức         80     3.8
3      Quận 7        100     7.0
```

```
           ┌──────── tên cột (columns) ────────┐
             quan      dien_tich    gia_ty
  index ┌ 0  Quận 1        50         6.5     ← một dòng = một mẫu
        │ 1  Bình Thạnh    65         4.2
        │ 2  Thủ Đức       80         3.8
        └ 3  Quận 7       100         7.0
                            ↑
                   một cột = một Series = một đặc trưng (feature)
```

---

## 3. Đọc dữ liệu thật

### 3.1. Giới thiệu dataset: Giá nhà California

Ta dùng dữ liệu điều tra dân số California (Mỹ) năm 1990. Đây là dataset "nhập môn" kinh điển của ML.

> ⚠️ **Đọc kỹ trước khi dùng:** mỗi dòng **không phải** một căn nhà, mà là một **khu dân cư** (vài trăm đến vài nghìn người). Giá nhà là **giá trung vị** của khu đó, đơn vị **USD**. Thu nhập là thu nhập trung vị, đơn vị **chục nghìn USD/năm** (ví dụ 3.5 = 35.000 USD).

Hiểu sai ý nghĩa cột là lỗi phổ biến và nguy hiểm nhất khi phân tích dữ liệu.

### 3.2. Đọc file

```python
# Đọc thẳng từ internet. Trên Colab cũng có sẵn file này ở:
# "/content/sample_data/california_housing_train.csv"
url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
df = pd.read_csv(url)        # df là tên quy ước cho DataFrame chính
print(df.shape)              # (số dòng, số cột)
```

**Kết quả mẫu:**

```
(17000, 9)
```

### 3.3. Đổi tên cột sang tiếng Việt cho dễ nhớ

```python
df = df.rename(columns={                 # dict {tên cũ: tên mới}
    "longitude": "kinh_do", "latitude": "vi_do",
    "housing_median_age": "tuoi_nha", "total_rooms": "tong_phong",
    "total_bedrooms": "tong_phong_ngu", "population": "dan_so",
    "households": "so_ho", "median_income": "thu_nhap",
    "median_house_value": "gia_nha",
})
pd.set_option("display.max_columns", None)   # hiện đủ mọi cột khi in
print(df.head())                              # xem 5 dòng đầu
```

**Kết quả mẫu:**

```
   kinh_do  vi_do  tuoi_nha  tong_phong  tong_phong_ngu  dan_so  so_ho  thu_nhap  gia_nha
0  -114.31  34.19      15.0      5612.0          1283.0  1015.0  472.0    1.4936  66900.0
1  -114.47  34.40      19.0      7650.0          1901.0  1129.0  463.0    1.8200  80100.0
2  -114.56  33.69      17.0       720.0           174.0   333.0  117.0    1.6509  85700.0
3  -114.57  33.64      14.0      1501.0           337.0   515.0  226.0    3.1917  73400.0
4  -114.57  33.57      20.0      1454.0           326.0   624.0  262.0    1.9250  65500.0
```

> 💡 Trên Colab, nếu dòng cuối của ô là `df.head()` (không có `print`), Colab sẽ hiển thị một bảng đẹp hơn. Cả hai cách đều đúng.

---

## 4. "Khám sức khỏe" dataset: 3 lệnh luôn chạy đầu tiên

### 4.1. `info()`: kiểu dữ liệu và ô trống

```python
df.info()
```

**Kết quả mẫu:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17000 entries, 0 to 16999
Data columns (total 9 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   kinh_do         17000 non-null  float64
 1   vi_do           17000 non-null  float64
 2   tuoi_nha        17000 non-null  float64
 3   tong_phong      17000 non-null  float64
 4   tong_phong_ngu  17000 non-null  float64
 5   dan_so          17000 non-null  float64
 6   so_ho           17000 non-null  float64
 7   thu_nhap        17000 non-null  float64
 8   gia_nha         17000 non-null  float64
dtypes: float64(9)
memory usage: 1.2 MB
```

Đọc: 17.000 dòng, mọi cột đều có đủ 17.000 giá trị (**non-null** = không trống). Tất cả là số thực. Dataset này khá sạch. Bài 1.6 sẽ cho bạn gặp dữ liệu "bẩn".

### 4.2. `describe()`: thống kê nhanh mọi cột số

```python
print(df.describe().round(2))     # round(2): làm tròn 2 số lẻ cho dễ đọc
```

**Kết quả mẫu:**

```
        kinh_do     vi_do  tuoi_nha  tong_phong  tong_phong_ngu    dan_so     so_ho  thu_nhap    gia_nha
count  17000.00  17000.00  17000.00    17000.00        17000.00  17000.00  17000.00  17000.00   17000.00
mean    -119.56     35.63     28.59     2643.66          539.41   1429.57    501.22      3.88  207300.91
std        2.01      2.14     12.59     2179.95          421.50   1147.85    384.52      1.91  115983.76
min     -124.35     32.54      1.00        2.00            1.00      3.00      1.00      0.50   14999.00
25%     -121.79     33.93     18.00     1462.00          297.00    790.00    282.00      2.57  119400.00
50%     -118.49     34.25     29.00     2127.00          434.00   1167.00    409.00      3.54  180400.00
75%     -118.00     37.72     37.00     3151.25          648.25   1721.00    605.25      4.77  265000.00
max     -114.31     41.95     52.00    37937.00         6445.00  35682.00   6082.00     15.00  500001.00
```

| Dòng | Ý nghĩa | (sẽ học kỹ ở) |
|---|---|---|
| count | Số ô có dữ liệu | |
| mean | Trung bình | bài 1.7 |
| std | Độ lệch chuẩn: dữ liệu phân tán rộng hay hẹp | bài 1.7 |
| min / max | Nhỏ nhất / lớn nhất | |
| 25% / 50% / 75% | Phân vị. 50% chính là **trung vị** | bài 1.7 |

**Thám tử dữ liệu 🔍:** Nhìn cột `gia_nha`, bạn có thấy gì lạ không?
- `max` = **500001**, một con số "tròn" đáng ngờ.
- Đó là vì người thu thập đã **giới hạn trần**: mọi khu có giá trên 500.000 USD đều bị ghi là 500.001. Ta sẽ đếm xem có bao nhiêu khu như vậy ở mục 8.
- `tong_phong` max = 37.937 trong khi 75% số khu dưới 3.152 → có vài khu **cực lớn** (outlier).

Đây là kỹ năng quan trọng nhất của người làm ML: **nghi ngờ dữ liệu trước khi tin nó**.

---

## 5. Chọn cột, chọn dòng

### 5.1. Chọn cột

```python
print(df["gia_nha"].head(3))                 # 1 cột → ra Series
print(df[["thu_nhap", "gia_nha"]].head(3))   # nhiều cột → truyền LIST tên cột → ra DataFrame
```

**Kết quả mẫu:**

```
0    66900.0
1    80100.0
2    85700.0
Name: gia_nha, dtype: float64
   thu_nhap  gia_nha
0    1.4936  66900.0
1    1.8200  80100.0
2    1.6509  85700.0
```

Để ý **hai lớp ngoặc** `[[...]]` khi chọn nhiều cột: ngoặc ngoài là "lấy từ df", ngoặc trong là "danh sách tên cột".

### 5.2. `iloc` và `loc`

| | `iloc` | `loc` |
|---|---|---|
| Chọn theo | **Vị trí** (số thứ tự 0, 1, 2...) | **Nhãn** (tên index, tên cột) |
| Mẹo nhớ | **i** = **i**nteger (số nguyên) | **l** = **l**abel (nhãn) |
| Cắt lát `0:2` | Lấy 0, 1 (bỏ 2) | Lấy 0, 1, **2** (tính cả 2) |

```python
print(df.iloc[0])                              # dòng ở vị trí 0
print(df.iloc[0:3, 0:2])                       # dòng 0–2, cột 0–1 (theo vị trí)
print(df.loc[0:2, ["tuoi_nha", "gia_nha"]])    # dòng có nhãn 0 đến 2, cột theo tên
```

**Kết quả mẫu:**

```
kinh_do            -114.3100
vi_do                34.1900
tuoi_nha             15.0000
tong_phong         5612.0000
tong_phong_ngu     1283.0000
dan_so             1015.0000
so_ho               472.0000
thu_nhap              1.4936
gia_nha           66900.0000
Name: 0, dtype: float64
   kinh_do  vi_do
0  -114.31  34.19
1  -114.47  34.40
2  -114.56  33.69
   tuoi_nha  gia_nha
0      15.0  66900.0
1      19.0  80100.0
2      17.0  85700.0
```

---

## 6. Lọc dữ liệu

Giống hệt boolean mask của NumPy (bài 1.3).

```python
# Các khu có giá nhà trên 400.000 USD
dat = df[df["gia_nha"] > 400000]
print(len(dat))

# Khu thu nhập cao (> 80.000 USD) VÀ nhà mới (< 10 năm). Nhớ ngoặc và dấu &
loc2 = df[(df["thu_nhap"] > 8) & (df["tuoi_nha"] < 10)]
print(len(loc2))

# Cách viết khác dễ đọc hơn: query() dùng and/or như câu nói thường
print(df.query("thu_nhap > 8 and tuoi_nha < 10").shape)
```

**Kết quả mẫu:**

```
1458
61
(61, 9)
```

---

## 7. Tạo cột mới và sắp xếp

### 7.1. Tạo cột mới

```python
# Số phòng trung bình mỗi hộ: chia cả cột cho cả cột (vector hóa)
df["phong_moi_ho"] = df["tong_phong"] / df["so_ho"]

# Đổi giá sang triệu VND (giả sử 1 USD = 25.000 VND)
df["gia_trieu_vnd"] = df["gia_nha"] * 25000 / 1e6     # 1e6 = 1.000.000

# Cột dạng chữ: nhà từ 30 năm tuổi là "Cũ", còn lại là "Mới"
df["loai_nha"] = np.where(df["tuoi_nha"] >= 30, "Cũ", "Mới")

print(df[["phong_moi_ho", "gia_trieu_vnd", "loai_nha"]].head(3).round(2))
```

**Kết quả mẫu:**

```
   phong_moi_ho  gia_trieu_vnd loai_nha
0         11.89         1672.5      Mới
1         16.52         2002.5      Mới
2          6.15         2142.5      Mới
```

**Liên hệ ML:** `phong_moi_ho` là một **đặc trưng mới tạo ra** từ 2 cột có sẵn. Tổng số phòng của cả khu không nói lên nhiều, nhưng "số phòng mỗi hộ" phản ánh nhà rộng hay chật. Việc tạo đặc trưng thông minh như vậy gọi là **Feature Engineering (Kỹ thuật đặc trưng)** (sẽ học kỹ ở bài 2.16).

### 7.2. Sắp xếp

```python
# 3 khu giá cao nhất
print(df.sort_values("gia_nha", ascending=False)[["thu_nhap", "gia_nha"]].head(3))

# Cách viết gọn: nlargest lấy n dòng có giá trị lớn nhất
print(df.nlargest(3, "thu_nhap")[["thu_nhap", "gia_nha"]])
```

**Kết quả mẫu:**

```
      thu_nhap   gia_nha
2698   10.1122  500001.0
7156    6.8870  500001.0
7157   11.6677  500001.0
      thu_nhap   gia_nha
1625   15.0001  500001.0
3338   15.0001  500001.0
4461   15.0001  500001.0
```

Lại gặp con số 500001. Và thu nhập cao nhất là **15.0001**, cũng tròn đáng ngờ: cột thu nhập cũng bị giới hạn trần!

---

## 8. Đếm và nhóm dữ liệu

### 8.1. `value_counts()`: đếm số lần xuất hiện

```python
print(df["loai_nha"].value_counts())                          # đếm số lượng
print(df["loai_nha"].value_counts(normalize=True).round(3))   # tỉ lệ thay vì số lượng
print((df["gia_nha"] >= 500000).sum())                        # số khu bị giới hạn trần giá
```

**Kết quả mẫu:**

```
loai_nha
Mới    8811
Cũ     8189
Name: count, dtype: int64
loai_nha
Mới    0.518
Cũ     0.482
Name: proportion, dtype: float64
836
```

836 khu (gần 5%) có giá ≥ 500.000. Phần lớn là bị ghi đúng 500.001 do giới hạn trần. Khi dạy mô hình dự đoán giá (bài 2.2), mô hình sẽ "tưởng" giá không bao giờ vượt quá 500.001. Đây là vấn đề cần ghi chú lại.

### 8.2. `groupby()`: tách – tính – gộp

**Trực giác:** giống chia bài thành từng chồng theo màu, đếm điểm từng chồng, rồi ghi kết quả.

```
      Dữ liệu gốc            TÁCH theo loai_nha        TÍNH mean        GỘP
 loai_nha   gia_nha        ┌─ Cũ:  215k, 180k ... ─►  215.706  ─┐
  Mới       66.900         │                                     ├─► bảng kết quả
  Cũ       215.000   ──►   │                                     │
  Mới       80.100         └─ Mới:  66.9k, 80.1k... ─► 199.490 ─┘
   ...
```

```python
# Giá nhà trung bình của từng loại nhà
print(df.groupby("loai_nha")["gia_nha"].mean().round(0))

# Tính nhiều thứ một lúc với agg(): tên_cột_mới=(cột_gốc, phép_tính)
print(df.groupby("loai_nha").agg(
    so_khu=("gia_nha", "count"),
    gia_tb=("gia_nha", "mean"),
    thu_nhap_tb=("thu_nhap", "mean"),
).round(2))
```

**Kết quả mẫu:**

```
loai_nha
Cũ     215706.0
Mới    199490.0
Name: gia_nha, dtype: float64
          so_khu     gia_tb  thu_nhap_tb
loai_nha                                
Cũ          8189  215705.51         3.70
Mới         8811  199489.63         4.06
```

**Điều thú vị:** khu nhà cũ lại đắt hơn, dù thu nhập ở đó thấp hơn! Vì sao? Có thể nhà cũ thường nằm ở trung tâm thành phố lớn (xây từ sớm, đất đắt), còn nhà mới ở vùng ven. Đây chỉ là **giả thuyết**. Bài 1.5 sẽ vẽ bản đồ để kiểm tra.

### 8.3. `apply()`: áp dụng hàm của bạn cho cả cột

```python
def muc_thu_nhap(x):                     # hàm thường (bài 1.2)
    if x < 3:
        return "Thấp"
    elif x < 6:
        return "Trung bình"
    return "Cao"

df["muc_thu_nhap"] = df["thu_nhap"].apply(muc_thu_nhap)   # gọi hàm cho TỪNG giá trị
print(df.groupby("muc_thu_nhap")["gia_nha"].mean().round(0).sort_values())

# Bảng chéo: đếm số khu theo 2 tiêu chí cùng lúc
print(pd.crosstab(df["muc_thu_nhap"], df["loai_nha"]))
```

**Kết quả mẫu:**

```
muc_thu_nhap
Thấp          133297.0
Trung bình    218984.0
Cao           379918.0
Name: gia_nha, dtype: float64
loai_nha        Cũ   Mới
muc_thu_nhap            
Cao            734  1251
Thấp          3206  2840
Trung bình    4249  4720
```

Thu nhập càng cao, giá nhà càng cao, và chênh lệch rất rõ. `thu_nhap` hứa hẹn là đặc trưng mạnh nhất để dự đoán giá (sẽ kiểm chứng bằng hệ số tương quan ở bài 1.7).

> ⚠️ `apply` chậm hơn phép toán vector hóa vì nó gọi hàm Python cho từng dòng. Chỉ dùng khi logic phức tạp, không viết được bằng phép toán cả cột.

---

## 9. Nối bảng

### 9.1. Bài toán

Một shop online có 2 bảng: **đơn hàng** (chỉ có mã khách) và **khách hàng** (có tên, thành phố). Muốn biết "đơn nào của ai, ở đâu" thì phải nối lại. Đây chính là VLOOKUP trong Excel, hoặc JOIN trong SQL.

```python
don_hang = pd.DataFrame({
    "ma_don": [101, 102, 103, 104],
    "ma_kh": ["KH1", "KH2", "KH1", "KH9"],     # KH9 không có trong bảng khách hàng!
    "tien": [120000, 85000, 60000, 200000],
})
khach_hang = pd.DataFrame({
    "ma_kh": ["KH1", "KH2", "KH3"],
    "ten": ["An", "Bình", "Chi"],
    "thanh_pho": ["Hà Nội", "TP.HCM", "Đà Nẵng"],
})

# inner: chỉ giữ dòng khớp ở CẢ HAI bảng
print(pd.merge(don_hang, khach_hang, on="ma_kh", how="inner"))

# left: giữ MỌI dòng bảng trái (đơn hàng), không khớp thì điền NaN
print(pd.merge(don_hang, khach_hang, on="ma_kh", how="left"))
```

**Kết quả mẫu:**

```
   ma_don ma_kh    tien   ten thanh_pho
0     101   KH1  120000    An    Hà Nội
1     102   KH2   85000  Bình    TP.HCM
2     103   KH1   60000    An    Hà Nội
   ma_don ma_kh    tien   ten thanh_pho
0     101   KH1  120000    An    Hà Nội
1     102   KH2   85000  Bình    TP.HCM
2     103   KH1   60000    An    Hà Nội
3     104   KH9  200000   NaN       NaN
```

```
 how="inner"            how="left"
   ┌───┐ ┌───┐           ┌───┐ ┌───┐
   │ ▒▒│█│▒▒ │           │███│█│   │
   │ ▒▒│█│▒▒ │           │███│█│   │
   └───┘ └───┘           └───┘ └───┘
   chỉ phần giao         toàn bộ bảng trái
```

- **NaN – Not a Number:** ký hiệu "ô trống / thiếu dữ liệu" trong Pandas (sẽ học cách xử lý ở bài 1.6).
- Đơn 104 trị giá 200.000đ **biến mất** khi dùng inner. Nếu bạn tính tổng doanh thu sau khi inner-merge, bạn sẽ báo cáo thiếu tiền. Luôn kiểm tra số dòng trước và sau khi merge!

### 9.2. Nối dọc: thêm dòng

```python
khach_moi = pd.DataFrame({"ma_kh": ["KH4"], "ten": ["Dũng"], "thanh_pho": ["Huế"]})
print(pd.concat([khach_hang, khach_moi], ignore_index=True))   # chồng bảng lên nhau
```

**Kết quả mẫu:**

```
  ma_kh   ten thanh_pho
0   KH1    An    Hà Nội
1   KH2  Bình    TP.HCM
2   KH3   Chi   Đà Nẵng
3   KH4  Dũng       Huế
```

---

## 10. Lưu kết quả

```python
df.to_csv("nha_da_xu_ly.csv", index=False)   # index=False: không ghi cột số thứ tự thừa
print(pd.read_csv("nha_da_xu_ly.csv").shape) # đọc lại để kiểm tra
```

**Kết quả mẫu:**

```
(17000, 13)
```

13 cột = 9 cột gốc + 4 cột mới tạo (`phong_moi_ho`, `gia_trieu_vnd`, `loai_nha`, `muc_thu_nhap`).

> Trên Colab, nhớ lưu vào Google Drive (bài 0.2, mục 4.8) nếu muốn giữ file.

---

## 11. Bài tập

1. Giá nhà cao nhất là bao nhiêu? Có bao nhiêu khu có **đúng** giá đó?
2. Có bao nhiêu khu có tuổi nhà ≥ 50 năm?
3. Giá nhà trung bình của nhóm thu nhập "Cao" là bao nhiêu? (Dùng `loc` với điều kiện và tên cột.)
4. Tạo cột `mat_do` = số người mỗi hộ (`dan_so / so_ho`). Dùng `describe()`. Giá trị max có hợp lý không?
5. Số phòng mỗi hộ **trung vị** (`median`) của nhà Cũ và nhà Mới?
6. Dùng bảng đơn hàng và khách hàng ở mục 9: tính **tổng tiền mỗi khách** (theo tên).

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" → `Turn into → Toggle heading 3`.

```python
# 1
print(df["gia_nha"].max(), (df["gia_nha"] == df["gia_nha"].max()).sum())   # 500001.0 814

# 2
print(df[df["tuoi_nha"] >= 50].shape[0])                                   # 1196

# 3
print(round(df.loc[df["muc_thu_nhap"] == "Cao", "gia_nha"].mean()))        # 379918

# 4
df["mat_do"] = df["dan_so"] / df["so_ho"]
print(df["mat_do"].describe().round(2))
# max = 502.46 người/hộ → vô lý! Có thể là ký túc xá, trại lính, nhà tù...
# hoặc lỗi nhập liệu. Đây là outlier cần xử lý (bài 1.6).

# 5
print(df.groupby("loai_nha")["phong_moi_ho"].median().round(2))           # Cũ 5.01, Mới 5.49

# 6
tong = pd.merge(don_hang, khach_hang, on="ma_kh", how="left").groupby("ten")["tien"].sum()
print(tong)   # An 180000, Bình 85000
# Chú ý: đơn 104 (KH9) bị mất vì tên là NaN, groupby bỏ qua NaN theo mặc định.
```

---

## 12. Flashcard

| Mặt trước | Mặt sau |
|---|---|
| Series vs DataFrame? | Series là 1 cột. DataFrame là bảng nhiều cột. |
| 3 lệnh "khám sức khỏe" dataset? | `shape`, `info()`, `describe()` |
| Chọn nhiều cột? | `df[["a", "b"]]` (hai lớp ngoặc) |
| `iloc` vs `loc`? | `iloc` theo vị trí số. `loc` theo nhãn/tên. |
| Lọc nhiều điều kiện? | `df[(df.a > 1) & (df.b < 5)]` hoặc `df.query("a > 1 and b < 5")` |
| `groupby` hoạt động thế nào? | Tách theo nhóm → tính → gộp kết quả. |
| `inner` vs `left` merge? | inner: chỉ dòng khớp cả hai. left: giữ hết bảng trái. |
| NaN là gì? | Ô thiếu dữ liệu. |
| Dấu hiệu dữ liệu bị giới hạn trần? | Max là con số tròn bất thường, nhiều dòng trùng đúng giá trị đó. |

**Bài tiếp theo:** 1.5 – Matplotlib/Seaborn: vẽ biểu đồ để "nhìn" dữ liệu.
