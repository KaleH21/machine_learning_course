# Bài 1.5 – Matplotlib/Seaborn: vẽ biểu đồ để "nhìn" dữ liệu

> **Giai đoạn 1 – Python cho ML & Data Science** · Thời lượng: 2 buổi · Hình thức mở đầu: *Đố vui trước khi học*

## 🎯 Sau bài này bạn sẽ

- Hiểu vì sao **không bao giờ** chỉ nhìn con số thống kê mà bỏ qua biểu đồ.
- Chọn đúng loại biểu đồ cho từng câu hỏi.
- Vẽ histogram, scatter, bar, line, boxplot, heatmap và "bản đồ" bằng Matplotlib/Seaborn.
- Đọc biểu đồ để phát hiện vấn đề dữ liệu (giới hạn trần, outlier, phân bố lệch).

**Chia buổi:**
- **Buổi 1:** Mục 1 → 5 (Matplotlib, 4 biểu đồ cơ bản).
- **Buổi 2:** Mục 6 → 10 (Seaborn, bản đồ, kiểm chứng giả thuyết, bài tập).

---

## 1. Đố vui trước khi học 🤔

Có 4 bộ dữ liệu A, B, C, D, mỗi bộ có 11 điểm (x, y). Bạn tính thống kê và được:

```
A  TB x = 9.0 | TB y = 7.5 | ĐLC y = 2.03 | tương quan = 0.816
B  TB x = 9.0 | TB y = 7.5 | ĐLC y = 2.03 | tương quan = 0.816
C  TB x = 9.0 | TB y = 7.5 | ĐLC y = 2.03 | tương quan = 0.816
D  TB x = 9.0 | TB y = 7.5 | ĐLC y = 2.03 | tương quan = 0.817
```

(TB = trung bình, ĐLC = độ lệch chuẩn. Tương quan đo mức hai biến "đi cùng nhau", sẽ học kỹ ở bài 1.7.)

**Câu hỏi:** 4 bộ dữ liệu này có giống nhau không?

Hầu hết mọi người đoán "gần như giống hệt". Giờ hãy vẽ:

```python
import matplotlib.pyplot as plt   # thư viện vẽ cơ bản, quy ước tên tắt là plt

# Bộ tứ Anscombe (Anscombe's quartet), một ví dụ kinh điển năm 1973
x  = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
bo = {"A": (x, y1), "B": (x, y2), "C": (x, y3), "D": (x4, y4)}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))     # 1 hàng, 4 ô biểu đồ
for ax, (ten, (xs, ys)) in zip(axes, bo.items()):   # zip: ghép từng ô với từng bộ dữ liệu
    ax.scatter(xs, ys)                               # vẽ điểm
    ax.set_title(f"Bộ {ten}")
plt.tight_layout()                                   # tự căn chỉnh để chữ không đè nhau
plt.show()
```

**Bạn sẽ thấy 4 hình hoàn toàn khác nhau:**

```
  Bộ A              Bộ B              Bộ C              Bộ D
  y│      •  •      y│    • • •       y│          •     y│             •
   │   • •  •        │  •       •      │               │
   │  •  •           │ •                │        • •     │ •
   │ • •             │•                 │   • • •        │ •
   │•                │                  │• •             │ •
   └────────x        └────────x         └────────x       └──────────x
  Đường thẳng      Đường cong        Đường thẳng       Mọi điểm cùng x,
  có nhiễu         (parabol)         + 1 điểm lạc      trừ 1 điểm lạc
```

**Bài học:** Con số thống kê **giống hệt nhau** có thể che giấu những bức tranh **hoàn toàn khác nhau**. Một mô hình đường thẳng (bài 2.2) chỉ hợp với bộ A. Vì vậy, quy tắc số 1 của người làm ML là: **luôn vẽ dữ liệu trước khi làm gì khác**.

---

## 2. Khái niệm

### 2.1. Matplotlib

- **Định nghĩa:** Thư viện vẽ biểu đồ "gốc" của Python. Vẽ được mọi thứ, nhưng phải viết nhiều code.
- **Ví dụ đời thực:** Như bộ bút chì màu đầy đủ: vẽ gì cũng được, nhưng phải tự tô từng nét.

### 2.2. Seaborn

- **Định nghĩa:** Thư viện xây trên Matplotlib, vẽ biểu đồ thống kê đẹp với ít code, làm việc trực tiếp với DataFrame.
- **Ví dụ đời thực:** Như bộ khuôn vẽ có sẵn: đặt vào, tô một lần là ra hình đẹp.

### 2.3. Cấu tạo một biểu đồ Matplotlib

```
┌─────────────── Figure (cả tờ giấy) ───────────────┐
│  ┌────────── Axes (một ô biểu đồ) ─────────────┐  │
│  │              Title (tiêu đề)                │  │
│  │  y ▲                                        │  │
│  │    │      •    •                            │  │
│  │ Y  │   •    •                               │  │
│  │label│ •                                     │  │
│  │    └──────────────────────► x               │  │
│  │              X label                        │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

Mẫu code bạn sẽ dùng mãi:

```python
fig, ax = plt.subplots(figsize=(8, 4))   # tạo tờ giấy (fig) và 1 ô biểu đồ (ax), rộng 8 cao 4 inch
ax.plot(...)  /  ax.scatter(...)  /  ax.hist(...)   # vẽ lên ô
ax.set_xlabel("...")                     # nhãn trục x
ax.set_ylabel("...")                     # nhãn trục y
ax.set_title("...")                      # tiêu đề
plt.show()                               # hiển thị
```

---

## 3. Chọn biểu đồ nào?

| Câu hỏi bạn muốn trả lời | Biểu đồ | Hàm |
|---|---|---|
| Một biến số phân bố thế nào? (đa số nằm đâu, có lệch không) | **Histogram** | `ax.hist`, `sns.histplot` |
| Hai biến số có liên quan nhau không? | **Scatter** (biểu đồ điểm) | `ax.scatter`, `sns.scatterplot` |
| So sánh một con số giữa các nhóm | **Bar** (cột) | `ax.bar`, `sns.barplot` |
| Thay đổi theo thời gian | **Line** (đường) | `ax.plot`, `sns.lineplot` |
| So sánh phân bố giữa các nhóm, tìm outlier | **Boxplot** (hộp) | `sns.boxplot` |
| Nhiều cặp biến liên quan nhau thế nào? | **Heatmap** (bản đồ nhiệt) | `sns.heatmap` |

---

## 4. Chuẩn bị dữ liệu

Dùng lại dataset giá nhà California (bài 1.4):

```python
import pandas as pd
import numpy as np
import seaborn as sns     # quy ước tên tắt là sns

url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
df = pd.read_csv(url)
df.columns = ["kinh_do", "vi_do", "tuoi_nha", "tong_phong", "tong_phong_ngu",
              "dan_so", "so_ho", "thu_nhap", "gia_nha"]       # đổi tên cột (cách nhanh)
df["loai_nha"] = np.where(df["tuoi_nha"] >= 30, "Cũ", "Mới")  # như bài 1.4
```

---

## 5. Bốn biểu đồ cơ bản với Matplotlib

### 5.1. Histogram (Biểu đồ tần suất)

- **Định nghĩa:** Chia khoảng giá trị thành nhiều "thùng" (bin), rồi đếm có bao nhiêu điểm rơi vào mỗi thùng.
- **Ví dụ đời thực:** Phổ điểm thi đại học: bao nhiêu thí sinh được 5 điểm, 6 điểm, 7 điểm...

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["gia_nha"], bins=50, color="steelblue", edgecolor="white")  # chia 50 thùng
ax.set_xlabel("Giá nhà (USD)")
ax.set_ylabel("Số khu dân cư")
ax.set_title("Phân bố giá nhà")
plt.show()
```

**Bạn sẽ thấy:**

```
Số khu
  │                                                       █  ← cột đột biến
  │        ▄█▄ █                                          █     ở 500.000!
  │     ▄██████████▄                                      █
  │   ▄██████████████▄▄                                   █
  │  ████████████████████▄▄▄▄▄                            █
  │ ███████████████████████████████▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  └─────────────────────────────────────────────────────────► Giá
   0      100k     200k     300k     400k           500k
```

**Đọc biểu đồ:**
1. Đa số khu có giá 100.000–250.000 USD.
2. Phân bố **lệch phải**: đuôi kéo dài về phía giá cao (sẽ học kỹ ở bài 1.7).
3. Cột cao vọt ở 500.000: chính là **giới hạn trần** bạn đã phát hiện ở bài 1.4. Biểu đồ cho thấy nó ngay lập tức, không cần đếm.

### 5.2. Scatter (Biểu đồ điểm)

```python
fig, ax = plt.subplots(figsize=(6, 5))
# s: kích thước điểm; alpha: độ trong suốt (0 = vô hình, 1 = đặc), giúp thấy chỗ điểm dày đặc
ax.scatter(df["thu_nhap"], df["gia_nha"], s=3, alpha=0.2)
ax.set_xlabel("Thu nhập (chục nghìn USD)")
ax.set_ylabel("Giá nhà (USD)")
ax.set_title("Thu nhập càng cao, giá nhà càng cao")
plt.show()
```

**Bạn sẽ thấy:**

```
Giá │ ••••••••••••••••••••••••••••••••••••••  ← "trần" 500.000
    │           ⁘⁘⁘⁘⁘•  •
    │        ⁘⁘⁘⁘⁘⁘⁘•    •
    │     ⁘⁘⁘⁘⁘⁘⁘⁘•          •
    │  ⁘⁘⁘⁘⁘⁘⁘⁘                    •
    │⁘⁘⁘⁘⁘⁘
    └─────────────────────────────────► Thu nhập
```

**Đọc biểu đồ:**
1. Đám mây điểm đi lên từ trái sang phải → thu nhập và giá nhà **cùng tăng**. Đây là đặc trưng quan trọng.
2. Một đường ngang dày ở 500.000 (giới hạn trần). Nhìn kỹ còn có vài đường ngang mờ hơn. Đếm bằng `df["gia_nha"].value_counts().head()` sẽ thấy các giá "tròn" như 137.500, 162.500, 112.500 xuất hiện 70–95 lần mỗi giá. Có thể người khai đã làm tròn giá nhà. Đây là loại chi tiết chỉ biểu đồ mới cho thấy.
3. Một cụm điểm ở thu nhập = 15 (giới hạn trần của thu nhập).

### 5.3. Bar (Biểu đồ cột)

```python
# Giá trung bình theo nhóm tuổi nhà. pd.cut chia số thành các khoảng
nhom_tuoi = pd.cut(df["tuoi_nha"], bins=[0, 10, 20, 30, 40, 52])
gia_tb = df.groupby(nhom_tuoi, observed=True)["gia_nha"].mean()
print(gia_tb.round(0))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(gia_tb.index.astype(str), gia_tb.values, color="coral")   # trục x phải là chữ
ax.set_xlabel("Tuổi nhà (năm)")
ax.set_ylabel("Giá trung bình (USD)")
plt.show()
```

**Kết quả mẫu:**

```
tuoi_nha
(0, 10]     200928.0
(10, 20]    191312.0
(20, 30]    207208.0
(30, 40]    206814.0
(40, 52]    230420.0
Name: gia_nha, dtype: float64
```

`(0, 10]` nghĩa là "lớn hơn 0 và nhỏ hơn hoặc bằng 10". Nhà trên 40 năm tuổi có giá trung bình cao nhất.

### 5.4. Line (Biểu đồ đường)

Dataset này không có thời gian, nên ta dùng ví dụ doanh thu một quán cà phê:

```python
thang = ["T1", "T2", "T3", "T4", "T5", "T6"]
doanh_thu = [120, 95, 140, 160, 155, 190]      # triệu đồng

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(thang, doanh_thu, marker="o")           # marker="o": đánh dấu từng điểm
ax.set_ylabel("Doanh thu (triệu đồng)")
ax.set_title("Doanh thu 6 tháng đầu năm")
ax.grid(alpha=0.3)                              # lưới mờ cho dễ đọc
plt.show()
```

Bạn sẽ dùng biểu đồ đường rất nhiều ở Giai đoạn 3 để vẽ **loss qua từng vòng huấn luyện** (sẽ học kỹ ở bài 3.5).

---

## 6. Seaborn: đẹp hơn, ít code hơn

### 6.1. Histogram chia nhóm bằng màu (`hue`)

```python
plt.figure(figsize=(8, 4))
# hue: tô màu theo nhóm; kde=True: thêm đường cong làm mượt phân bố
sns.histplot(data=df, x="thu_nhap", hue="loai_nha", bins=40, kde=True)
plt.title("Phân bố thu nhập: nhà Cũ vs nhà Mới")
plt.show()
```

Chỉ 1 dòng vẽ, Seaborn tự lo màu sắc, chú thích (legend), tên trục.

### 6.2. Boxplot (Biểu đồ hộp)

- **Định nghĩa:** Tóm tắt phân bố bằng 5 con số: nhỏ nhất, phân vị 25%, trung vị, phân vị 75%, lớn nhất. Điểm nằm quá xa được vẽ riêng là outlier.
- **Ví dụ đời thực:** So sánh lương của nhân viên giữa các phòng ban chỉ bằng một hình.

```
        outlier →   •
                    •
                  ──┬──   ← "râu" trên (giá trị lớn nhất không tính outlier)
                    │
                ┌───┴───┐ ← phân vị 75%
                │       │
                ├───────┤ ← TRUNG VỊ (50%)
                │       │
                └───┬───┘ ← phân vị 25%
                    │
                  ──┴──   ← "râu" dưới
```

```python
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="loai_nha", y="gia_nha")   # mỗi nhóm một hộp
plt.title("Giá nhà theo loại nhà")
plt.show()
print(df.groupby("loai_nha")["gia_nha"].median())  # kiểm tra trung vị bằng số
```

**Kết quả mẫu:**

```
loai_nha
Cũ     185000.0
Mới    175000.0
Name: gia_nha, dtype: float64
```

Hai hộp khá giống nhau, đường trung vị của nhà Cũ cao hơn một chút. Cả hai đều có nhiều điểm outlier phía trên (các khu giá cao).

### 6.3. Heatmap (Bản đồ nhiệt) của tương quan

```python
cot = ["tuoi_nha", "tong_phong", "dan_so", "thu_nhap", "gia_nha"]
plt.figure(figsize=(7, 6))
sns.heatmap(df[cot].corr(),          # .corr(): bảng hệ số tương quan (sẽ học kỹ ở bài 1.7)
            annot=True, fmt=".2f",   # ghi số lên từng ô, 2 số lẻ
            cmap="coolwarm",         # xanh = âm, đỏ = dương
            vmin=-1, vmax=1)         # thang màu từ -1 đến 1
plt.show()
print(df[cot].corr()["gia_nha"].round(2))
```

**Kết quả mẫu:**

```
tuoi_nha      0.11
tong_phong    0.13
dan_so       -0.03
thu_nhap      0.69
gia_nha       1.00
Name: gia_nha, dtype: float64
```

Ô `thu_nhap`–`gia_nha` đỏ đậm nhất (0,69). Khớp với scatter ở mục 5.2. Các cột còn lại gần như không liên quan **theo đường thẳng** với giá nhà.

---

## 7. Vẽ "bản đồ" từ kinh độ, vĩ độ

Ở bài 1.4 có câu hỏi: *"Vì sao khu nhà Cũ lại đắt hơn?"* Giả thuyết: nhà cũ ở trung tâm thành phố. Hãy để dữ liệu trả lời.

```python
fig, ax = plt.subplots(figsize=(7, 6))
diem = ax.scatter(df["kinh_do"], df["vi_do"],
                  c=df["gia_nha"],          # màu theo giá nhà
                  cmap="viridis",           # bảng màu: tím (rẻ) → vàng (đắt)
                  s=df["dan_so"] / 200,     # kích thước theo dân số
                  alpha=0.4)
fig.colorbar(diem, label="Giá nhà (USD)")   # thanh chú thích màu
ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ")
ax.set_title("Giá nhà trên bản đồ California")
plt.show()
```

**Bạn sẽ thấy:** hình dáng bang California hiện ra chỉ từ các chấm. Những vùng **vàng (đắt)** tập trung **dọc bờ biển**, quanh hai khu đô thị lớn: San Francisco (vĩ độ ~37,8) và Los Angeles (vĩ độ ~34). Vùng nội địa phía đông chủ yếu màu tím (rẻ).

Kiểm chứng giả thuyết bằng con số:

```python
# Khoanh vùng gần đúng trung tâm 2 thành phố lớn bằng kinh độ, vĩ độ
la = df["vi_do"].between(33.9, 34.2) & df["kinh_do"].between(-118.5, -118.1)
sf = df["vi_do"].between(37.6, 37.9) & df["kinh_do"].between(-122.55, -122.2)

print("Tuổi nhà TB - LA:", df[la]["tuoi_nha"].mean().round(1),
      "| SF:", df[sf]["tuoi_nha"].mean().round(1),
      "| Toàn bang:", df["tuoi_nha"].mean().round(1))
```

**Kết quả mẫu:**

```
Tuổi nhà TB - LA: 36.8 | SF: 45.0 | Toàn bang: 28.6
```

Đúng như giả thuyết: nhà ở trung tâm LA và SF **cũ hơn** trung bình toàn bang rất nhiều, và đó cũng là nơi giá cao. Vậy **không phải "cũ" làm nhà đắt**, mà **vị trí** làm nhà vừa cũ vừa đắt.

> 💡 **Tương quan không phải nhân quả.** Hai thứ đi cùng nhau chưa chắc cái này gây ra cái kia. Có thể có một yếu tố thứ ba (ở đây là vị trí) đứng sau cả hai. Đây là một trong những bài học quan trọng nhất của khoa học dữ liệu.

---

## 8. Lưu biểu đồ thành file ảnh

```python
fig.savefig("ban_do_gia_nha.png", dpi=150, bbox_inches="tight")  # dpi: độ nét; tight: cắt viền thừa
```

Dùng để đưa vào báo cáo, slide, hoặc dán vào Notion.

---

## 9. Checklist một biểu đồ tốt ✅

- [ ] Có **tiêu đề** nói lên điều biểu đồ muốn khẳng định (ví dụ "Thu nhập càng cao, giá nhà càng cao"), không chỉ là "Biểu đồ 1".
- [ ] Có **nhãn trục** kèm **đơn vị** (USD, m², năm...).
- [ ] Dữ liệu nhiều điểm thì dùng `alpha` nhỏ để thấy chỗ dày đặc.
- [ ] Chọn đúng loại biểu đồ theo bảng ở mục 3.
- [ ] Không dùng biểu đồ tròn 3D. Không dùng quá nhiều màu.

---

## 10. Bài tập

1. Vẽ histogram cột `tuoi_nha` với 52 bins. Bạn có thấy điểm gì bất thường ở giá trị lớn nhất (52) không? Nó giống hiện tượng nào ở cột giá nhà?
2. Vẽ scatter `tong_phong` (trục x) và `tong_phong_ngu` (trục y). Hai cột này có liên quan mạnh không? Vì sao điều đó hợp lý?
3. Tạo cột `phong_moi_ho = tong_phong / so_ho`, rồi vẽ boxplot của nó. Bạn thấy vấn đề gì? Thử vẽ lại chỉ với các dòng có `phong_moi_ho < 15`.
4. Vẽ lại bản đồ ở mục 7, nhưng tô màu theo `thu_nhap` thay vì `gia_nha`. Hai bản đồ có giống nhau không?

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" → `Turn into → Toggle heading 3`.

```python
# 1
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["tuoi_nha"], bins=52)
plt.show()
print((df["tuoi_nha"] == 52).sum(), (df["tuoi_nha"] == 51).sum())   # 1052 32
# Có 1052 khu đúng 52 năm, trong khi 51 năm chỉ có 32 khu → cột cao vọt ở 52
# → cũng là GIỚI HẠN TRẦN, giống cột giá nhà ở 500.001.

# 2
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["tong_phong"], df["tong_phong_ngu"], s=3, alpha=0.2)
plt.show()
print(df[["tong_phong", "tong_phong_ngu"]].corr().round(2))   # hệ số 0.93
# Gần như một đường thẳng → liên quan rất mạnh. Hợp lý: khu nhiều phòng thì nhiều phòng ngủ.
# Trong ML, hai cột "nói cùng một điều" như vậy gây ra vấn đề đa cộng tuyến (sẽ học ở bài 2.6).

# 3
df["phong_moi_ho"] = df["tong_phong"] / df["so_ho"]
sns.boxplot(x=df["phong_moi_ho"]); plt.show()
# Hộp bị ép bẹp ở bên trái vì vài outlier cực lớn kéo trục ra rất xa.
sns.boxplot(x=df.loc[df["phong_moi_ho"] < 15, "phong_moi_ho"]); plt.show()
# Lọc bỏ outlier thì mới thấy rõ hình dạng phân bố. Xử lý outlier đúng cách: bài 1.6.

# 4
fig, ax = plt.subplots(figsize=(7, 6))
d = ax.scatter(df["kinh_do"], df["vi_do"], c=df["thu_nhap"], cmap="viridis", s=2, alpha=0.4)
fig.colorbar(d, label="Thu nhập")
plt.show()
# Rất giống bản đồ giá: vùng ven biển gần SF, LA vừa thu nhập cao vừa giá cao.
```

---

## 11. Flashcard

| Mặt trước | Mặt sau |
|---|---|
| Bộ tứ Anscombe dạy điều gì? | Thống kê giống nhau vẫn có thể là dữ liệu khác hẳn. Luôn vẽ! |
| Figure vs Axes? | Figure là tờ giấy. Axes là một ô biểu đồ trên tờ giấy. |
| Xem phân bố một biến số? | Histogram |
| Xem quan hệ 2 biến số? | Scatter |
| So sánh phân bố giữa các nhóm, tìm outlier? | Boxplot |
| `alpha` dùng để làm gì? | Làm điểm trong suốt, thấy được chỗ dày đặc. |
| `hue` trong Seaborn? | Tô màu theo nhóm. |
| Cột cao đột biến ở giá trị lớn nhất của histogram gợi ý gì? | Dữ liệu bị giới hạn trần. |
| Tương quan có phải nhân quả? | Không. Có thể có yếu tố thứ ba đứng sau. |

**Bài tiếp theo:** 1.6 – Làm sạch dữ liệu: dữ liệu thiếu, outlier, encoding.
