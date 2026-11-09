# Bài 1.3 – NumPy: mảng và phép toán vector hóa

> **Giai đoạn 1 – Python cho ML & Data Science** · Thời lượng: 2 buổi · Hình thức mở đầu: *Thử thách có tính giờ*

## 🎯 Sau bài này bạn sẽ

- Hiểu vì sao mọi thư viện ML đều xây trên NumPy.
- Tạo và thao tác mảng 1 chiều, 2 chiều.
- Tính toán trên cả mảng mà **không cần vòng lặp** (vector hóa).
- Lọc dữ liệu bằng điều kiện (boolean mask), tính thống kê theo hàng/cột.
- Hiểu **broadcasting** và **shape**, hai nguồn gây lỗi nhiều nhất cho người mới.

**Chia buổi:**
- **Buổi 1:** Mục 1 → 5 (tạo mảng, vector hóa, thống kê).
- **Buổi 2:** Mục 6 → 11 (mảng 2 chiều, lọc, broadcasting, bài tập).

---

## 1. Thử thách có tính giờ ⏱️

**Đề:** Tính tổng bình phương của 1 triệu số: 0² + 1² + 2² + ... + 999.999².

Dán vào Colab và chạy:

```python
import time                 # thư viện đo thời gian
import numpy as np          # NumPy, quy ước luôn đặt tên tắt là np

n = 1_000_000               # dấu _ chỉ để dễ đọc, Python bỏ qua

# --- Cách 1: vòng lặp Python ---
ds = list(range(n))
t0 = time.time()                       # bấm giờ
tong_vong_lap = 0
for x in ds:
    tong_vong_lap += x * x
t1 = time.time()

# --- Cách 2: NumPy ---
mang = np.arange(n, dtype=np.int64)    # mảng 0, 1, 2, ..., 999999
tong_numpy = (mang * mang).sum()       # bình phương CẢ MẢNG một lúc rồi cộng
t2 = time.time()

print("Vòng lặp:", tong_vong_lap, f"- {t1 - t0:.4f} giây")
print("NumPy   :", tong_numpy, f"- {t2 - t1:.4f} giây")
print(f"NumPy nhanh hơn khoảng {(t1 - t0) / (t2 - t1):.0f} lần")
```

**Kết quả mẫu** (thời gian trên máy bạn sẽ khác):

```
Vòng lặp: 333332833333500000 - 0.0822 giây
NumPy   : 333332833333500000 - 0.0048 giây
NumPy nhanh hơn khoảng 17 lần
```

Cùng kết quả, nhưng NumPy nhanh hơn **hàng chục lần** (thường 10–100 lần tùy máy). Với dữ liệu ML thật (hàng triệu dòng, hàng nghìn cột, chạy lặp hàng trăm lần), khác biệt là **vài phút so với vài giờ**.

**Vì sao nhanh?**

```
List Python:  [ ●→int ][ ●→int ][ ●→int ] ...  mỗi ô là một "mũi tên" trỏ tới số nằm rải rác
               Python phải kiểm tra kiểu từng phần tử, từng bước một.

Mảng NumPy:   [ 0 | 1 | 2 | 3 | 4 | ... ]      các số nằm sát nhau, cùng một kiểu
               Phép tính chạy bằng code C đã tối ưu, xử lý cả khối một lúc.
```

---

## 2. Khái niệm

### 2.1. NumPy

- **Định nghĩa:** Thư viện Python để làm toán với **mảng số** thật nhanh.
- **Ví dụ đời thực:** Pandas, Scikit-learn, OpenCV đều dùng NumPy bên trong. Một bức ảnh trong OpenCV **chính là** một mảng NumPy (sẽ gặp lại ở bài CV1.1).

### 2.2. ndarray – N-dimensional array (Mảng nhiều chiều)

- **Định nghĩa:** Kiểu dữ liệu chính của NumPy. Một khối số **cùng kiểu**, có thể 1 chiều, 2 chiều, hoặc nhiều hơn.
- **Ví dụ đời thực:**

| Số chiều | Hình dung | Ví dụ ML |
|---|---|---|
| 1 chiều | Một hàng số | Giá của 4 căn nhà |
| 2 chiều | Bảng Excel | 4 căn nhà × 3 đặc điểm |
| 3 chiều | Chồng nhiều bảng | Ảnh màu: cao × rộng × 3 màu (RGB) |

### 2.3. Vectorization (Vector hóa)

- **Định nghĩa:** Viết phép tính cho **cả mảng** thay vì từng phần tử bằng vòng lặp.
- **Ví dụ đời thực:** Thay vì đóng dấu từng tờ giấy, bạn dùng máy in in cả xấp một lúc.

---

## 3. Tạo mảng

```python
gia_nha = np.array([2.5, 3.1, 4.0, 5.2])   # tạo mảng từ list (giá nhà, tỷ đồng)
print(gia_nha, gia_nha.dtype, gia_nha.shape)   # dtype: kiểu số; shape: kích thước

print(np.zeros(3),               # 3 số 0
      np.ones(3),                # 3 số 1
      np.arange(0, 10, 2),       # từ 0 đến trước 10, bước nhảy 2
      np.linspace(0, 1, 5))      # 5 số cách đều từ 0 đến 1 (tính cả 1)
```

**Kết quả mẫu:**

```
[2.5 3.1 4.  5.2] float64 (4,)
[0. 0. 0.] [1. 1. 1.] [0 2 4 6 8] [0.   0.25 0.5  0.75 1.  ]
```

- `float64` = số thực 64 bit. `4.` chính là `4.0`, NumPy viết gọn.
- `shape (4,)` = mảng 1 chiều có 4 phần tử. Dấu phẩy thừa là để nhấn mạnh đây là tuple 1 phần tử.

---

## 4. Phép toán vector hóa

```python
dien_tich = np.array([50, 65, 80, 100])        # diện tích 4 căn (m²)

print(gia_nha * 1000)                          # đổi tỷ → triệu: nhân CẢ MẢNG với 1000
gia_m2 = gia_nha * 1000 / dien_tich            # chia TỪNG CẶP phần tử tương ứng
print(gia_m2.round(1))                         # giá mỗi m² (triệu đồng), làm tròn 1 số lẻ

print(gia_nha + np.array([0.1, 0.2, 0.3, 0.4]))   # cộng từng cặp
```

**Kết quả mẫu:**

```
[2500. 3100. 4000. 5200.]
[50.  47.7 50.  52. ]
[2.6 3.3 4.3 5.6]
```

Hình dung phép chia từng cặp:

```
gia_nha*1000:  [ 2500 , 3100 , 4000 , 5200 ]
                   ÷      ÷      ÷      ÷
dien_tich:     [   50 ,   65 ,   80 ,  100 ]
                   =      =      =      =
gia_m2:        [   50 , 47.7 ,   50 ,   52 ]
```

**Quy tắc:** hai mảng cùng kích thước thì phép toán áp dụng **từng cặp vị trí tương ứng**. Trong list Python, `[1,2] * 2` lại cho `[1,2,1,2]`. NumPy khác hẳn!

---

## 5. Thống kê nhanh

```python
print(gia_nha.sum(),       # tổng
      gia_nha.mean(),      # trung bình
      gia_nha.min(),       # nhỏ nhất
      gia_nha.max(),       # lớn nhất
      gia_nha.argmax())    # VỊ TRÍ của phần tử lớn nhất
print(gia_nha.std().round(3))   # độ lệch chuẩn (sẽ học kỹ ở bài 1.7)
```

**Kết quả mẫu:**

```
14.8 3.7 2.5 5.2 3
1.017
```

`argmax()` trả về `3` nghĩa là căn ở vị trí 3 (căn thứ tư) đắt nhất. Bạn sẽ dùng `argmax` rất nhiều: mô hình phân loại trả về xác suất cho từng lớp, `argmax` chọn lớp có xác suất cao nhất (sẽ gặp lại ở bài 3.13).

---

## 6. Mảng 2 chiều: dữ liệu dạng bảng

Trong ML, dữ liệu đầu vào thường được gọi là **X**: mỗi **hàng** là một mẫu (một căn nhà), mỗi **cột** là một **feature (đặc trưng)**, tức một thông tin mô tả mẫu.

```python
# 4 căn nhà × 3 đặc trưng: diện tích (m²), số phòng ngủ, khoảng cách tới trung tâm (km)
X = np.array([
    [50,  2, 5],
    [65,  2, 3],
    [80,  3, 7],
    [100, 3, 2],
])
print(X.shape, X.ndim)    # (số hàng, số cột), số chiều
```

**Kết quả mẫu:**

```
(4, 3) 2
```

### 6.1. Lấy dữ liệu: `X[hàng, cột]`

```
              cột 0   cột 1   cột 2
             (m²)    (phòng)  (km)
   hàng 0  [  50       2       5  ]
   hàng 1  [  65       2       3  ]
   hàng 2  [  80       3       7  ]
   hàng 3  [ 100       3       2  ]
```

```python
print(X[0])          # hàng 0: toàn bộ thông tin căn đầu tiên
print(X[:, 0])       # cột 0: diện tích của mọi căn. Dấu ":" nghĩa là "lấy hết"
print(X[1, 2])       # hàng 1, cột 2: khoảng cách của căn thứ hai
print(X[:2, :2])     # 2 hàng đầu, 2 cột đầu
```

**Kết quả mẫu:**

```
[50  2  5]
[ 50  65  80 100]
3
[[50  2]
 [65  2]]
```

### 6.2. `axis`: tính theo cột hay theo hàng?

Đây là chỗ người mới hay nhầm nhất.

```
axis=0: "đi dọc xuống", gộp các HÀNG lại → mỗi CỘT ra 1 số
axis=1: "đi ngang qua", gộp các CỘT lại → mỗi HÀNG ra 1 số

          axis=1 →
axis=0  [  50   2   5 ]
  ↓     [  65   2   3 ]
        [  80   3   7 ]
        [ 100   3   2 ]
```

```python
print(X.mean(axis=0))            # trung bình mỗi cột (mỗi đặc trưng)
print(X.mean(axis=1).round(2))   # trung bình mỗi hàng (ít ý nghĩa ở đây, chỉ để minh họa)
```

**Kết quả mẫu:**

```
[73.75  2.5   4.25]
[19.   23.33 30.   35.  ]
```

Đọc kết quả: diện tích trung bình 73,75 m², trung bình 2,5 phòng, trung bình cách trung tâm 4,25 km.

**Mẹo nhớ:** `axis=0` là trục **biến mất**. Bảng (4, 3) tính theo `axis=0` thì còn lại 3 số.

---

## 7. Boolean mask: lọc dữ liệu bằng điều kiện

```python
print(gia_nha > 3.5)                       # so sánh cả mảng → mảng True/False
print(gia_nha[gia_nha > 3.5])              # chỉ giữ phần tử ở vị trí True

# Kết hợp điều kiện: & (và), | (hoặc). BẮT BUỘC có ngoặc quanh mỗi điều kiện
print(dien_tich[(gia_nha > 3) & (dien_tich < 90)])

print((gia_nha > 3.5).sum())               # đếm số căn trên 3,5 tỷ (True tính là 1)

# np.where(điều kiện, giá trị nếu đúng, giá trị nếu sai)
print(np.where(gia_nha > 3.5, "Đắt", "Vừa"))
```

**Kết quả mẫu:**

```
[False False  True  True]
[4.  5.2]
[65 80]
2
['Vừa' 'Vừa' 'Đắt' 'Đắt']
```

Hình dung:

```
gia_nha:        [ 2.5 ,  3.1 ,  4.0 ,  5.2 ]
gia_nha > 3.5:  [ False, False, True , True ]   ← cái "mặt nạ"
kết quả:                        [ 4.0 ,  5.2 ]   ← chỉ lộ ra chỗ True
```

> ⚠️ Dùng `&`, `|` chứ **không** dùng `and`, `or` với mảng NumPy. Viết `gia_nha > 3 and dien_tich < 90` sẽ báo lỗi `ValueError: The truth value of an array... is ambiguous`.

---

## 8. Broadcasting: khi hai mảng khác kích thước

### 8.1. Trực giác

`gia_nha * 1000` ở mục 4: một mảng 4 phần tử nhân với **một con số**. NumPy tự "kéo giãn" con số 1000 thành `[1000, 1000, 1000, 1000]`. Đó là **broadcasting (lan truyền)**.

Quy tắc này còn áp dụng cho mảng: bảng (4, 3) trừ đi một hàng (3,) → hàng đó được "sao chép" xuống cả 4 hàng.

```
X (4×3)          trừ     X.mean(axis=0) (3,)  →  được kéo giãn thành (4×3)
[ 50  2  5]              [73.75 2.5 4.25]        [73.75 2.5 4.25]
[ 65  2  3]                                      [73.75 2.5 4.25]
[ 80  3  7]                                      [73.75 2.5 4.25]
[100  3  2]                                      [73.75 2.5 4.25]
```

```python
print(X - X.mean(axis=0))   # mỗi giá trị trừ đi trung bình của cột chứa nó
```

**Kết quả mẫu:**

```
[[-23.75  -0.5    0.75]
 [ -8.75  -0.5   -1.25]
 [  6.25   0.5    2.75]
 [ 26.25   0.5   -2.25]]
```

Đọc kết quả: căn đầu tiên nhỏ hơn diện tích trung bình 23,75 m². Phép "trừ trung bình" này là bước đầu của **chuẩn hóa dữ liệu**, cực kỳ quan trọng trong ML (sẽ học kỹ ở bài 1.11).

---

## 9. Shape và reshape

```python
a = np.arange(6)              # [0 1 2 3 4 5], shape (6,)
print(a.reshape(2, 3))        # sắp lại thành 2 hàng × 3 cột
print(a.reshape(-1, 1).shape) # -1 = "NumPy tự tính". Thành 1 cột, 6 hàng
```

**Kết quả mẫu:**

```
[[0 1 2]
 [3 4 5]]
(6, 1)
```

**Vì sao cần?** Scikit-learn yêu cầu X luôn là **bảng 2 chiều**, kể cả khi chỉ có 1 đặc trưng. Nếu bạn đưa mảng 1 chiều `dien_tich` vào, nó báo lỗi `Expected 2D array, got 1D array instead`. Cách sửa: `dien_tich.reshape(-1, 1)`. Bạn sẽ gặp lỗi này ở bài 2.2, giờ thì bạn đã biết cách sửa.

---

## 10. Ba điều cần biết thêm

### 10.1. Số ngẫu nhiên có thể lặp lại

```python
rng = np.random.default_rng(42)     # 42 là "hạt giống" (seed): cùng seed → cùng dãy số
print(rng.integers(1, 7, size=5))   # 5 lần tung xúc xắc (từ 1 đến 6)
print(rng.normal(loc=3.5, scale=0.5, size=3).round(2))  # 3 số quanh 3.5 (sẽ học kỹ ở bài 1.10)
```

**Kết quả mẫu** (cùng seed 42 sẽ ra đúng như vậy):

```
[1 5 4 3 3]
[3.97 2.52 2.85]
```

Trong ML, luôn đặt seed để **người khác chạy lại ra đúng kết quả của bạn**.

### 10.2. Số thực không chính xác tuyệt đối

```python
print(0.1 + 0.2, 0.1 + 0.2 == 0.3, np.isclose(0.1 + 0.2, 0.3))
```

**Kết quả mẫu:**

```
0.30000000000000004 False True
```

Máy tính lưu số thập phân dạng nhị phân nên có sai số rất nhỏ. **Không so sánh số thực bằng `==`**, hãy dùng `np.isclose()`.

### 10.3. Cắt lát là "cửa sổ", không phải bản sao

```python
b = gia_nha[:2]      # b là cửa sổ nhìn vào gia_nha, KHÔNG phải bản sao
b[0] = 999
print(gia_nha)       # gia_nha cũng bị đổi!
```

**Kết quả mẫu:**

```
[999.    3.1   4.    5.2]
```

Giống câu 1 mini-game bài 1.1. Muốn bản sao độc lập: `b = gia_nha[:2].copy()`.

---

## 11. Bài tập

Chạy lại từ đầu để có dữ liệu sạch:

```python
gia_nha = np.array([2.5, 3.1, 4.0, 5.2])
dien_tich = np.array([50, 65, 80, 100])
so_phong = np.array([2, 2, 3, 3])
```

1. Tính giá trung bình mỗi m² (triệu đồng/m²) của cả 4 căn. Làm tròn 2 số lẻ.
2. Căn **rẻ nhất** có diện tích bao nhiêu? (Gợi ý: `argmin`.)
3. Giá trung bình của các căn **có 3 phòng**?
4. **Chuẩn hóa** `gia_nha`: lấy mỗi giá trừ trung bình, rồi chia cho độ lệch chuẩn. Kiểm tra: kết quả có trung bình ≈ 0 và độ lệch chuẩn = 1.
5. Tạo bảng điểm ngẫu nhiên 5 học sinh × 3 môn (điểm nguyên 0–10) với `rng.integers(0, 11, size=(5, 3))`. Tính điểm trung bình **mỗi môn** và **mỗi học sinh**.

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" → `Turn into → Toggle heading 3`.

```python
# 1
print((gia_nha * 1000 / dien_tich).mean().round(2))     # 49.92

# 2
print(dien_tich[gia_nha.argmin()])                       # 50

# 3
print(gia_nha[so_phong == 3].mean())                     # 4.6

# 4
chuan = (gia_nha - gia_nha.mean()) / gia_nha.std()
print(chuan.round(2))                                    # [-1.18 -0.59  0.29  1.47]
print(np.isclose(chuan.mean(), 0), chuan.std())          # True 1.0

# 5 (số ngẫu nhiên nên kết quả của bạn sẽ khác)
diem = rng.integers(0, 11, size=(5, 3))
print(diem.mean(axis=0))    # trung bình mỗi môn (gộp các hàng)
print(diem.mean(axis=1))    # trung bình mỗi học sinh (gộp các cột)
```

---

## 12. Flashcard

| Mặt trước | Mặt sau |
|---|---|
| Vì sao NumPy nhanh hơn list? | Số nằm liền nhau, cùng kiểu, tính bằng code C cho cả khối. |
| `shape (4, 3)` nghĩa là? | 4 hàng (mẫu) × 3 cột (đặc trưng). |
| `X[:, 0]` lấy gì? | Toàn bộ cột 0. |
| `axis=0` tính theo? | Dọc xuống, mỗi cột ra 1 số. |
| Lọc phần tử lớn hơn 3? | `a[a > 3]` |
| Kết hợp điều kiện trong NumPy? | `(a > 1) & (b < 5)`, không dùng `and`. |
| Broadcasting là gì? | NumPy tự kéo giãn mảng nhỏ cho khớp mảng lớn. |
| Sửa lỗi "Expected 2D array"? | `a.reshape(-1, 1)` |
| Vì sao đặt seed? | Để kết quả ngẫu nhiên lặp lại được. |
| So sánh hai số thực? | `np.isclose(a, b)`, không dùng `==`. |

**Bài tiếp theo:** 1.4 – Pandas: đọc, lọc, nhóm, nối dữ liệu.
