# Bài 0.2 – Cài đặt môi trường và notebook đầu tiên

> **Giai đoạn 0 – Làm quen** · Thời lượng: 2 buổi (mỗi buổi 2–3 giờ) · Hình thức mở đầu: *Đố vui trước khi học*

## 🎯 Sau bài này bạn sẽ

- Dùng thành thạo Google Colab: tạo notebook, chạy code, bật GPU, lưu bài.
- Cài Python trên máy Windows và tạo môi trường ảo cho từng dự án.
- Chạy được Jupyter Notebook trên máy mình (qua VS Code).
- Dùng Git để lưu lại lịch sử bài học.

**Chia buổi:**
- **Buổi 1:** Mục 1 → 5 (Google Colab, không cần cài gì).
- **Buổi 2:** Mục 6 → 10 (cài trên máy, môi trường ảo, Git).

---

## 1. Đố vui trước khi học (2 phút, chưa cần biết đáp án)

1. Bạn cài thư viện `pandas` phiên bản 2.2 cho dự án A. Dự án B cũ chỉ chạy được với `pandas` 1.5. Làm sao để cả hai cùng chạy trên một máy?
2. Bạn học ML trên chiếc laptop văn phòng không có card đồ họa rời. Bạn có huấn luyện được mạng nơ-ron không?
3. Đồng nghiệp gửi bạn file code, nhưng trên máy bạn báo lỗi `ModuleNotFoundError`. Nguyên nhân thường là gì?

> Ghi lại câu trả lời của bạn. Cuối bài (mục 11) có đáp án.

---

## 2. Vấn đề thực tế: "Máy tôi chạy được mà!"

Câu nói nổi tiếng nhất trong giới lập trình. Nguyên nhân gần như luôn là: **mỗi máy có một bộ "đồ nghề" khác nhau**. Khác phiên bản Python, khác phiên bản thư viện, thiếu thư viện.

Với người học ML, ta cần giải quyết 3 việc:

| Việc | Công cụ | Ví von |
|---|---|---|
| Có chỗ để viết và chạy code ML | **Google Colab** hoặc **Jupyter** | Căn bếp |
| Mỗi dự án có bộ đồ nghề riêng, không đá nhau | **venv** hoặc **conda** | Mỗi món ăn một bộ nồi riêng |
| Lưu lại từng phiên bản code, quay lại khi hỏng | **Git** | Nút "Undo" có nhớ lịch sử |

---

## 3. Khái niệm cần biết

### 3.1. Python

- **Định nghĩa:** Ngôn ngữ lập trình phổ biến nhất trong AI, vì cú pháp dễ đọc và có rất nhiều thư viện ML.
- **Ví dụ đời thực:** ChatGPT, YouTube, Instagram đều dùng Python ở một phần hệ thống.

### 3.2. Package / Library (Gói / Thư viện)

- **Định nghĩa:** Code người khác đã viết sẵn, ta cài vào để dùng lại, không phải viết từ đầu.
- **Ví dụ đời thực:** Như mua gói gia vị phở có sẵn thay vì tự rang quế, hồi.
- Công cụ cài thư viện của Python tên là **pip**. Ví dụ: `pip install pandas`.

### 3.3. Notebook (Sổ tay lập trình)

- **Định nghĩa:** File cho phép viết code **từng ô (cell)**, chạy từng ô, và thấy kết quả (số, bảng, biểu đồ) ngay bên dưới. Đuôi file là `.ipynb`.
- **Ví dụ đời thực:** Như vở nháp môn toán: viết một dòng, tính ngay, ghi chú bên cạnh.
- **Jupyter Notebook** là phần mềm notebook chạy trên máy bạn. **Google Colab** là notebook chạy trên máy chủ của Google.

### 3.4. GPU – Graphics Processing Unit (Card đồ họa)

- **Định nghĩa:** Con chip tính được hàng nghìn phép toán **cùng lúc**, rất hợp để huấn luyện mạng nơ-ron.
- **Ví dụ đời thực:** CPU như 8 giáo sư giải toán giỏi. GPU như 5.000 học sinh, mỗi em làm một phép cộng, nhưng làm cùng lúc. Với việc "cộng nhân hàng triệu con số", 5.000 học sinh nhanh hơn.
- Giai đoạn 1–2 chưa cần GPU. Giai đoạn 3 trở đi mới cần (sẽ học kỹ ở bài 3.8).

### 3.5. Virtual Environment (Môi trường ảo)

- **Định nghĩa:** Một thư mục riêng chứa Python và các thư viện của **một dự án**, tách biệt với dự án khác.
- **Ví dụ đời thực:** Mỗi dự án là một chiếc vali riêng. Vali đi biển có đồ bơi. Vali đi công tác có vest. Không lẫn lộn.

```
Máy tính của bạn
├── du_an_gia_nha/
│   └── .venv/   → pandas 2.2, scikit-learn 1.5
└── du_an_cu/
    └── .venv/   → pandas 1.5   (không ảnh hưởng dự án kia)
```

### 3.6. Git

- **Định nghĩa:** Phần mềm ghi lại lịch sử thay đổi của code, cho phép quay về bất kỳ phiên bản nào.
- **Ví dụ đời thực:** Như chức năng "Lịch sử phiên bản" của Google Docs, nhưng dành cho thư mục code.
- **GitHub** là trang web để lưu kho Git lên mạng và chia sẻ. Nó như Google Drive dành cho code.

---

## 4. BUỔI 1 – Google Colab: học ML không cần cài gì

### 4.1. Vì sao bắt đầu bằng Colab?

- Miễn phí, chỉ cần tài khoản Google.
- Đã cài sẵn gần hết thư viện ML: NumPy, Pandas, Scikit-learn, PyTorch...
- Có GPU miễn phí (giới hạn thời gian dùng).
- Máy yếu cũng học được.

**Nhược điểm:** mất file tạm khi phiên làm việc kết thúc, phụ thuộc mạng, GPU miễn phí có giới hạn.

### 4.2. Tạo notebook đầu tiên

1. Vào https://colab.research.google.com.
2. Bấm **New notebook** (Sổ tay mới).
3. Bấm vào tên `Untitled0.ipynb` ở góc trên bên trái, đổi thành `bai_0.2_notebook_dau_tien.ipynb`.

Notebook tự lưu vào Google Drive của bạn, thư mục **Colab Notebooks**.

### 4.3. Hai loại ô (cell)

| Loại ô | Dùng để | Cách tạo |
|---|---|---|
| **Code** | Viết và chạy code Python | Nút `+ Code` |
| **Text** | Viết ghi chú, tiêu đề (dùng Markdown) | Nút `+ Text` |

**Phím tắt cần nhớ:**

| Phím | Tác dụng |
|---|---|
| `Shift + Enter` | Chạy ô hiện tại, nhảy xuống ô dưới |
| `Ctrl + Enter` | Chạy ô hiện tại, đứng yên |
| `Ctrl + M` rồi `B` | Thêm ô code bên dưới |
| `Ctrl + M` rồi `D` | Xóa ô |
| `Ctrl + /` | Bật/tắt comment dòng đang chọn |

### 4.4. Chạy code đầu tiên

Dán vào ô code thứ nhất và bấm `Shift + Enter`:

```python
import sys                                   # thư viện có sẵn, chứa thông tin về Python
print("Xin chào Machine Learning!")          # in một câu ra màn hình
print("Phiên bản Python:", sys.version.split()[0])  # xem đang dùng Python bản nào

tong = sum(range(1, 101))                    # cộng các số từ 1 đến 100
print("Tổng 1 đến 100 =", tong)
```

**Kết quả mẫu** (số phiên bản trên máy bạn có thể khác):

```
Xin chào Machine Learning!
Phiên bản Python: 3.12.12
Tổng 1 đến 100 = 5050
```

### 4.5. Thử dùng thư viện ML có sẵn

Ô code thứ hai:

```python
import numpy as np                  # thư viện tính toán số (sẽ học kỹ ở bài 1.3)
import matplotlib.pyplot as plt     # thư viện vẽ biểu đồ (sẽ học kỹ ở bài 1.5)

dien_tich = [50, 65, 80, 100]                 # diện tích 4 căn nhà (m²)
gia_nha = np.array([2.5, 3.1, 4.0, 5.2])      # giá tương ứng (tỷ đồng)

print("Giá nhà trung bình:", gia_nha.mean(), "tỷ")   # tính trung bình

plt.plot(dien_tich, gia_nha, "o-")            # vẽ điểm và nối đường
plt.xlabel("Diện tích (m²)")                  # nhãn trục ngang
plt.ylabel("Giá (tỷ đồng)")                   # nhãn trục dọc
plt.title("Nhà càng rộng, giá càng cao")
plt.show()                                    # hiện biểu đồ
```

**Kết quả mẫu:**

```
Giá nhà trung bình: 3.7 tỷ
```

Kèm một biểu đồ đường đi lên từ trái sang phải. Đây chính là bài toán **Linear Regression (Hồi quy tuyến tính)** bạn sẽ giải ở bài 2.2: tìm đường thẳng đi gần các điểm nhất để đoán giá nhà mới.

### 4.6. Lệnh hệ thống trong Colab

Thêm dấu `!` phía trước để chạy lệnh như trong cửa sổ dòng lệnh:

```python
!pip install underthesea      # cài thêm thư viện tách từ tiếng Việt (sẽ dùng ở bài NLP1.1)
!ls                           # liệt kê file trong thư mục hiện tại
```

> ⚠️ Thư viện cài bằng `!pip install` **mất đi** khi phiên Colab kết thúc. Mỗi lần mở lại notebook phải chạy lại ô cài đặt. Vì vậy, luôn đặt ô `!pip install` ở **đầu notebook**.

### 4.7. Bật GPU miễn phí

1. Menu **Runtime** (Thời gian chạy) → **Change runtime type** (Thay đổi loại thời gian chạy).
2. Chọn **T4 GPU** → **Save**.
3. Kiểm tra bằng ô code:

```python
import torch                                   # thư viện Deep Learning (sẽ học kỹ ở bài 3.8)
print("Có GPU không?", torch.cuda.is_available())   # True nghĩa là đã có GPU
if torch.cuda.is_available():
    print("Tên GPU:", torch.cuda.get_device_name(0))  # in tên card
```

**Kết quả mẫu khi đã bật GPU:**

```
Có GPU không? True
Tên GPU: Tesla T4
```

> 💡 Chỉ bật GPU khi thật sự cần (từ Giai đoạn 3). Colab giới hạn thời gian GPU miễn phí, dùng hết sẽ phải đợi.

### 4.8. Đọc file từ Google Drive

```python
from google.colab import drive      # công cụ kết nối Drive, chỉ có trên Colab
drive.mount('/content/drive')       # Colab sẽ hỏi quyền truy cập, bấm cho phép

# Sau khi kết nối, file trong Drive nằm ở đường dẫn:
# /content/drive/MyDrive/<tên thư mục>/<tên file>
```

**Kết quả mẫu:**

```
Mounted at /content/drive
```

Từ giờ, hãy để dataset trong Drive (ví dụ thư mục `MyDrive/hoc_ml/data/`) để không bị mất sau mỗi phiên.

---

## 5. Bài tập Buổi 1

1. Tạo notebook `bai_0.2_notebook_dau_tien.ipynb` gồm:
   - Một ô **Text** tiêu đề: `# Notebook đầu tiên của tôi`.
   - Các ô code ở mục 4.4 và 4.5.
2. Sửa dữ liệu ở mục 4.5: thêm căn nhà thứ 5 rộng 120 m², giá 6.0 tỷ. Chạy lại. Giá trung bình mới là bao nhiêu?
3. Bật GPU và chạy ô kiểm tra ở mục 4.7. Sau đó **tắt lại** (chọn CPU).

**Đáp án câu 2:** trung bình mới = (2.5 + 3.1 + 4.0 + 5.2 + 6.0) / 5 = 20.8 / 5 = **4.16 tỷ**. Python có thể in `4.16` hoặc `4.159999999999999` – cả hai đều đúng (lý do: máy tính lưu số thập phân xấp xỉ, sẽ gặp lại ở bài 1.3).

---

## 6. BUỔI 2 – Cài Python trên máy Windows

> Vì sao vẫn cần cài trên máy dù đã có Colab? Vì đến Phần C (triển khai sản phẩm), bạn sẽ chạy API, Docker trên máy mình. Làm quen từ sớm sẽ đỡ bỡ ngỡ.

### 6.1. Cài Python

1. Vào https://www.python.org/downloads/ và tải bản **Python 3.12** (bản ổn định, hầu hết thư viện ML đều hỗ trợ).
2. Chạy file cài đặt. **Quan trọng:** tick ô **"Add python.exe to PATH"** ở màn hình đầu tiên.
3. Bấm **Install Now**.

### 6.2. Kiểm tra

Mở **PowerShell** (bấm phím Windows, gõ `PowerShell`, Enter) rồi gõ:

```powershell
python --version
pip --version
```

**Kết quả mẫu:**

```
Python 3.12.10
pip 25.0.1 from C:\Users\<tên bạn>\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip (python 3.12)
```

### 6.3. Cài VS Code

1. Tải tại https://code.visualstudio.com và cài đặt.
2. Mở VS Code → bấm biểu tượng **Extensions** (ô vuông bên trái) → cài 2 extension: **Python** và **Jupyter** (đều của Microsoft).

---

## 7. Môi trường ảo với venv

### 7.1. Tạo thư mục dự án và môi trường ảo

Trong PowerShell:

```powershell
mkdir C:\hoc_ml                 # tạo thư mục học tập
cd C:\hoc_ml                    # đi vào thư mục đó
python -m venv .venv            # tạo môi trường ảo tên ".venv" bên trong thư mục
.\.venv\Scripts\Activate.ps1    # kích hoạt (bật) môi trường ảo
```

Khi kích hoạt thành công, đầu dòng lệnh có thêm `(.venv)`:

```
(.venv) PS C:\hoc_ml>
```

⚠️ Nếu gặp lỗi `running scripts is disabled on this system`, chạy lệnh dưới đây **một lần**, gõ `Y` để đồng ý, rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 7.2. Cài thư viện cho khóa học

Khi đang thấy `(.venv)` ở đầu dòng:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel
```

Kiểm tra:

```powershell
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
```

**Kết quả mẫu** (phiên bản có thể khác):

```
scikit-learn 1.7.2
```

### 7.3. Lưu danh sách thư viện: requirements.txt

```powershell
pip freeze > requirements.txt     # ghi toàn bộ thư viện + phiên bản vào file
```

Người khác (hoặc chính bạn trên máy mới) chỉ cần:

```powershell
pip install -r requirements.txt   # cài đúng y hệt danh sách
```

Đây là cách giải quyết câu "Máy tôi chạy được mà!".

### 7.4. Tắt môi trường ảo

```powershell
deactivate
```

### 7.5. Còn conda thì sao?

- **Conda** (dùng qua bản nhỏ gọn **Miniconda**) cũng tạo môi trường ảo, và còn cài được cả phiên bản Python khác nhau, cùng thư viện không phải Python.
- Ví dụ lệnh tương đương:

```powershell
conda create -n hoc_ml python=3.12    # tạo môi trường tên hoc_ml
conda activate hoc_ml                 # bật môi trường
```

| | venv | conda |
|---|---|---|
| Có sẵn trong Python | Có | Không, phải cài Miniconda |
| Đổi phiên bản Python | Không | Có |
| Nhẹ, đơn giản | Có | Nặng hơn |
| Khuyên dùng trong khóa học | **Mặc định** | Khi cần nhiều bản Python |

**Khóa học này dùng venv.** Chỉ chọn một trong hai, đừng trộn lẫn để tránh rối.

---

## 8. Chạy Jupyter Notebook trên máy (qua VS Code)

1. Trong VS Code: **File → Open Folder** → chọn `C:\hoc_ml`.
2. **File → New File** → chọn **Jupyter Notebook**. Lưu tên `notebook_dau_tien.ipynb`.
3. Bấm **Select Kernel** (góc trên bên phải) → **Python Environments** → chọn `.venv`.
4. Dán lại code ở mục 4.4, bấm `Shift + Enter`.

> **Kernel** (nhân chạy) là "bộ máy Python" đứng sau notebook. Chọn sai kernel là nguyên nhân số 1 của lỗi `ModuleNotFoundError`, vì notebook dùng một Python không có thư viện bạn đã cài.

---

## 9. Git: lưu lịch sử bài học

### 9.1. Cài và cấu hình

1. Tải tại https://git-scm.com/download/win, cài với các tùy chọn mặc định.
2. Mở PowerShell mới, khai báo tên và email (chỉ làm một lần):

```powershell
git config --global user.name "Tên của bạn"
git config --global user.email "email@cua-ban.com"
```

### 9.2. Bốn lệnh dùng hằng ngày

```
  Thư mục làm việc ──git add──► Khu chuẩn bị ──git commit──► Lịch sử (kho Git)
   (file bạn sửa)               (chọn file sẽ lưu)           (ảnh chụp đã lưu)
```

```powershell
cd C:\hoc_ml
git init                                  # biến thư mục thành kho Git (chỉ làm 1 lần)
git status                                # xem file nào đã thay đổi
git add notebook_dau_tien.ipynb           # chọn file muốn lưu
git commit -m "Bai 0.2: notebook dau tien" # chụp ảnh và ghi chú
git log --oneline                         # xem lịch sử các lần lưu
```

**Kết quả mẫu của `git log --oneline`:**

```
a1b2c3d (HEAD -> master) Bai 0.2: notebook dau tien
```

### 9.3. Đừng đưa môi trường ảo vào Git

Thư mục `.venv` rất nặng và tạo lại được bằng `requirements.txt`. Tạo file tên `.gitignore` trong `C:\hoc_ml` với nội dung:

```
.venv/
.ipynb_checkpoints/
__pycache__/
data/
```

Dòng `data/` để không đưa dataset lớn lên Git.

### 9.4. (Tùy chọn) Đưa lên GitHub

1. Tạo tài khoản tại https://github.com, bấm **New repository**, đặt tên `hoc_ml`, bấm **Create**.
2. GitHub sẽ hiện sẵn các lệnh dạng `git remote add origin ...` và `git push ...`. Copy và chạy chúng trong PowerShell.

Lợi ích: có "portfolio" để xin việc, và không sợ mất bài khi hỏng máy.

---

## 10. Lỗi thường gặp và cách sửa

| Thông báo lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `'python' is not recognized...` | Quên tick "Add to PATH" khi cài | Cài lại Python, nhớ tick ô đó |
| Gõ `python` thì mở Microsoft Store | Windows dùng lối tắt giả | Settings → tìm "App execution aliases" → tắt 2 dòng `python.exe`, `python3.exe` |
| `running scripts is disabled...` | PowerShell chặn script | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: No module named 'sklearn'` | Chưa cài thư viện, hoặc chưa bật venv, hoặc notebook chọn sai kernel | Bật `(.venv)` rồi `pip install scikit-learn`; kiểm tra kernel |
| Colab báo không tìm thấy file vừa tải lên | Phiên Colab đã khởi động lại, file tạm bị xóa | Để file trong Google Drive và mount Drive (mục 4.8) |
| Colab báo hết GPU | Dùng hết hạn mức miễn phí | Chuyển về CPU, quay lại sau vài giờ |

---

## 11. Đáp án đố vui đầu bài

1. Tạo **hai môi trường ảo** riêng cho hai dự án. Mỗi môi trường có phiên bản `pandas` riêng.
2. **Được.** Dùng GPU miễn phí của Google Colab. Với ML cổ điển (Giai đoạn 2), CPU laptop là đủ.
3. Thường là **chưa cài thư viện**, hoặc **chưa bật đúng môi trường ảo / chọn sai kernel**. Cách phòng: dùng `requirements.txt`.

---

## 12. Thử thách "Tìm lỗi" (10 phút)

Bạn Lan chạy các lệnh sau trong PowerShell và gặp lỗi `ModuleNotFoundError: No module named 'pandas'` ở dòng cuối. Có **2 lỗi**. Đó là gì?

```powershell
cd C:\hoc_ml
python -m venv .venv
pip install pandas
.\.venv\Scripts\Activate.ps1
python -c "import panda"
```

### Đáp án

> Mẹo Notion: chọn tiêu đề "Đáp án" và bấm `Turn into → Toggle heading 3` để giấu đáp án.

1. Cài `pandas` **trước khi** bật môi trường ảo, nên thư viện được cài vào Python chung của máy, không vào `.venv`. Phải đổi thứ tự: bật `.venv` trước, rồi mới `pip install pandas`.
2. Gõ sai tên: `import panda` thiếu chữ `s`. Phải là `import pandas`.

---

## 13. Flashcard ôn tập

| Mặt trước | Mặt sau |
|---|---|
| Colab khác Jupyter ở đâu? | Colab chạy trên máy chủ Google, có sẵn thư viện và GPU. Jupyter chạy trên máy bạn. |
| Phím chạy một ô trong notebook? | `Shift + Enter` |
| Vì sao cần môi trường ảo? | Để mỗi dự án có thư viện, phiên bản riêng, không xung đột. |
| Lệnh tạo và bật venv trên Windows? | `python -m venv .venv` rồi `.\.venv\Scripts\Activate.ps1` |
| `requirements.txt` để làm gì? | Ghi danh sách thư viện, để cài lại y hệt bằng `pip install -r requirements.txt`. |
| Kernel là gì? | "Bộ máy Python" chạy notebook. Sai kernel → không tìm thấy thư viện. |
| Ba lệnh Git cơ bản để lưu bài? | `git add`, `git commit -m "..."`, `git log` để xem lại |
| Vì sao không đưa `.venv` lên Git? | Nặng, và tạo lại được từ `requirements.txt`. |

---

## ✅ Đầu ra Giai đoạn 0 – Tự kiểm tra

- [ ] Có notebook `bai_0.2_notebook_dau_tien.ipynb` chạy được trên Colab, lưu trong Drive.
- [ ] `python --version` trên máy in ra 3.12.x.
- [ ] Có thư mục `C:\hoc_ml` với `.venv`, `requirements.txt`, `.gitignore`.
- [ ] Mở notebook trong VS Code, chọn kernel `.venv`, chạy được code ở mục 4.4 và 4.5.
- [ ] `git log --oneline` hiện ít nhất 1 lần commit.

Hoàn thành hết? Chúc mừng bạn đã xong **Giai đoạn 0**! 🎉

**Bài tiếp theo:** 1.1 – Python cốt lõi: biến, list, dict, vòng lặp, comprehension.
