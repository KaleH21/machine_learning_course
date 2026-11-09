# Bài 0.1 – AI, Machine Learning, Deep Learning khác nhau thế nào?

> **Giai đoạn 0 – Làm quen** · Thời lượng: 1 buổi (2–3 giờ) · Hình thức mở đầu: *Câu chuyện*

## 🎯 Sau bài này bạn sẽ

- Phân biệt được AI, Machine Learning và Deep Learning bằng lời của chính mình.
- Giải thích được vì sao "cho máy học từ ví dụ" khác với "viết luật bằng if-else".
- Biết 3 kiểu học chính của máy và các lĩnh vực lớn của AI.
- Chạy thử đoạn code ML đầu tiên trên Google Colab (chỉ để xem, chưa cần hiểu hết).

---

## 1. Câu chuyện mở đầu: Anh Nam và hộp thư đầy rác

Anh Nam là lập trình viên ở một công ty thương mại điện tử. Sếp giao việc: *"Lọc email lừa đảo cho khách hàng."*

**Ngày 1.** Anh Nam viết một luật:

```
nếu email chứa "trúng thưởng" → SPAM
```

Chạy tốt. Anh Nam vui.

**Ngày 3.** Kẻ lừa đảo đổi chữ: "trúng thuởng", "tr.úng th.ưởng", "nhận quà miễn phí". Luật cũ bó tay. Anh Nam thêm 10 luật mới.

**Ngày 30.** Anh Nam có 500 luật. Luật này đá luật kia. Email của sếp có chữ "thưởng Tết" cũng bị chặn. Anh Nam mệt mỏi.

**Ngày 31.** Một đồng nghiệp gợi ý:

> *"Sao anh không đưa cho máy 10.000 email đã được đánh dấu SPAM / Bình thường, rồi để máy **tự tìm ra quy luật**?"*

Đó chính là **Machine Learning**. Và đó là toàn bộ ý tưởng của khóa học này.

---

## 2. Trực giác: hai cách dạy một đứa trẻ nhận ra con mèo

| Cách | Bạn làm gì | Đứa trẻ làm gì |
|---|---|---|
| **Cách 1: Đưa luật** | "Con mèo có 4 chân, có ria, tai nhọn, kêu meo meo" | Thuộc luật. Gặp con mèo cụt tai → bối rối. |
| **Cách 2: Đưa ví dụ** | Chỉ cho xem 100 con mèo và 100 con chó, nói "đây là mèo", "đây là chó" | Tự rút ra cảm nhận. Gặp mèo cụt tai vẫn nhận ra. |

- Lập trình truyền thống = **Cách 1**.
- Machine Learning = **Cách 2**.

Sơ đồ so sánh:

```
LẬP TRÌNH TRUYỀN THỐNG
   Dữ liệu ──┐
             ├──►  [ Máy tính ]  ──►  Kết quả
   Luật ─────┘      (con người viết luật)

MACHINE LEARNING
   Dữ liệu ──┐
             ├──►  [ Máy tính ]  ──►  Luật (gọi là "mô hình")
   Kết quả ──┘      (máy tự tìm luật)
```

Điểm mấu chốt: trong ML, **đầu ra của quá trình học chính là "luật"**. Sau đó ta dùng luật đó cho dữ liệu mới.

---

## 3. Khái niệm

### 3.1. Artificial Intelligence – AI (Trí tuệ nhân tạo)

- **Định nghĩa:** Mọi kỹ thuật giúp máy tính làm được việc mà ta thường nghĩ cần "trí thông minh" của con người.
- **Ví dụ đời thực:** Google Maps tìm đường nhanh nhất. Máy chơi cờ vua. Trợ lý ảo trả lời câu hỏi.
- **Lưu ý:** AI là cái ô rất to. Một chương trình cờ vua viết toàn bằng luật if-else cũng được gọi là AI, dù nó không "học" gì cả.

### 3.2. Machine Learning – ML (Học máy)

- **Định nghĩa:** Một nhánh của AI, nơi máy **tự rút ra quy luật từ dữ liệu** thay vì được lập trình từng luật.
- **Ví dụ đời thực:** Ngân hàng dự đoán khách có trả nợ đúng hạn không, dựa trên lịch sử của hàng nghìn khách cũ.

### 3.3. Deep Learning – DL (Học sâu)

- **Định nghĩa:** Một nhánh của ML, dùng **mạng nơ-ron nhiều lớp** để tự học các đặc điểm phức tạp từ dữ liệu thô như ảnh, âm thanh, văn bản (mạng nơ-ron sẽ học kỹ ở bài 3.1–3.4).
- **Ví dụ đời thực:** Camera nhận diện biển số xe ở bãi gửi xe. Face ID mở khóa điện thoại. ChatGPT.

### 3.4. Quan hệ giữa ba khái niệm

Hình dung như búp bê Nga lồng nhau:

```
┌───────────────────────────────────────────────┐
│ AI (Trí tuệ nhân tạo)                         │
│   ví dụ: luật if-else chơi cờ, tìm đường      │
│   ┌───────────────────────────────────────┐   │
│   │ Machine Learning (Học máy)            │   │
│   │   ví dụ: dự đoán giá nhà, lọc spam    │   │
│   │   ┌───────────────────────────────┐   │   │
│   │   │ Deep Learning (Học sâu)       │   │   │
│   │   │   ví dụ: nhận diện khuôn mặt, │   │   │
│   │   │   ChatGPT                     │   │   │
│   │   └───────────────────────────────┘   │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

**Mọi Deep Learning đều là Machine Learning. Mọi Machine Learning đều là AI. Nhưng chiều ngược lại thì không.**

### 3.5. Khi nào dùng ML cổ điển, khi nào dùng Deep Learning?

| Tiêu chí | ML cổ điển (Giai đoạn 2) | Deep Learning (Giai đoạn 3) |
|---|---|---|
| Loại dữ liệu hợp nhất | Bảng (Excel): giá nhà, thông tin khách hàng | Ảnh, âm thanh, văn bản dài |
| Lượng dữ liệu cần | Vài trăm đến vài chục nghìn dòng | Thường hàng chục nghìn trở lên |
| Phần cứng | Máy tính thường | Thường cần GPU (card đồ họa) |
| Giải thích kết quả | Dễ hơn | Khó hơn ("hộp đen") |
| Ví dụ | Chấm điểm tín dụng | Nhận diện biển số xe |

Mẹo nhớ: **dữ liệu dạng bảng → thử ML cổ điển trước**. Đừng dùng dao mổ trâu để giết gà.

---

## 4. Ba kiểu học của máy

### 4.1. Supervised Learning (Học có giám sát)

- **Định nghĩa:** Máy học từ dữ liệu **đã có đáp án** (gọi là **nhãn**, tiếng Anh là *label*).
- **Ví dụ:** 10.000 email, mỗi email đã được đánh dấu "SPAM" hoặc "Bình thường".
- Giống học sinh làm bài tập **có đáp án ở cuối sách**.
- Hai dạng chính (sẽ học kỹ ở bài 2.1):
  - **Regression (Hồi quy):** đoán một **con số**. Ví dụ: giá căn nhà là bao nhiêu tỷ.
  - **Classification (Phân loại):** đoán một **nhóm**. Ví dụ: email là SPAM hay không.

### 4.2. Unsupervised Learning (Học không giám sát)

- **Định nghĩa:** Máy tự tìm cấu trúc trong dữ liệu **không có đáp án**.
- **Ví dụ:** Siêu thị có dữ liệu mua hàng của 50.000 khách. Máy tự chia thành các nhóm: "nhóm hay mua đồ ăn vặt", "nhóm mua đồ cho em bé"... Không ai nói trước có những nhóm nào.
- Giống bạn đổ một thùng đồ chơi ra và tự xếp chúng thành từng đống giống nhau.

### 4.3. Reinforcement Learning (Học tăng cường)

- **Định nghĩa:** Máy học bằng cách **thử – sai – nhận thưởng/phạt**.
- **Ví dụ:** AI chơi game, robot tập đi, xe tự lái tập đỗ xe trong mô phỏng.
- Giống dạy chó: làm đúng thì cho bánh, làm sai thì không.
- Khóa học này **không đi sâu** vào kiểu học này, nhưng bạn cần biết nó tồn tại.

### Bảng tóm tắt

| Kiểu học | Có đáp án sẵn? | Câu hỏi máy trả lời | Ví dụ |
|---|---|---|---|
| Có giám sát | Có | "Cái này là gì / bao nhiêu?" | Giá nhà, lọc spam |
| Không giám sát | Không | "Dữ liệu này có những nhóm nào?" | Phân nhóm khách hàng |
| Tăng cường | Không, chỉ có thưởng/phạt | "Nên làm hành động nào?" | AI chơi game |

---

## 5. Bản đồ các lĩnh vực AI (và lộ trình của bạn)

```
                         ┌──────────────────────┐
                         │   NỀN TẢNG CHUNG      │
                         │ Python → ML → DL      │  ← Giai đoạn 0–3
                         └──────────┬───────────┘
                ┌───────────────────┼────────────────────┐
                ▼                   ▼                    ▼
   ┌──────────────────────┐ ┌───────────────────┐ ┌──────────────────┐
   │ Computer Vision (CV) │ │      NLP          │ │  Lĩnh vực khác   │
   │ Thị giác máy tính    │ │ Xử lý ngôn ngữ    │ │ (không học sâu)  │
   │ - Nhận diện biển số  │ │ - Chatbot CSKH    │ │ - Giọng nói      │
   │ - Đếm xe qua camera  │ │ - Phân tích bình  │ │ - Gợi ý sản phẩm │
   │ - Face ID            │ │   luận khách hàng │ │ - Robot          │
   └──────────┬───────────┘ └─────────┬─────────┘ └──────────────────┘
              └────────────┬──────────┘
                           ▼
                ┌──────────────────────┐
                │ TRIỂN KHAI SẢN PHẨM   │  ← Phần C
                │ API, Docker, Cloud    │
                └──────────────────────┘
```

- **Computer Vision – CV (Thị giác máy tính):** dạy máy "nhìn" và hiểu ảnh, video. Ví dụ: camera bãi xe đọc biển số.
- **Natural Language Processing – NLP (Xử lý ngôn ngữ tự nhiên):** dạy máy "đọc" và "viết" ngôn ngữ của người. Ví dụ: tự động phân loại bình luận "khen" hay "chê" trên Shopee.
- **Generative AI (AI tạo sinh):** AI tạo ra nội dung mới như chữ, ảnh, nhạc. ChatGPT thuộc nhóm này. Nó là giao điểm của Deep Learning và NLP (sẽ học ở nhánh NLP4).

---

## 6. Code: Luật thủ công vs Machine Learning

> Bạn **chưa cần hiểu từng dòng**. Mục tiêu là *thấy tận mắt* máy tự học được quy luật mà ta không hề viết ra. Bài 0.2 sẽ hướng dẫn kỹ cách dùng Colab.

**Cách chạy nhanh:**
1. Mở https://colab.research.google.com và đăng nhập Google.
2. Bấm **New notebook** (Sổ tay mới).
3. Dán code dưới đây vào ô đầu tiên, bấm nút ▶ (hoặc `Shift + Enter`).

```python
# ===== CÁCH 1: LẬP TRÌNH TRUYỀN THỐNG (con người tự viết luật) =====
def loc_spam_bang_luat(email):
    # Luật do lập trình viên nghĩ ra: có cụm "trúng thưởng" thì là spam
    if "trúng thưởng" in email.lower():
        return "SPAM"
    return "Bình thường"

# ===== CÁCH 2: MACHINE LEARNING (máy tự học luật từ ví dụ) =====
from sklearn.feature_extraction.text import CountVectorizer  # biến chữ thành số (sẽ học kỹ ở bài NLP1.3)
from sklearn.naive_bayes import MultinomialNB                # một thuật toán ML (sẽ học kỹ ở bài 2.8)

# Dữ liệu huấn luyện: mỗi email đã được con người dán nhãn sẵn
emails = [
    "Bạn đã trúng thưởng iPhone, nhận ngay hôm nay",
    "Quà tặng miễn phí, bấm link nhận ngay",
    "Khuyến mãi miễn phí 100%, nhận quà ngay",
    "Chúc mừng bạn được tặng voucher miễn phí",
    "Lịch họp dự án chuyển sang 3 giờ chiều",
    "Gửi anh báo cáo doanh thu tháng 10",
    "Tối nay cả nhà ăn cơm lúc 7 giờ nhé",
    "Nhắc lịch nộp báo cáo dự án thứ sáu",
]
nhan = ["SPAM", "SPAM", "SPAM", "SPAM",
        "Bình thường", "Bình thường", "Bình thường", "Bình thường"]

bo_dem_tu = CountVectorizer()            # công cụ đếm số lần mỗi từ xuất hiện
X = bo_dem_tu.fit_transform(emails)     # học bộ từ vựng + biến mỗi email thành dãy số
mo_hinh = MultinomialNB()               # tạo mô hình
mo_hinh.fit(X, nhan)                    # "cho máy học": tự tìm quy luật từ ví dụ

# ===== SO SÁNH TRÊN EMAIL MỚI CHƯA TỪNG THẤY =====
email_moi = [
    "Chúc mừng, bạn nhận quà miễn phí ngay",   # spam, nhưng KHÔNG chứa "trúng thưởng"
    "Gửi lịch họp báo cáo tuần sau",           # email công việc
]
for e in email_moi:
    du_doan_ml = mo_hinh.predict(bo_dem_tu.transform([e]))[0]  # máy dự đoán
    print(f"Email: {e}")
    print(f"  Luật thủ công : {loc_spam_bang_luat(e)}")
    print(f"  Machine Learning: {du_doan_ml}")
```

**Kết quả mẫu để đối chiếu:**

```
Email: Chúc mừng, bạn nhận quà miễn phí ngay
  Luật thủ công : Bình thường
  Machine Learning: SPAM
Email: Gửi lịch họp báo cáo tuần sau
  Luật thủ công : Bình thường
  Machine Learning: Bình thường
```

**Điều gì vừa xảy ra?**
- Luật thủ công **bỏ lọt** email spam đầu tiên, vì nó không chứa chữ "trúng thưởng".
- Mô hình ML **bắt được**. Ta không hề viết luật "miễn phí → spam". Máy **tự nhận ra** các từ như "miễn phí", "nhận", "quà", "ngay" hay xuất hiện trong email spam.
- Đây chính là anh Nam ở ngày 31.

> ⚠️ Lưu ý trung thực: 8 email là quá ít để tin tưởng. Thực tế cần hàng nghìn email. Bài 2.5 sẽ dạy cách đo xem mô hình tốt đến đâu.

---

## 7. Tự vẽ hình: AI ⊃ ML ⊃ DL

Dán vào một ô mới trong Colab và chạy:

```python
import matplotlib.pyplot as plt          # thư viện vẽ biểu đồ (sẽ học kỹ ở bài 1.5)
from matplotlib.patches import Circle    # công cụ vẽ hình tròn

fig, ax = plt.subplots(figsize=(6, 6))   # tạo khung vẽ 6x6 inch
vong = [
    # (bán kính, màu, tên, vị trí chữ theo chiều dọc)
    (3.0, "#cfe8ff", "AI (Trí tuệ nhân tạo)", 2.4),
    (2.0, "#9fd0ff", "Machine Learning\n(Học máy)", 1.3),
    (1.0, "#4fa3ff", "Deep Learning\n(Học sâu)", 0.0),
]
for ban_kinh, mau, ten, y_chu in vong:
    ax.add_patch(Circle((0, 0), ban_kinh, color=mau, ec="black"))           # vẽ vòng tròn
    ax.text(0, y_chu, ten, ha="center", va="center", fontsize=12, weight="bold")  # ghi tên

ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)  # giới hạn khung nhìn
ax.set_aspect("equal")                           # giữ hình tròn không bị méo
ax.axis("off")                                   # ẩn trục tọa độ
ax.set_title("AI ⊃ Machine Learning ⊃ Deep Learning")
plt.show()                                       # hiển thị hình
```

**Kết quả mong đợi:** ba vòng tròn lồng nhau. Vòng ngoài nhạt nhất là AI, vòng giữa là ML, vòng trong đậm nhất là DL.

---

## 8. Ứng dụng thực tế: bạn đã gặp AI hôm nay chưa?

| Ứng dụng | Thuộc loại | Kiểu học |
|---|---|---|
| Gmail tự chuyển thư rác vào mục Spam | ML / DL | Có giám sát (phân loại) |
| Shopee gợi ý "Có thể bạn cũng thích" | ML | Kết hợp nhiều kiểu |
| Ngân hàng duyệt khoản vay online trong 5 phút | ML cổ điển | Có giám sát (phân loại) |
| Camera bãi xe tự đọc biển số | DL – Computer Vision | Có giám sát |
| Chatbot chăm sóc khách hàng | DL – NLP | Có giám sát + nhiều kỹ thuật khác |
| Google Maps tìm đường | AI (thuật toán tìm đường) | Không phải ML |

---

## 9. Bài tập

**Bài 1 – Phân loại.** Mỗi ví dụ dưới đây là bài toán Hồi quy, Phân loại hay Phân nhóm (không giám sát)?

1. Dự đoán số lượng ly trà sữa bán được vào ngày mai.
2. Xác định một bình luận là "khen" hay "chê".
3. Chia 10.000 khách hàng thành các nhóm có thói quen mua sắm giống nhau.
4. Dự đoán điểm tín dụng (từ 300 đến 850) của một người.
5. Nhận diện ảnh có chứa con mèo hay không.

**Bài 2 – Thử nghiệm code.** Ở mục 6, thêm vào danh sách `email_moi` một email do bạn tự nghĩ. Thử tìm một email spam mà mô hình ML **bị lừa**. Vì sao nó bị lừa? (Gợi ý: nó chỉ học từ 8 email.)

**Bài 3 – Suy ngẫm.** Kể tên 1 việc trong công việc của bạn hiện tại có thể dùng ML. Dữ liệu cần có là gì? Đáp án (nhãn) là gì?

### Đáp án Bài 1

> Mẹo Notion: chọn tiêu đề "Đáp án Bài 1" và bấm `Turn into → Toggle heading 3` để giấu đáp án khi tự làm.

1. Hồi quy (đoán một con số).
2. Phân loại (2 nhóm: khen/chê).
3. Phân nhóm – học không giám sát (không ai nói trước có những nhóm nào).
4. Hồi quy (đoán một con số).
5. Phân loại (có/không), thường dùng Deep Learning vì dữ liệu là ảnh.

---

## 10. Flashcard ôn tập

| Mặt trước | Mặt sau |
|---|---|
| AI là gì? | Mọi kỹ thuật giúp máy làm việc cần "trí thông minh", kể cả luật if-else. |
| ML khác lập trình truyền thống ở đâu? | Lập trình truyền thống: người viết luật. ML: máy tự tìm luật từ dữ liệu + đáp án. |
| Deep Learning là gì? | Nhánh của ML dùng mạng nơ-ron nhiều lớp, mạnh với ảnh, âm thanh, văn bản. |
| Nhãn (label) là gì? | Đáp án đúng đi kèm mỗi ví dụ, như "SPAM" cho một email. |
| Hồi quy vs Phân loại? | Hồi quy đoán con số (giá nhà). Phân loại đoán nhóm (spam hay không). |
| Học không giám sát dùng khi nào? | Khi không có đáp án, muốn máy tự tìm nhóm/cấu trúc. |
| Dữ liệu dạng bảng nên thử gì trước? | ML cổ điển, trước khi nghĩ đến Deep Learning. |

---

## ✅ Tự kiểm tra trước khi sang bài 0.2

- [ ] Tôi giải thích được AI, ML, DL cho một người không học IT trong 1 phút.
- [ ] Tôi vẽ được sơ đồ "Dữ liệu + Luật → Kết quả" và "Dữ liệu + Kết quả → Luật".
- [ ] Tôi phân biệt được Hồi quy, Phân loại, Phân nhóm.
- [ ] Tôi đã chạy được code ở mục 6 và thấy kết quả giống mẫu.

**Bài tiếp theo:** 0.2 – Cài đặt môi trường và notebook đầu tiên.
