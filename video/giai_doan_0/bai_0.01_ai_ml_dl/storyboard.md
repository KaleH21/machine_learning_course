# Storyboard – Bài 0.1: AI, Machine Learning, Deep Learning khác nhau thế nào?

> Nguồn: `bai_hoc/giai_doan_0/bai_0.01_ai_ml_dl.md` · Trạng thái: **ĐÃ DUYỆT · ĐÃ DỰNG** (bai_0.01_ai_ml_dl.mp4, 8:39)
> Khung hình 1920×1080, 30 fps · Thời lượng dự kiến **≈ 11 phút 20 giây** (21 cảnh)

---

## Bước 1 – Phân tích bài học

### Mục tiêu học tập
1. Phân biệt AI, ML, DL bằng lời của chính mình.
2. Hiểu vì sao "cho máy học từ ví dụ" khác "viết luật if-else".
3. Biết 3 kiểu học (có giám sát, không giám sát, tăng cường) và các lĩnh vực lớn của AI (CV, NLP).
4. Thấy tận mắt một đoạn code ML đầu tiên tự học được quy luật.

### Kiến thức tiên quyết
Không cần kiến thức ML. Chỉ cần biết máy tính chạy theo chương trình; đọc hiểu `if-else` ở mức khái niệm là đủ.

### 5 ý quan trọng nhất
1. **Đảo chiều:** Lập trình truyền thống: *Dữ liệu + Luật → Kết quả*. ML: *Dữ liệu + Kết quả → Luật (mô hình)*.
2. **Lồng nhau:** AI ⊃ ML ⊃ DL. Mọi DL là ML, mọi ML là AI, chiều ngược lại thì không (cờ vua viết bằng if-else vẫn là AI).
3. **Chọn công cụ:** Dữ liệu dạng bảng → thử ML cổ điển trước. DL hợp với ảnh, âm thanh, văn bản, cần nhiều dữ liệu và thường cần GPU.
4. **Ba kiểu học:** Có giám sát (có nhãn; gồm hồi quy → con số, phân loại → nhóm), không giám sát (tự tìm nhóm), tăng cường (thử – sai – thưởng/phạt).
5. **Bằng chứng:** Mô hình ML bắt được email spam không chứa "trúng thưởng", điều mà luật thủ công bỏ lọt.

### Ví dụ, code, lỗi hiểu nhầm đưa vào video
- **Ví dụ xuyên suốt:** câu chuyện anh Nam lọc spam (ngày 1 → 3 → 30 → 31); dạy trẻ nhận ra con mèo.
- **Ví dụ đời thực:** Google Maps, cờ vua (AI); ngân hàng duyệt vay (ML); Face ID, biển số xe, ChatGPT (DL); phân nhóm khách siêu thị; AI chơi game.
- **Code:** mục 6, rút gọn còn 4 khung: hàm luật → dữ liệu + nhãn → `fit` → `predict` + kết quả. **Đã chạy thử (scikit-learn): kết quả khớp đúng với kết quả mẫu trong bài.**
- **Hiểu nhầm phổ biến** (đều có căn cứ trong bài):
  - "AI thì phải biết học" → sai, AI dùng luật if-else vẫn là AI.
  - "Deep Learning lúc nào cũng tốt hơn" → sai, dữ liệu bảng nên thử ML cổ điển trước.
  - "Mô hình đúng trên 2 email là tin được" → sai, 8 email là quá ít, thực tế cần hàng nghìn.

### Lược bỏ / rút gọn
- **Bỏ:** hướng dẫn mở Colab từng bước (bài 0.2 dạy kỹ), code matplotlib mục 7 (thay bằng hoạt ảnh vòng tròn), toàn bộ bảng ứng dụng mục 8 (chỉ lấy 3–4 ví dụ cho phần mở đầu), bài tập 2–3, flashcard, checklist.
- **Rút gọn:** bảng ML cổ điển vs DL còn 4 tiêu chí; bản đồ lĩnh vực chỉ nhấn CV, NLP và chặng triển khai.
- **Giữ 1 bài tập** (trà sữa → hồi quy) làm câu hỏi ôn tập cuối video.

### ⚠️ Điểm cần anh/chị xem lại trong tài liệu (mình KHÔNG tự sửa file bài học)
| # | Vị trí | Vấn đề | Đề xuất |
|---|---|---|---|
| 1 | Mục 3.4, dòng 89–99 | Có 10 dòng trống thừa trong khối sơ đồ búp bê Nga. | Xóa các dòng trống. |
| 2 | Mục 3.5, bảng | "Loại dữ liệu **hợp nhất**", có thể là gõ thiếu chữ. | "Loại dữ liệu **phù hợp nhất**". |
| 3 | Mục 5, sơ đồ | "Lĩnh vực khác **(không học sâu)**" dễ bị hiểu là "không phải Deep Learning (Học sâu)", trong khi giọng nói và gợi ý sản phẩm đều có thể dùng DL. | "(khóa học không đi sâu)". |
| 4 | Mục 5, Generative AI | "Giao điểm của Deep Learning và NLP" chưa chính xác, vì chính câu trước nói AI tạo sinh tạo cả ảnh và nhạc (không thuộc NLP). | "AI tạo sinh thường dựa trên Deep Learning. ChatGPT là AI tạo sinh cho văn bản, nên nằm ở chỗ DL gặp NLP." Video sẽ nói theo cách này. |
| 5 | Mục 4.2 | Ví dụ đặt tên nhóm ("nhóm mua đồ ăn vặt") có thể khiến người học nghĩ máy tự đặt tên. | Video sẽ cho thấy: máy chỉ gom nhóm, **con người** xem rồi đặt tên. |

---

## Quy ước trình bày (áp dụng cả video)

- **Phong cách:** đồ họa phẳng, nền tối dịu `#0F172A`, khối bo góc, nét đơn giản. Hầu hết hình **tự tạo bằng SVG/hoạt ảnh Remotion**. Đây là bài khái niệm, sơ đồ chính xác hơn ảnh chụp.
- **Màu ngữ nghĩa (cố định):** AI = xanh nhạt `#93C5FD` · ML = xanh vừa `#3B82F6` · DL = xanh đậm `#1D4ED8` · SPAM = đỏ cam `#F97316` · Bình thường = xanh lá `#22C55E` · Luật do người viết = vàng `#FACC15`.
- **Font:** *Be Vietnam Pro* (Google Fonts, giấy phép OFL, hỗ trợ đủ dấu tiếng Việt), code dùng *JetBrains Mono* (OFL).
- **Cỡ chữ:** tiêu đề ≥ 72px, chữ trên màn hình ≥ 48px, code ≥ 40px (tối đa ~8 dòng/khung) để xem được trên điện thoại.
- **Phụ đề:** dải dưới cùng, cao 140px. Mọi công thức, code và sơ đồ chỉ nằm trong vùng phía trên dải này.
- **Âm thanh:** giọng đọc là chính. Nhạc nền (nếu dùng) ở mức −24 dB so với giọng, chỉ bật ở mở đầu và kết thúc. Một tiếng "ting" nhẹ khi đáp án/kết luận xuất hiện.
- **Chuyển cảnh:** mặc định cắt hoặc mờ dần 10 frame. Dùng "phóng to vào sơ đồ" khi chuyển từ tổng quát sang chi tiết.

---

## Bước 2 – Storyboard

> Thời lượng là ước tính theo tốc độ đọc ~140 từ/phút. Thời lượng thật sẽ khớp theo file giọng đọc.

### PHẦN A – Mở đầu bằng câu chuyện (≈ 1:45)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 1 | 0:20 | Gợi tò mò, liên hệ đời sống | Sáng nay, có thể bạn đã dùng AI vài lần mà không để ý. Điện thoại mở khóa bằng khuôn mặt. Hộp thư tự gom thư rác. Bản đồ chỉ đường tránh kẹt xe. Vậy AI, Machine Learning và Deep Learning khác nhau thế nào? Mười phút tới, ta sẽ trả lời câu hỏi đó. | 3 biểu tượng lần lượt bật lên: khuôn mặt + ổ khóa, hộp thư có mục Spam, bản đồ có đường đi. Sau đó ba từ AI / ML / DL bay vào giữa kèm dấu "?". | **AI · ML · DL – khác nhau thế nào?** | Tự tạo (biểu tượng SVG) | Nhạc nền nhẹ bắt đầu, nhỏ dần khi vào cảnh 2 |
| 2 | 0:45 | Cho thấy cách "viết luật" hoạt động, rồi bị lách | Đây là anh Nam, lập trình viên ở một công ty thương mại điện tử. Sếp giao việc: lọc email lừa đảo. Ngày đầu tiên, anh viết một luật: nếu email chứa "trúng thưởng" thì là spam. Chạy tốt. Nhưng đến ngày thứ ba, kẻ lừa đảo đổi cách viết: "trúng thuởng" sai dấu, chữ bị chấm ngắt quãng, hoặc đổi hẳn sang "nhận quà miễn phí". Luật cũ không bắt được nữa. | Nhân vật anh Nam tối giản trước màn hình. Băng chuyền email chạy qua một "cổng luật" màu vàng. Email có "trúng thưởng" → bị chặn, chuyển màu đỏ cam. Rồi 3 email biến thể đi **lọt** qua cổng, được đánh dấu ❌ đỏ. | **Ngày 1:** `"trúng thưởng" → SPAM` ✅<br>**Ngày 3:** `trúng thuởng` · `tr.úng th.ưởng` · `nhận quà miễn phí` → lọt ❌ | Tự tạo | Tiếng "ting" khi chặn đúng, tiếng "bụp" nhẹ khi lọt |
| 3 | 0:40 | Cho thấy giới hạn của việc viết luật; giới thiệu ML | Anh Nam thêm luật mới. Ngày thứ ba mươi, anh đã có năm trăm luật. Các luật bắt đầu chồng chéo nhau, và một email của chính sếp có chữ "thưởng Tết" cũng bị chặn. Rồi một đồng nghiệp gợi ý: sao không đưa cho máy mười nghìn email đã được đánh dấu spam hay bình thường, rồi để máy tự tìm ra quy luật? Ý tưởng đó có tên là Machine Learning, học máy. | Cổng luật phình ra thành một bức tường dày các thẻ luật vàng, con số đếm tăng lên 500. Email "Thưởng Tết – từ Sếp" bị chặn nhầm, nhấp nháy. Chuyển sang: một chồng email có nhãn đỏ/xanh đổ vào một hộp "Máy tính" và hộp phát sáng. | **Ngày 30:** 500 luật 😩<br>**Ngày 31:** 10.000 email có nhãn → máy tự tìm luật<br>**Machine Learning (Học máy)** | Tự tạo | Nhịp nhanh dần khi đếm luật; ngừng lại 0,5 giây trước chữ "Machine Learning" |

### PHẦN B – Trực giác (≈ 1:25)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 4 | 0:40 | So sánh "đưa luật" với "đưa ví dụ" bằng tình huống ai cũng hiểu | Hãy nghĩ đến việc dạy một đứa trẻ nhận ra con mèo. Cách thứ nhất: đưa luật, mèo có bốn chân, có ria, tai nhọn, kêu meo meo. Đứa trẻ thuộc luật, nhưng gặp một con mèo bị cụt tai thì bối rối. Cách thứ hai: cho bé xem thật nhiều ảnh mèo và chó, nói đây là mèo, đây là chó. Bé tự rút ra cảm nhận, và gặp mèo cụt tai vẫn nhận ra. Lập trình truyền thống giống cách một. Machine Learning giống cách hai. | Màn hình chia đôi. **Trái:** danh sách luật được đánh dấu ✓ lần lượt; một con mèo cụt tai xuất hiện → dấu "?" trên đầu đứa trẻ. **Phải:** lưới nhiều ô mèo/chó (hình vẽ phẳng) có nhãn; con mèo cụt tai xuất hiện → "Mèo!" ✓. | **Cách 1: Đưa luật** = Lập trình truyền thống<br>**Cách 2: Đưa ví dụ** = Machine Learning | Tự tạo (hình mèo/chó vẽ bằng SVG) | Cắt cứng giữa hai nửa để nhấn sự đối lập |
| 5 | 0:45 | **Ý mấu chốt:** đảo chiều đầu vào/đầu ra | Đây là điểm quan trọng nhất của bài. Trong lập trình truyền thống, ta đưa vào dữ liệu và luật, máy cho ra kết quả. Trong Machine Learning, ta đưa vào dữ liệu và kết quả đúng, còn máy tự tìm ra luật. Luật do máy học được ấy có tên riêng: mô hình. Sau khi có mô hình, ta dùng nó cho dữ liệu mới. | Sơ đồ 1: hai khối "Dữ liệu", "Luật" có mũi tên vào hộp "Máy tính" → "Kết quả". Sau đó hoạt ảnh **đổi chỗ** hai khối "Luật" và "Kết quả" (cong vòng sang vị trí của nhau) để thành Sơ đồ 2. Khối "Luật" đổi tên thành "Mô hình". Cuối cảnh: một email mới đi vào "Mô hình" → nhãn SPAM. | **Dữ liệu + Luật → Kết quả**<br>**Dữ liệu + Kết quả → Luật (mô hình)** | Tự tạo | Dừng 1 giây khi hai khối đổi chỗ xong; tiếng "ting" |

### PHẦN C – Khái niệm AI, ML, DL (≈ 2:35)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 6 | 0:30 | Định nghĩa AI là "cái ô" lớn | Bắt đầu từ khái niệm rộng nhất: AI, Artificial Intelligence, trí tuệ nhân tạo. Đó là mọi kỹ thuật giúp máy tính làm được việc mà ta thường nghĩ cần trí thông minh của con người. Tìm đường nhanh nhất, chơi cờ vua, trả lời câu hỏi. Lưu ý: một chương trình cờ vua viết toàn bằng luật if-else vẫn được gọi là AI, dù nó không học gì cả. | Một chiếc ô lớn màu `#93C5FD` mở ra. Dưới ô: biểu tượng bản đồ, bàn cờ, bong bóng chat. Bàn cờ có nhãn nhỏ `if-else` màu vàng. | **AI – Trí tuệ nhân tạo**<br>Không nhất thiết phải "học" | Tự tạo | Mờ dần sang cảnh 7 |
| 7 | 0:25 | Định nghĩa ML | Bên trong AI có một nhánh: Machine Learning, học máy. Ở đây, máy tự rút ra quy luật từ dữ liệu thay vì được lập trình từng luật. Ví dụ, ngân hàng dự đoán một khách hàng có trả nợ đúng hạn không, dựa trên lịch sử của hàng nghìn khách cũ. | Một bảng dữ liệu nhỏ (thu nhập, số khoản vay, đã trả đúng hạn ✓/✗) cuộn vào hộp "Mô hình". Một dòng khách mới không có đáp án đi qua → "Dự đoán: trả đúng hạn". | **Machine Learning – Học máy**<br>Tự rút quy luật từ dữ liệu | Tự tạo (dữ liệu minh họa, ghi chú "số liệu minh họa") | — |
| 8 | 0:30 | Định nghĩa DL | Bên trong Machine Learning lại có một nhánh nhỏ hơn: Deep Learning, học sâu. Deep Learning dùng mạng nơ-ron nhiều lớp để tự học những đặc điểm phức tạp từ dữ liệu thô như ảnh, âm thanh, văn bản. Face ID, camera đọc biển số xe và ChatGPT đều dùng Deep Learning. Mạng nơ-ron sẽ được học kỹ ở giai đoạn 3. | Ảnh con mèo dạng lưới điểm ảnh đi vào một mạng gồm 4 lớp hình tròn nối nhau, các kết nối sáng lần lượt từ trái sang phải → đầu ra "Mèo". Sau đó 3 biểu tượng nhỏ: khuôn mặt, biển số, bong bóng chat. | **Deep Learning – Học sâu**<br>Mạng nơ-ron nhiều lớp · ảnh, âm thanh, văn bản | Tự tạo | Tín hiệu chạy qua mạng có tiếng "vút" rất nhẹ |
| 9 | 0:30 | Tổng hợp quan hệ lồng nhau | Ghép lại, ta có ba vòng tròn lồng nhau như búp bê Nga. AI ở ngoài cùng, Machine Learning ở giữa, Deep Learning ở trong cùng. Nên nhớ một câu: mọi Deep Learning đều là Machine Learning, mọi Machine Learning đều là AI, nhưng chiều ngược lại thì không. | Ba vòng tròn đồng tâm (3 sắc xanh) phóng to dần từ trong ra ngoài, mỗi vòng có nhãn và 1 ví dụ: AI: cờ if-else; ML: lọc spam, giá nhà; DL: nhận diện khuôn mặt, ChatGPT. Mũi tên "⊃" nhấp nháy giữa các vòng. | **AI ⊃ ML ⊃ DL**<br>Chiều ngược lại: không | Tự tạo (thay cho code matplotlib mục 7) | Tiếng "ting" khi câu kết luận hiện |
| 10 | 0:40 | Biết khi nào chọn ML cổ điển, khi nào chọn DL | Vậy khi nào dùng Machine Learning cổ điển, khi nào dùng Deep Learning? Với dữ liệu dạng bảng như giá nhà hay thông tin khách hàng, chỉ cần vài trăm đến vài chục nghìn dòng, máy tính thường là đủ, và kết quả dễ giải thích. Với ảnh, âm thanh, văn bản dài, Deep Learning thường mạnh hơn, nhưng cần nhiều dữ liệu hơn, thường cần card đồ họa, và khó giải thích hơn, giống một hộp đen. Mẹo nhớ: dữ liệu dạng bảng thì thử ML cổ điển trước. | Hai cột: trái là bảng Excel thu nhỏ, phải là ảnh, sóng âm, đoạn văn. 4 dòng so sánh hiện lần lượt: *Dữ liệu · Lượng dữ liệu · Phần cứng (laptop vs GPU) · Giải thích được (kính lúp vs hộp đen)*. | **Dữ liệu bảng → thử ML cổ điển trước** | Tự tạo | Mỗi dòng hiện theo nhịp lời đọc |

### PHẦN D – Ba kiểu học (≈ 2:05)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 11 | 0:50 | Học có giám sát; phân biệt hồi quy và phân loại | Máy có ba kiểu học chính. Kiểu đầu tiên là Supervised Learning, học có giám sát. Máy học từ dữ liệu đã có đáp án, đáp án này gọi là nhãn, tiếng Anh là label. Giống học sinh làm bài tập có đáp án ở cuối sách. Học có giám sát có hai dạng. Hồi quy, regression, đoán một con số, ví dụ căn nhà giá bao nhiêu tỷ. Phân loại, classification, đoán một nhóm, ví dụ email này là spam hay không. | Các email có thẻ nhãn đỏ/xanh dán vào. Rồi màn hình tách hai nhánh: **Hồi quy:** một căn nhà → kim chạy trên trục số → "3,2 tỷ" (số liệu minh họa). **Phân loại:** một email rơi vào một trong hai giỏ SPAM / Bình thường. | **Có giám sát:** có nhãn (label)<br>**Hồi quy** → con số · **Phân loại** → nhóm | Tự tạo | Chia đôi màn hình khi sang hai dạng |
| 12 | 0:30 | Học không giám sát | Kiểu thứ hai là Unsupervised Learning, học không giám sát. Lần này dữ liệu không có đáp án. Ví dụ, siêu thị có dữ liệu mua hàng của năm mươi nghìn khách. Máy tự gom những khách có thói quen giống nhau thành từng nhóm. Không ai nói trước có những nhóm nào. Sau đó con người mới nhìn vào và đặt tên, như "nhóm hay mua đồ ăn vặt". | Một đám chấm tròn cùng màu xám rải ngẫu nhiên → các chấm từ từ di chuyển, tụ thành 3 cụm và đổi màu. Sau đó nhãn tên cụm hiện lên kèm biểu tượng bàn tay người viết. | **Không giám sát:** không có nhãn → máy tự gom nhóm | Tự tạo | — |
| 13 | 0:25 | Học tăng cường (giới thiệu) | Kiểu thứ ba là Reinforcement Learning, học tăng cường. Máy học bằng cách thử, sai, rồi nhận thưởng hoặc phạt, giống dạy chó: làm đúng thì được bánh. AI chơi game và robot tập đi dùng cách này. Khóa học không đi sâu vào kiểu học này, nhưng bạn cần biết nó tồn tại. | Một nhân vật nhỏ trong mê cung lưới 5×5: đi sai vào ô đỏ → "−1", đi đúng tới ô cờ → "+10" ⭐. Lặp 3 lượt, đường đi ngắn dần. | **Tăng cường:** thử – sai – thưởng/phạt | Tự tạo | Tiếng thưởng nhỏ khi tới đích |
| 14 | 0:20 | Ôn nhanh 3 kiểu học | Tóm lại: có giám sát thì có đáp án và trả lời "cái này là gì, bao nhiêu". Không giám sát thì không có đáp án và trả lời "dữ liệu có những nhóm nào". Tăng cường chỉ có thưởng phạt và trả lời "nên làm hành động nào". | Bảng 3 hàng hiện lần lượt, mỗi hàng có biểu tượng thu nhỏ từ cảnh 11–13. | Có giám sát · Không giám sát · Tăng cường | Tự tạo | — |

### PHẦN E – Bản đồ lĩnh vực (≈ 0:35)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 15 | 0:35 | Định hướng lộ trình học | Đây là bản đồ khóa học của bạn. Nền tảng chung là Python, rồi Machine Learning, rồi Deep Learning. Từ đó tách ra các lĩnh vực. Computer Vision, thị giác máy tính, dạy máy nhìn và hiểu ảnh, như đọc biển số xe. NLP, xử lý ngôn ngữ tự nhiên, dạy máy đọc và viết ngôn ngữ của người, như phân loại bình luận khen hay chê. ChatGPT là AI tạo sinh cho văn bản, nằm ở chỗ Deep Learning gặp NLP. Chặng cuối là đưa mô hình thành sản phẩm. | Bản đồ dạng lộ trình: khối "Nền tảng: Python → ML → DL" ở trên, tách 3 nhánh **CV · NLP · Lĩnh vực khác**, hai nhánh CV và NLP hội tụ về "Triển khai sản phẩm: API, Docker, Cloud". Một chấm "Bạn ở đây" nhấp nháy ở giai đoạn 0. | **Python → ML → DL → CV / NLP → Triển khai** | Tự tạo (theo sơ đồ mục 5, đã sửa nhãn "khóa học không đi sâu") | Lia camera từ trên xuống theo lộ trình |

### PHẦN F – Code: luật thủ công vs ML (≈ 1:25)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 16 | 0:40 | Thấy hai cách trong code thật (chưa cần hiểu hết) | Giờ ta xem lại câu chuyện anh Nam bằng code Python. Bạn chưa cần hiểu từng dòng. Cách một là luật thủ công: nếu email có chữ "trúng thưởng" thì trả về spam. Cách hai là Machine Learning. Ta chuẩn bị tám email, bốn email spam và bốn email bình thường, mỗi email có nhãn đúng do con người dán. | **Khung 1:** hàm `loc_spam_bang_luat` (5 dòng), dòng `if` được tô vàng. **Khung 2:** danh sách `emails` rút gọn còn 4 dòng hiển thị + "…", danh sách `nhan`; nhãn SPAM tô đỏ cam, Bình thường tô xanh lá. | `def loc_spam_bang_luat(email): …`<br>`emails = [...]` · `nhan = [...]` | Code từ mục 6 của bài | Gõ chữ hiện dần (typewriter) chậm |
| 17 | 0:45 | Thấy máy tự học được luật không ai viết | Ba dòng tiếp theo là phần "cho máy học". Bộ đếm từ biến mỗi email thành dãy số. Hàm fit nghĩa là cho máy học, tự tìm quy luật từ ví dụ. Giờ ta thử một email spam mới, không hề có chữ "trúng thưởng". Luật thủ công cho là bình thường, bị lọt. Còn mô hình Machine Learning bắt được, là spam. Ta không hề viết luật "miễn phí là spam". Máy tự nhận ra các từ như "miễn phí", "quà", "ngay" hay xuất hiện trong thư rác. | **Khung 3:** 4 dòng `CountVectorizer`, `fit_transform`, `MultinomialNB()`, `fit` (dòng `fit` phát sáng). **Khung 4:** hai cột kết quả cho email "Chúc mừng, bạn nhận quà miễn phí ngay": *Luật thủ công: Bình thường ❌* · *ML: SPAM ✅*. Các từ "miễn phí", "quà", "ngay" trong email được gạch chân. | `mo_hinh.fit(X, nhan)` ← cho máy học<br>Luật: **Bình thường ❌** · ML: **SPAM ✅** | Code + kết quả mục 6 (đã chạy thử, khớp kết quả mẫu) | Tiếng "ting" khi ML ra SPAM |

### PHẦN G – Hiểu nhầm, tóm tắt, ôn tập (≈ 1:30)

| # | Thời lượng | Mục đích sư phạm | Lời thuyết minh | Hình ảnh / hoạt ảnh | Chữ trên màn hình | Nguồn | Chuyển cảnh / âm thanh |
|---|---|---|---|---|---|---|---|
| 18 | 0:35 | Sửa 3 hiểu nhầm phổ biến | Có ba hiểu nhầm hay gặp. Một: AI thì phải biết học. Sai, chương trình cờ vua dùng if-else vẫn là AI. Hai: Deep Learning lúc nào cũng tốt hơn. Sai, với dữ liệu dạng bảng, hãy thử ML cổ điển trước. Ba: mô hình vừa rồi đoán đúng nên đã tin được. Cũng sai, tám email là quá ít, thực tế cần hàng nghìn email, và bài 2.5 sẽ dạy cách đo độ tốt của mô hình. | 3 thẻ lần lượt lật: mặt trước là câu hiểu nhầm có dấu ❌, mặt sau là câu đúng có dấu ✓ kèm biểu tượng (bàn cờ, bảng Excel, chồng email nhỏ xíu). | ❌ AI phải biết học<br>❌ DL luôn tốt hơn<br>❌ 8 email là đủ | Tự tạo | Hiệu ứng lật thẻ |
| 19 | 0:25 | Tóm tắt ý chính | Tóm lại bài hôm nay. Machine Learning đảo chiều cách lập trình: đưa dữ liệu và đáp án, máy tự tìm luật. AI chứa ML, ML chứa DL. Máy có ba kiểu học: có giám sát, không giám sát và tăng cường. Và dữ liệu dạng bảng thì thử ML cổ điển trước. | 4 biểu tượng thu nhỏ từ cảnh 5, 9, 14, 10 xếp thành lưới 2×2, mỗi ô sáng lên theo lời đọc. | 4 ý chính (mỗi ý ≤ 6 từ) | Tự tạo | — |
| 20 | 0:20 | Kiểm tra hiểu bài | Một câu hỏi cho bạn: dự đoán số ly trà sữa bán được vào ngày mai là bài toán hồi quy hay phân loại? Hãy dừng lại vài giây và nghĩ thử. … Đáp án là hồi quy, vì ta đoán một con số. | Ly trà sữa và lịch "Ngày mai". Đồng hồ đếm ngược 5 giây. Sau đó hiện đáp án "Hồi quy – đoán một con số" ✓. | **Hồi quy hay Phân loại?** → **Hồi quy** | Tự tạo (lấy từ Bài tập 1, câu 1) | Im lặng trong 5 giây đếm ngược; tiếng "ting" khi có đáp án |
| 21 | 0:10 | Kết bài, dẫn sang bài sau | Bài tiếp theo, ta sẽ cài đặt môi trường và chạy notebook đầu tiên. Hẹn gặp lại bạn. | Thẻ kết thúc: "Bài tiếp theo: 0.2 – Cài đặt môi trường và notebook đầu tiên". | **Tiếp theo: Bài 0.2** | Tự tạo | Nhạc nền nhỏ dần đến hết |

**Tổng ước tính:** ≈ 11:20, nằm trong khoảng 8–12 phút. Nếu bản đọc thật dài hơn 12 phút, cảnh 14 (bảng ôn 3 kiểu học) và cảnh 19 có thể gộp lại để tiết kiệm khoảng 25 giây.

---

## Tài nguyên ngoài dự kiến
- **Font:** Be Vietnam Pro và JetBrains Mono (cả hai giấy phép SIL Open Font License, sẽ tải qua `@remotion/google-fonts`).
- **Ảnh/video Internet:** dự kiến **không cần**. Mọi hình là SVG/hoạt ảnh tự tạo. Nếu anh/chị muốn thêm cảnh quay thật (ví dụ camera bãi xe), mình sẽ tìm nguồn có giấy phép rõ ràng (Pexels, Wikimedia Commons) và ghi nguồn đầy đủ.
- **Âm thanh hiệu ứng / nhạc nền:** chọn sau khi duyệt, chỉ dùng nguồn có giấy phép rõ ràng (ví dụ Pixabay Music), hoặc bỏ nhạc nền.

## Cần anh/chị quyết định trước khi dựng
1. **Giọng đọc:** hiện **chưa có công cụ TTS/MCP nào được kết nối**. Các lựa chọn:
   - (a) Mình xuất kịch bản thuyết minh + file phụ đề SRT, anh/chị tự tạo giọng (ElevenLabs, FPT.AI, Viettel AI, Google TTS…) rồi gửi file âm thanh lại.
   - (b) Mình dùng `edge-tts` (giọng `vi-VN-HoaiMyNeural` hoặc `vi-VN-NamMinhNeural`, miễn phí, chạy cục bộ) để tạo bản đọc thử. Lưu ý điều khoản sử dụng thương mại của giọng Microsoft chưa rõ ràng.
   - (c) Kết nối một MCP giọng nói (ví dụ ElevenLabs) rồi mình làm tiếp.
2. **Nhạc nền:** có hay không?
3. **5 điểm cần xem lại** trong tài liệu ở trên: anh/chị có muốn sửa file bài học không, hay chỉ áp dụng cách diễn đạt đã sửa trong video?
