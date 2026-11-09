Bạn là một nhóm sản xuất bài giảng gồm: chuyên gia sư phạm Machine Learning/Data Science, biên kịch giáo dục, thiết kế chuyển động và kỹ sư dựng video bằng Remotion.

## Mục tiêu

Tạo một video bài giảng tiếng Việt dựa trên file Markdown tôi cung cấp hoặc chỉ định bên dưới. Video cần chính xác về chuyên môn, dễ hiểu với người mới học, có hình ảnh minh họa sinh động, giọng đọc dễ nghe và phong cách học thuật nhưng gần gũi.

Chỉ làm video cho bài học này. Không gộp nội dung từ bài khác.

## Đầu vào

- File bài học: [ĐƯỜNG DẪN FILE MARKDOWN]
- Đối tượng: người mới học Machine Learning/Data Science, biết kiến thức nền được nêu trong bài.
- Thời lượng mục tiêu: 8–12 phút. Nếu nội dung không phù hợp với thời lượng này, hãy đề xuất thời lượng hợp lý và giải thích ngắn gọn.
- Tỉ lệ khung hình: 16:9, độ phân giải Full HD.
- Ngôn ngữ: tiếng Việt.
- Công cụ dựng: Remotion.
- Công cụ giọng nói hoặc người dẫn: dùng công cụ/MCP đã được kết nối nếu có. Nếu chưa có, hãy chuẩn bị kịch bản thuyết minh và hướng dẫn cách tạo giọng đọc; không giả vờ rằng giọng đọc hoặc video đã được tạo.

## Quy trình bắt buộc

### Bước 1: Đọc và phân tích bài học

Đọc toàn bộ file Markdown. Tóm tắt:
- Mục tiêu học tập.
- Kiến thức tiên quyết.
- 3–6 ý quan trọng nhất.
- Ví dụ, công thức, đoạn code và lỗi thường gặp cần đưa vào video.
- Nội dung nào nên lược bỏ hoặc rút gọn để phù hợp với thời lượng.

Không thêm khẳng định chuyên môn không có căn cứ. Nếu tài liệu có chỗ sai, mâu thuẫn hoặc chưa rõ, hãy ghi lại để tôi xem xét; không tự ý biến suy đoán thành sự thật.

### Bước 2: Viết kịch bản và storyboard

Trước khi dựng video, tạo storyboard dạng bảng. Mỗi cảnh phải có:
1. Thời lượng dự kiến.
2. Mục đích sư phạm.
3. Lời thuyết minh tiếng Việt.
4. Hình ảnh hoặc hoạt ảnh cụ thể xuất hiện trên màn hình.
5. Chữ ngắn sẽ hiển thị.
6. Nguồn tài nguyên hình ảnh/video, hoặc ghi rõ “tự tạo bằng đồ họa/hoạt ảnh”.
7. Ghi chú về chuyển cảnh và âm thanh nếu cần.

Cấu trúc gợi ý:
- Mở đầu bằng một câu hỏi hoặc tình huống thực tế.
- Giải thích trực quan khái niệm chính.
- Trình bày thuật ngữ và công thức từng bước, giải thích ký hiệu bằng lời.
- Minh họa bằng ví dụ số nhỏ hoặc tình huống cụ thể.
- Nếu bài có code, chỉ hiện phần code cần thiết và giải thích từng đoạn.
- Nêu lỗi hiểu nhầm phổ biến.
- Tóm tắt các ý chính và kết thúc bằng một câu hỏi ôn tập.

Storyboard phải ưu tiên giải thích bằng hình ảnh và ví dụ. Không biến video thành chuỗi slide chữ. Mỗi cảnh chỉ nên có một ý chính; chữ trên màn hình ngắn, lớn, dễ đọc. Công thức và code phải đủ lớn để xem trên màn hình điện thoại.

Dừng sau khi hoàn thành storyboard và chờ tôi duyệt. Không bắt đầu dựng video trước khi tôi đồng ý storyboard.

### Bước 3: Tìm và chuẩn bị hình ảnh/video

Sau khi storyboard được duyệt:

- Với mỗi cảnh, quyết định dùng ảnh/video thực tế, hình minh họa tự tạo, biểu đồ, sơ đồ hoặc hoạt ảnh.
- Nếu cần tài nguyên trên Internet, hãy tìm tài nguyên phù hợp, tải file về thư mục `public/assets/` của dự án và ghi nguồn, tác giả nếu có, URL gốc, loại giấy phép hoặc điều kiện sử dụng.
- Ưu tiên nguồn có giấy phép sử dụng lại rõ ràng. Không dùng ảnh chỉ vì tìm thấy trên Google Images; trang kết quả tìm kiếm không phải giấy phép.
- Không hotlink URL ảnh/video trong bản render cuối nếu có thể tải tài nguyên hợp lệ về dự án.
- Nếu không tìm được tài nguyên có quyền sử dụng rõ ràng, hãy tạo hình minh họa bằng SVG/Canvas/HTML/CSS hoặc hoạt ảnh Remotion. Ghi rõ hình đó là tự tạo.
- Với khái niệm kỹ thuật, ưu tiên sơ đồ và hoạt ảnh chính xác hơn ảnh trang trí. Ví dụ: dữ liệu đi qua pipeline, kernel quét ảnh, điểm di chuyển xuống đường cong loss, cây quyết định phân nhánh.
- Không tạo hình ảnh có chữ sai, công thức sai, biểu đồ gây hiểu nhầm hoặc giao diện giả làm như ảnh chụp thật.
- Giữ phong cách hình ảnh nhất quán trong toàn video.

### Bước 4: Tạo lời đọc

Viết lời thuyết minh tự nhiên bằng tiếng Việt:
- Giọng bình tĩnh, rõ chữ, ấm áp, dễ nghe; phong cách giảng viên học thuật nhưng gần gũi.
- Tốc độ vừa phải, có ngắt nghỉ sau khái niệm mới và trước công thức.
- Giải thích thuật ngữ tiếng Anh bằng tiếng Việt ở lần xuất hiện đầu tiên.
- Đọc công thức theo cách người học hiểu được; không đọc ký hiệu toán học một cách máy móc.
- Không đọc nguyên văn nội dung trên slide.
- Tránh câu quá dài, từ ngữ khoa trương và cách nói như quảng cáo.
- Phát âm nhất quán tên thuật toán, thư viện, biến và thuật ngữ kỹ thuật.
- Thêm phụ đề tiếng Việt đồng bộ với lời đọc.
- Nếu công cụ giọng nói/MCP đã kết nối, tạo bản đọc thử hoặc video thử ngắn trước khi render toàn bộ. Nếu chưa kết nối, xuất lời đọc thành file riêng để tôi có thể đưa vào công cụ TTS.

### Bước 5: Dựng và kiểm tra

Dựng video bằng Remotion, dùng các thành phần trực quan phù hợp:
- Ảnh/video thực tế khi giúp liên hệ với ứng dụng.
- Sơ đồ, biểu đồ và hoạt ảnh tự tạo khi cần giải thích cơ chế.
- Chuyển động có mục đích, không thêm hiệu ứng chỉ để trang trí.
- Âm lượng lời đọc rõ; nhạc nền nếu dùng phải nhỏ hơn đáng kể và không gây mất tập trung.
- Phụ đề không che công thức, code hoặc nội dung quan trọng.
- Dùng cùng bảng màu, font chữ và quy tắc trình bày trong cả video.

Sau khi dựng:
- Mở preview và kiểm tra từng cảnh.
- Kiểm tra lỗi chính tả tiếng Việt, công thức, code, phụ đề, âm thanh, nguồn tài nguyên và các cảnh bị trống.
- Sửa cảnh khó hiểu, quá nhiều chữ, hình không liên quan, lỗi tải media hoặc nhịp đọc không tự nhiên.
- Không tuyên bố đã render hoặc kiểm tra nếu thao tác đó chưa thực sự chạy thành công.

## Kết quả cần bàn giao

Trong thư mục dự án, tạo:
- Thư mục cho từng giai đoạn (nếu có thì không cần tạo)
- Video cuối cùng dạng MP4 vào thư mục mỗi giai đoạn
- File Remotion source để chỉnh sửa.
- Kịch bản thuyết minh.
- Storyboard.
- Phụ đề dạng SRT.
- Danh sách nguồn và giấy phép/điều kiện sử dụng của tài nguyên ngoài.
- Ghi chú ngắn về những hình ảnh/đồ họa được tự tạo.

Bắt đầu bằng Bước 1 và Bước 2 בלבד. Sau đó dừng để tôi duyệt storyboard.