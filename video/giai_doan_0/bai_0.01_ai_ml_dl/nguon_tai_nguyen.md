# Nguồn tài nguyên và giấy phép – Bài 0.01

## Tài nguyên bên ngoài

| Tài nguyên | Tác giả / nhà cung cấp | Nguồn | Giấy phép / điều kiện |
|---|---|---|---|
| Font **Be Vietnam Pro** | Lâm Bảo, Tony Le, ViệtAnh Nguyễn | Google Fonts, tải qua `@remotion/google-fonts` | SIL Open Font License 1.1 |
| Font **JetBrains Mono** | JetBrains | Google Fonts, tải qua `@remotion/google-fonts` | SIL Open Font License 1.1 |
| Giọng đọc (TTS) | Microsoft Edge neural TTS, giọng `vi-VN-HoaiMyNeural`, tốc độ −5% | Tạo bằng thư viện mã nguồn mở `edge-tts` 7.3.1 (`video/remotion/scripts/tts_edge.py`) | `edge-tts` (giấy phép LGPL/GPL) chỉ là công cụ gọi dịch vụ đọc văn bản của trình duyệt Edge; Microsoft **không công bố điều khoản rõ ràng cho việc dùng thương mại** giọng đọc tạo theo cách này. Phù hợp cho học tập/nội bộ; nếu phát hành thương mại nên thay bằng dịch vụ có giấy phép rõ (Azure TTS, ElevenLabs trả phí, FPT.AI…). |
| Remotion | Remotion AG | npm `remotion@4.0.541` | Miễn phí cho cá nhân / nhóm ≤ 3 người; công ty lớn hơn cần giấy phép công ty (remotion.pro/license) |

## Không dùng
- **Ảnh/video Internet:** không dùng. Mọi hình ảnh là đồ họa tự tạo.
- **Nhạc nền và hiệu ứng âm thanh ("ting", "bụp"):** chưa dùng. Storyboard để ngỏ quyết định nhạc nền, và thư viện hiệu ứng mẫu của Remotion không ghi rõ giấy phép, nên mình bỏ qua thay vì dùng file không rõ quyền.

## Đồ họa tự tạo
Tất cả hình trong video được vẽ bằng SVG/CSS và hoạt ảnh Remotion trong `video/remotion/src/bai_0_01/` (`ui.tsx`, `scenes.tsx`, `code.tsx`):
- Biểu tượng: khuôn mặt + ổ khóa, hộp thư, bản đồ, bàn cờ, bong bóng chat, nhà, biển số, laptop, GPU, bảng, ảnh, sóng âm, văn bản, kính lúp, hộp đen, người, trẻ em, bàn tay, ly trà sữa, lịch, mèo, chó.
- Sơ đồ: cổng luật lọc email, tường luật, sơ đồ đảo chiều "Dữ liệu + Luật → Kết quả", ô AI, mạng nơ-ron 4 lớp, ba vòng tròn lồng nhau, bảng so sánh ML cổ điển / DL, trục số hồi quy, giỏ phân loại, gom cụm điểm, mê cung 5×5 học tăng cường, bản đồ khóa học, thẻ lật hiểu nhầm.
- **Số liệu minh họa (không phải dữ liệu thật):** bảng khách vay ở cảnh 7, giá nhà "3,2 tỷ" ở cảnh 11, biển số "30A-123.45" ở cảnh 8, vị trí các điểm ở cảnh 12 (sinh ngẫu nhiên có seed).
- Code ở cảnh 16–17 lấy từ mục 6 của bài học (rút gọn danh sách email bằng `...`).
