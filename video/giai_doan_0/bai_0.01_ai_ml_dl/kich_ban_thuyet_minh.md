# Kịch bản thuyết minh – Bài 0.01: AI, Machine Learning, Deep Learning

> Giọng: edge-tts `vi-VN-HoaiMyNeural` (−5%). Nguồn dữ liệu: `video/remotion/src/bai_0_01/script.json`; các thẻ như [pause], [warmly] dành cho ElevenLabs và bị bỏ qua khi dùng edge-tts. Cảnh 20 có 5 giây im lặng giữa câu hỏi và đáp án.

## Cảnh 1

Sáng nay, có thể bạn đã dùng AI vài lần mà không để ý. Điện thoại mở khóa bằng khuôn mặt. Hộp thư tự gom thư rác. Bản đồ chỉ đường tránh kẹt xe. Vậy AI, Machine Learning và Deep Learning khác nhau thế nào? Mười phút tới, ta sẽ trả lời câu hỏi đó.

## Cảnh 2

Đây là anh Nam, lập trình viên ở một công ty thương mại điện tử. Sếp giao việc: lọc email lừa đảo. Ngày đầu tiên, anh viết một luật: nếu email chứa "trúng thưởng" thì là spam. Chạy tốt. Nhưng đến ngày thứ ba, kẻ lừa đảo đổi cách viết: "trúng thuởng" sai dấu, chữ bị chấm ngắt quãng, hoặc đổi hẳn sang "nhận quà miễn phí". Luật cũ không bắt được nữa.

## Cảnh 3

Anh Nam thêm luật mới. Ngày thứ ba mươi, anh đã có năm trăm luật. Các luật bắt đầu chồng chéo nhau, và một email của chính sếp có chữ "thưởng Tết" cũng bị chặn. Rồi một đồng nghiệp gợi ý: sao không đưa cho máy mười nghìn email đã được đánh dấu spam hay bình thường, rồi để máy tự tìm ra quy luật? Ý tưởng đó có tên là Machine Learning, học máy.

## Cảnh 4

Hãy nghĩ đến việc dạy một đứa trẻ nhận ra con mèo. Cách thứ nhất: đưa luật. Mèo có bốn chân, có ria, tai nhọn, kêu meo meo. Đứa trẻ thuộc luật, nhưng gặp một con mèo bị cụt tai thì bối rối. Cách thứ hai: cho bé xem thật nhiều ảnh mèo và chó, nói đây là mèo, đây là chó. Bé tự rút ra cảm nhận, và gặp mèo cụt tai vẫn nhận ra. Lập trình truyền thống giống cách một. Machine Learning giống cách hai.

## Cảnh 5

Đây là điểm quan trọng nhất của bài. Trong lập trình truyền thống, ta đưa vào dữ liệu và luật, máy cho ra kết quả. Trong Machine Learning, ta đưa vào dữ liệu và kết quả đúng, còn máy tự tìm ra luật. Luật do máy học được ấy có tên riêng: mô hình. Sau khi có mô hình, ta dùng nó cho dữ liệu mới.

## Cảnh 6

Bắt đầu từ khái niệm rộng nhất: AI, Artificial Intelligence, trí tuệ nhân tạo. Đó là mọi kỹ thuật giúp máy tính làm được việc mà ta thường nghĩ cần trí thông minh của con người. Tìm đường nhanh nhất, chơi cờ vua, trả lời câu hỏi. Lưu ý: một chương trình cờ vua viết toàn bằng luật if else vẫn được gọi là AI, dù nó không học gì cả.

## Cảnh 7

Bên trong AI có một nhánh: Machine Learning, học máy. Ở đây, máy tự rút ra quy luật từ dữ liệu thay vì được lập trình từng luật. Ví dụ, ngân hàng dự đoán một khách hàng có trả nợ đúng hạn không, dựa trên lịch sử của hàng nghìn khách cũ.

## Cảnh 8

Bên trong Machine Learning lại có một nhánh nhỏ hơn: Deep Learning, học sâu. Deep Learning dùng mạng nơ-ron nhiều lớp để tự học những đặc điểm phức tạp từ dữ liệu thô như ảnh, âm thanh, văn bản. Face ID, camera đọc biển số xe và ChatGPT đều dùng Deep Learning. Mạng nơ-ron sẽ được học kỹ ở giai đoạn ba.

## Cảnh 9

Ghép lại, ta có ba vòng tròn lồng nhau như búp bê Nga. AI ở ngoài cùng, Machine Learning ở giữa, Deep Learning ở trong cùng. Nên nhớ một câu: mọi Deep Learning đều là Machine Learning, mọi Machine Learning đều là AI, nhưng chiều ngược lại thì không.

## Cảnh 10

Vậy khi nào dùng Machine Learning cổ điển, khi nào dùng Deep Learning? Với dữ liệu dạng bảng như giá nhà hay thông tin khách hàng, chỉ cần vài trăm đến vài chục nghìn dòng, máy tính thường là đủ, và kết quả dễ giải thích. Với ảnh, âm thanh, văn bản dài, Deep Learning thường mạnh hơn, nhưng cần nhiều dữ liệu hơn, thường cần card đồ họa, và khó giải thích hơn, giống một hộp đen. Mẹo nhớ: dữ liệu dạng bảng thì thử ML cổ điển trước.

## Cảnh 11

Máy có ba kiểu học chính. Kiểu đầu tiên là Supervised Learning, học có giám sát. Máy học từ dữ liệu đã có đáp án, đáp án này gọi là nhãn, tiếng Anh là label. Giống học sinh làm bài tập có đáp án ở cuối sách. Học có giám sát có hai dạng. Hồi quy, regression, đoán một con số, ví dụ căn nhà giá bao nhiêu tỷ. Phân loại, classification, đoán một nhóm, ví dụ email này là spam hay không.

## Cảnh 12

Kiểu thứ hai là Unsupervised Learning, học không giám sát. Lần này dữ liệu không có đáp án. Ví dụ, siêu thị có dữ liệu mua hàng của năm mươi nghìn khách. Máy tự gom những khách có thói quen giống nhau thành từng nhóm. Không ai nói trước có những nhóm nào. Sau đó con người mới nhìn vào và đặt tên, như "nhóm hay mua đồ ăn vặt".

## Cảnh 13

Kiểu thứ ba là Reinforcement Learning, học tăng cường. Máy học bằng cách thử, sai, rồi nhận thưởng hoặc phạt, giống dạy chó: làm đúng thì được bánh. AI chơi game và robot tập đi dùng cách này. Khóa học không đi sâu vào kiểu học này, nhưng bạn cần biết nó tồn tại.

## Cảnh 14

Tóm lại: có giám sát thì có đáp án, và trả lời "cái này là gì, bao nhiêu". Không giám sát thì không có đáp án, và trả lời "dữ liệu có những nhóm nào". Tăng cường chỉ có thưởng phạt, và trả lời "nên làm hành động nào".

## Cảnh 15

Đây là bản đồ khóa học của bạn. Nền tảng chung là Python, rồi Machine Learning, rồi Deep Learning. Từ đó tách ra các lĩnh vực. Computer Vision, thị giác máy tính, dạy máy nhìn và hiểu ảnh, như đọc biển số xe. NLP, xử lý ngôn ngữ tự nhiên, dạy máy đọc và viết ngôn ngữ của người, như phân loại bình luận khen hay chê. ChatGPT là AI tạo sinh cho văn bản, nằm ở chỗ Deep Learning gặp NLP. Chặng cuối là đưa mô hình thành sản phẩm.

## Cảnh 16

Giờ ta xem lại câu chuyện anh Nam bằng code Python. Bạn chưa cần hiểu từng dòng. Cách một là luật thủ công: nếu email có chữ "trúng thưởng" thì trả về spam. Cách hai là Machine Learning. Ta chuẩn bị tám email, bốn email spam và bốn email bình thường, mỗi email có nhãn đúng do con người dán.

## Cảnh 17

Ba dòng tiếp theo là phần "cho máy học". Bộ đếm từ biến mỗi email thành dãy số. Hàm fit nghĩa là cho máy học, tự tìm quy luật từ ví dụ. Giờ ta thử một email spam mới, không hề có chữ "trúng thưởng". Luật thủ công cho là bình thường, bị lọt. Còn mô hình Machine Learning bắt được, là spam. Ta không hề viết luật "miễn phí là spam". Máy tự nhận ra các từ như "miễn phí", "quà", "ngay" hay xuất hiện trong thư rác.

## Cảnh 18

Có ba hiểu nhầm hay gặp. Một: AI thì phải biết học. Sai, chương trình cờ vua dùng if else vẫn là AI. Hai: Deep Learning lúc nào cũng tốt hơn. Sai, với dữ liệu dạng bảng, hãy thử ML cổ điển trước. Ba: mô hình vừa rồi đoán đúng nên đã tin được. Cũng sai, tám email là quá ít, thực tế cần hàng nghìn email, và bài hai chấm năm sẽ dạy cách đo độ tốt của mô hình.

## Cảnh 19

Tóm lại bài hôm nay. Machine Learning đảo chiều cách lập trình: đưa dữ liệu và đáp án, máy tự tìm luật. AI chứa ML, ML chứa DL. Máy có ba kiểu học: có giám sát, không giám sát và tăng cường. Và dữ liệu dạng bảng thì thử ML cổ điển trước.

## Cảnh 20a

Một câu hỏi cho bạn: dự đoán số ly trà sữa bán được vào ngày mai là bài toán hồi quy hay phân loại? Hãy dừng lại vài giây và nghĩ thử.

## Cảnh 20b

Đáp án là hồi quy, vì ta đoán một con số.

## Cảnh 21

Bài tiếp theo, ta sẽ cài đặt môi trường và chạy notebook đầu tiên. Hẹn gặp lại bạn.

