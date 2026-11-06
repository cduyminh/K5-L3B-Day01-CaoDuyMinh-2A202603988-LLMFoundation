# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603988
- **Họ và tên:** Cao Duy Minh
- **SEED:** 4369

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.894 | 0.919 | 0.09 | 0 |
| 2. BoW + MLP | 0.934 | 0.919 | 0.22 | 0 |
| 3a. MiniLM + kNN | 0.908 | 0.976 | 61.7 | 0 |
| 3b. MiniLM + LogReg | 0.971 | 0.947 | 43.2 | 0 |
| 3c. MiniLM + MLP head | 0.960 | 0.947 | 40.3 | 0 |

BoW có precision 1.0 nhưng recall chỉ 0.81; embedding + LogReg cân bằng hơn (precision 0.988, recall 0.955).

## 3. Dự đoán và thực tế

1. **lr = 50 (ml_lr).** Tôi đoán loss zigzag liên tục vì nhảy vượt đáy. Thực tế loss chỉ vọt lên một lần (~2.6 ở epoch đầu) rồi giảm nhanh xuống 0.005. Lý do: bước đầu bias và các từ thông dụng bị đẩy mạnh về phía ham nên gần như mọi spam bị đoán là ham (0.13 × 20.7 ≈ 2.7, khớp đỉnh); bước sau các từ spam thật được đẩy lên mạnh. Loss của logistic regression là lồi, sigmoid giới hạn gradient, và tập train gần như tách được nên không có zigzag kéo dài.
2. **ml_f1.** Tôi đoán F1 ~0.7 và recall thấp hơn precision. Hướng đúng (P = 1.0, R = 0.809) nhưng F1 thực tế là 0.894, cao hơn dự đoán vì phần lớn spam trong tập test là spam lộ liễu, giống kiểu spam trong tập train.
3. **tradeoff.** Tôi đoán đường Pareto là 1, 2, 3b. Thực tế trên SUB là 1, 3b, 3a: BoW + MLP bị BoW + LogReg lấn át, còn kNN lên đường vì SUB cân bằng 20/20 nên precision thấp của kNN không bị phạt. Với 40 tin, mỗi tin sai đổi 1 − F1 khoảng 2.5 điểm %, nên thứ tự giữa các chấm gần nhau không đáng tin.

## 4. Phân tích 50 case khó

BoW tụt nhiều nhất: F1 từ 0.919 (SUB) xuống 0.368 trên 50 case khó; kNN tụt ít nhất (0.976 xuống 0.759). Nhóm khó nhất với BoW là thao túng / lừa đảo (12%) và né từ khoá (17%); với embedding là "nói về spam" (kNN 25%) và spam giọng tự nhiên.

- **#22 "Your bank account has been temporarily suspended…"**: BoW sai vì không có từ spam nào trong danh sách trọng số học được (call, txt, free…); nó chỉ nhìn token rời rạc.
- **#47 "lol I just got a text saying I won £1000…"** (ham): embedding đặt tin này cạnh spam trúng thưởng vì vector nắm chủ đề, không nắm ý định chế giễu của người gửi.

Spam giọng tự nhiên (#12, #13, #16, #19) đánh bại mọi model vì cả từ ngữ lẫn ý nghĩa bề mặt đều giống tin bạn bè. Điểm test thường quá lạc quan vì tập test được chia ngẫu nhiên từ cùng kho dữ liệu với train.

## 5. Thử thách cá nhân

Đề #2: giảm vocab còn 100 từ và tăng lên 5000 từ, train lại, báo F1 test.

Dự kiến: vocab 100 chủ yếu là từ chức năng (i, to, you…) nên mất nhiều từ spam đặc trưng, recall và F1 giảm (underfit). Từ 2000 lên 5000, F1 gần như không đổi vì các từ thêm vào đều hiếm: gradient của mỗi từ tỉ lệ với số tin chứa nó, nên trọng số của từ hiếm gần như không học được trong 300 epoch, và nếu học được thì dễ overfit.

Lưu ý trung thực: thử thách yêu cầu không dùng AI, nhưng tôi đã nhờ AI viết vòng lặp thử nghiệm và giải thích xu hướng (xem AI log). Kết quả chạy chưa có trong notebook đã nộp.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 tin/s), p95 < 50 ms/tin.

- Đạt độ trễ: BoW + LogReg (p95 0.09 ms), BoW + MLP (0.22 ms), 3b (43 ms). 3b đạt cả độ trễ lẫn chất lượng (F1 0.971) nhưng chỉ ~43 tin/s trên CPU Colab, nên cần 3–4 máy song song hoặc GPU.
- LLM không đạt p95 và tốn tiền: với gpt-6-luna (100 token input, 1 token output) là ~$105/ngày ≈ $3,150/tháng; gpt-6.1-terra gấp 10.
- Đề xuất router: BoW xử lý mọi tin (p < 0.1 cho qua, p > 0.9 chặn, vì precision BoW = 1.0); tin lưỡng lự chuyển sang 3b; tin còn lưỡng lự (p ∈ [0.3, 0.7]) mới gửi LLM, chạy bất đồng bộ để không phá SLA. Đánh đổi: hệ thống phức tạp hơn, nhưng nhanh, rẻ và bền hơn trước prompt injection vì LLM không phải lớp duy nhất. Thêm danh sách trắng đầu số OTP ngân hàng để tránh chặn nhầm tin như case #41.

## 7. AI log và điều đã học

Tôi dùng Claude (Claude Code) làm trợ giảng trong suốt buổi.

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 0 | Gợi ý Q0.1 | Giải thích data leakage | Tự viết câu trả lời |
| 1 (TODO 1.1–1.4) | Giải thích docstring, xin code vectorize, sigmoid, predict_proba | Giải thích và code mẫu cho 1.1, 1.2 | Tự viết 1.3, 1.4; AI chỉ ra lỗi (thiếu `np.asarray`, thiếu `return`), tôi tự sửa; test PASS |
| 1 (thí nghiệm, bắt lỗi) | Precision/recall/F1 khi mất cân bằng; xem lại dự đoán; đáp án Bắt lỗi 1 | Giải thích, nhận xét, đáp án | Tự viết dự đoán; đối chiếu kết quả thật; sửa giải thích lr = 50 theo cơ chế đúng |
| 2 | Giải thích train loop; dự đoán 2.A, 2.B | Giải thích, câu dự đoán mẫu | Tự viết 2.1 và train loop; đối chiếu val loss, F1 0.934, bỏ ReLU 0.929 |
| 3 | Code 3.2, 3.3; gợi ý câu hỏi 3, Bắt lỗi 2 | Viết top_k_similar, knn_predict | Tự viết 3.1; test PASS; đối chiếu biểu đồ PCA |
| 4 | Code 4.1–4.4; tỉ lệ giá model | Viết call_openai, prompt, parse_label, llm_classify_batch | Kiểm tra parse_label; đối chiếu PRICE_PER_1M |
| 5–6 | Pareto, stress test, case tự viết | Phân tích, 6 case gợi ý | Đối chiếu số theo nhóm |
| Tính tay | Tính 5 giá trị | Đáp số kèm bước | Tự chép tay, kiểm tra từng bước |
| Thử thách #2 | Code thử vocab 100 / 5000 | Vòng lặp và giải thích | Xem mục 5 |
| Nộp bài | Viết báo cáo này và đẩy lên repo | Soạn báo cáo từ số liệu trong notebook và lab1_report.json | Tôi kiểm tra lại trước khi nộp |

**Điều đã học:** Điểm trên tập test được chia ngẫu nhiên có thể lạc quan hơn nhiều so với thực tế khi kiểu dữ liệu thay đổi (BoW 0.92 xuống 0.37 trên case khó). Thứ giới hạn kết quả thường là cách biểu diễn đầu vào chứ không phải độ phức tạp của mô hình: bỏ ReLU gần như không đổi F1, còn đổi sang embedding mới tăng rõ. Và với dữ liệu mất cân bằng, accuracy gần như vô nghĩa nếu không nhìn precision, recall và F1 của lớp hiếm.
