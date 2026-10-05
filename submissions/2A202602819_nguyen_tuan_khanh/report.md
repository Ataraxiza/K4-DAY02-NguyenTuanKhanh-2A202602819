# Báo cáo Lab Day 2: Backbone và công thức huấn luyện

## 1. Tóm tắt

Báo cáo này chỉ bao quát phần đã làm đến hết Bước 2 trong notebook: kiểm tra dữ liệu và pipeline, so sánh ba backbone, rồi thử các ablation công thức huấn luyện. Thí nghiệm backbone B01 cho thấy EfficientNet-B0 có macro-F1 validation cao nhất trong ba mô hình đã chạy (0,8548). Bước 2 chưa có bảng kết quả: cell dừng ở thí nghiệm CutMix do lỗi trong `losses.py`; vì vậy chưa thể kết luận phương pháp huấn luyện nào tốt hơn. Các số dưới đây chỉ là kết quả đã lưu trong notebook, không bổ sung hay suy diễn số liệu thiếu.

## 2. Dữ liệu và thiết lập

Dùng DeepWeeds, fold 0, với 10.501 ảnh train, 3.501 ảnh validation và 3.507 ảnh test, tổng cộng 17.509 ảnh. Kiểm tra trong notebook ghi nhận không có filename giao nhau giữa các tập và không thiếu ảnh. Phân bố toàn bộ dữ liệu gồm 9.106 ảnh Negative và 1.009 ảnh Rubber Vine ở hai đầu; tỷ lệ lớp lớn nhất/nhỏ nhất là 9,02 lần. Mất cân bằng này khiến macro-F1 phù hợp hơn accuracy để so sánh mô hình.

| Tập | Chinee Apple | Lantana | Parkinsonia | Parthenium | Prickly Acacia | Rubber Vine | Siam Weed | Snake Weed | Negatives | Tổng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 675 | 637 | 618 | 613 | 637 | 605 | 644 | 609 | 5.463 | 10.501 |
| Val | 225 | 213 | 206 | 204 | 212 | 202 | 215 | 203 | 1.821 | 3.501 |
| Test | 226 | 213 | 207 | 205 | 213 | 202 | 215 | 204 | 1.822 | 3.507 |

Notebook có biểu đồ phân bố lớp và lưới ảnh huấn luyện mẫu theo lớp. Các ảnh cho thấy nhiều cảnh có nền thực vật khô, bóng đổ và đối tượng nhỏ trong khung hình, nên bối cảnh và ánh sáng có thể làm việc phân biệt lớp khó hơn. Không có đánh giá định lượng về độ khó nhầm lẫn giữa các lớp trong các kết quả hiện có.

### Kiểm tra pipeline

Kiểm tra sanity dùng seed 42, batch 8 ảnh kích thước 224 × 224 và GPU Tesla T4. Cross-entropy ban đầu đo được 2,1919, gần giá trị kỳ vọng $-\ln(1/9) = 2,1972$. Khi overfit một batch, loss giảm từ 2,181829 xuống 0,000107 sau 100 bước và accuracy đạt 1,0000. Đây là bằng chứng pipeline cơ bản có thể học và khớp một batch nhỏ; không phải bằng chứng về khả năng tổng quát hóa.

## 3. Bước 1: So sánh backbone

Các lần chạy B01 dùng seed 0; cấu hình mặc định của `Config` là fine-tuning, 12 epoch, batch 64, ảnh 224, AdamW, learning rate 1e-4 cho backbone và 1e-3 cho head, warmup 1 epoch, AMP bật. Bảng dưới lấy trực tiếp từ bảng kết quả validation được lưu trong notebook.

| Backbone | Tham số (M) | GMAC | Macro-F1 val | Top-1 val | Thời gian/epoch (s) | Epoch tốt nhất |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 4,02 | 0,44 | **0,8548** | **0,8900** | 55,98 | 12 |
| ResNet-50 | 23,53 | 6,41 | 0,8094 | 0,8626 | 63,19 | 12 |
| ResNet-18 | 11,18 | 2,75 | 0,7755 | 0,8320 | 52,55 | 11 |

Trong ba mô hình đã thử, EfficientNet-B0 đứng đầu về macro-F1 và top-1 validation. So với ResNet-50, mô hình này có macro-F1 cao hơn 0,0454, top-1 cao hơn 0,0274, ít tham số hơn và GMAC thấp hơn theo bảng benchmark. ResNet-18 có thời gian mỗi epoch ngắn nhất, thấp hơn EfficientNet-B0 3,43 giây, nhưng điểm validation thấp hơn. Theo kết quả hiện có, notebook chọn EfficientNet-B0 để đi tiếp.

Đây mới là so sánh sàng lọc với một seed và ba kiến trúc. Mục tiêu GUIDE/RUBRIC yêu cầu ít nhất năm backbone, gồm cả họ ResNeXt/ConvNeXt và transformer; các nhóm này chưa có trong kết quả, nên chưa thể xem Bước 1 là hoàn tất hoặc khái quát thứ hạng sang các họ kiến trúc khác. Ngoài ra, Bước 2 bên dưới lại đặt backbone nền là ResNet-50, không phải EfficientNet-B0 đã chọn ở Bước 1; vì vậy đây không phải phép tiếp nối trực tiếp trên backbone được chọn.

Notebook cũng ghi nhận cảnh báo PyTorch về thứ tự gọi `lr_scheduler.step()` trước `optimizer.step()` trong lúc huấn luyện và cảnh báo API `GradScaler` cũ. Đây là các điểm cần xem xét khi diễn giải/tái lập kết quả; báo cáo giữ nguyên các số đã ghi và không điều chỉnh chúng.

## 4. Bước 2: Ablation công thức huấn luyện

Cấu hình nền được khai báo là ResNet-50 fine-tune, seed 0, 12 epoch, batch 64, augmentation basic, cross-entropy, không EMA và không Mixup/CutMix. Các biến thể được khai báo lần lượt là T01 đóng băng backbone, T02 label smoothing 0,1, T03 focal loss với gamma 2, T04 CutMix, T05 EMA và T06 RandAugment.

Theo trình tự output đã lưu, T00 đến T03 đã chạy xong để vòng lặp bắt đầu T04. T04 dừng khi gọi `mix_batch` với CutMix: `torch.sqrt(1.0 - lam)` nhận `lam` dạng Python float và phát sinh `TypeError`. Do đó cell không tới bước tạo/hiển thị bảng `training_df`, và notebook không lưu các macro-F1/top-1/epoch tốt nhất hay delta của T00–T03 trong output này. Các số liệu đó không được suy đoán trong báo cáo. T05, T06 và cấu hình kết hợp T07 chưa được chạy tới.

Vì không có bảng metric ablation, hiện chưa thể trả lời liệu đóng băng backbone, label smoothing hay focal loss có cải thiện validation hay không; cũng chưa đủ bằng chứng cho yêu cầu tối thiểu ba trục có kết quả so sánh và một cấu hình kết hợp. Kết luận hợp lệ ở giai đoạn này chỉ là đã khởi chạy các phép thử T00–T03 và phát hiện lỗi thực thi tại CutMix.

Lỗi CutMix trong `losses.py` đã được sửa bằng phép căn bậc hai trên scalar Python. Theo yêu cầu, không chạy lại `losses.py` hoặc cell notebook để kiểm tra; do đó việc sửa chưa được xác nhận bằng thực thi và không thay đổi trạng thái/kết quả đã ghi của notebook.

## 5. Kết luận trong phạm vi báo cáo

EDA xác nhận split fold 0 không giao nhau và bao phủ đủ 17.509 ảnh; kiểm tra sanity cho thấy loss ban đầu hợp lý và mô hình có thể overfit một batch. Trong ba backbone đã chạy, EfficientNet-B0 có validation tốt nhất và đã được chọn trong notebook. Tuy nhiên, số backbone còn thiếu so với yêu cầu, còn các ablation Bước 2 chưa có metric lưu được và CutMix dừng vì lỗi. Vì thế chưa có căn cứ để kết luận công thức huấn luyện nào tốt hơn hoặc hoàn tất mục tiêu so sánh đến hết Bước 2.

Phạm vi báo cáo kết thúc tại Bước 2. Không đưa ra kết quả hay kết luận cho các phần sau.