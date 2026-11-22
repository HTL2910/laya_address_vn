# Laya Address VN

Bộ cổng xác thực & kiểm tra độ đủ đến Xã/Phường cho chuỗi địa chỉ giao hàng Việt Nam (từ bộ lọc 0đ chạy trực tiếp trên trình duyệt đến dịch vụ CPU backend và bộ công cụ đo trần dữ liệu thật offline).

> **Chính sách dữ liệu:** Repo này tuyệt đối không chứa dữ liệu khách hàng thật hay SĐT. Mọi danh mục hành chính đều dùng nguồn mở MIT; mọi script kiểm tra trên đơn thật trong [`ceiling_audit/`](ceiling_audit/README.md) chạy offline 100% ở máy cục bộ và bị loại khỏi Git qua [`.gitignore`](.gitignore).

## Cấu trúc dự án

| Thư mục | Mô tả | Tài liệu chi tiết |
|---|---|---|
| `way_a_browser_onnx/` | **Way A v1** — Char-CNN ONNX chạy trong trình duyệt (`<= 1 MB`) | [way_a_browser_onnx](way_a_browser_onnx/) |
| `way_b_backend_api/` | **Way B v1** — Dual-Granularity Transformer + Char-CNN ONNX INT8 FastAPI | [way_b_backend_api](way_b_backend_api/) |
| `way_a_browser_onnx_v2/` | **Way A v2** — Bản nâng cấp 3 lớp `sufficiency`, vendored `onnxruntime-web@1.20.1`, tổng `1.283 MB` | [MODEL_CARD.md](way_a_browser_onnx_v2/MODEL_CARD.md) |
| `way_b_backend_api_v2/` | **Way B v2** — Bản nâng cấp 3 lớp `sufficiency`, ONNX INT8 (`1.789 MB`), FastAPI stateless | [MODEL_CARD.md](way_b_backend_api_v2/MODEL_CARD.md) |
| `shared_data/` | Danh mục hành chính Việt Nam chung (63 tỉnh, 656 huyện, 7.044 xã + sáp nhập 2025) | [PROVENANCE.json](shared_data/PROVENANCE.json) |
| `benchmark/` | Bộ kiểm chuẩn v1 (90 ca) & báo cáo đối chiếu | [shared_eval_report.json](benchmark/shared_eval_report.json) |
| `benchmark_v2/` | Bộ hold-out độc lập 600 ca (khác seed/mẫu sinh), kiểm tra rò rỉ n-gram & acceptance tests | [eval_report_v2.json](benchmark_v2/eval_report_v2.json) |
| `ceiling_audit/` | **Bước 1 & Bước 2 Offline Audit** — Đo trần Pre-Goong & Post-Goong, kiểm tra nhiễu vùng miền và gán nhãn mù 300 mẫu IPW | [README.md](ceiling_audit/README.md) |

## Nguồn dữ liệu hành chính & giấy phép

- Nguồn: [`tranngocminhhieu/vietnamadminunits`](https://github.com/tranngocminhhieu/vietnamadminunits) (**MIT License** — xem [`LICENSE_vietnamadminunits.txt`](shared_data/LICENSE_vietnamadminunits.txt)).
- Tổng hợp từ danh mục GSO (`danhmuchanhchinh.gso.gov.vn`) kết hợp bộ sáp nhập 34 tỉnh / 3.321 xã năm 2025.

## Bắt đầu nhanh

- **Đo trần trên log đơn DEV thật (offline hoàn toàn):** Xem hướng dẫn & lệnh PowerShell chi tiết trong [`ceiling_audit/README.md`](ceiling_audit/README.md).
- **Thông số kỹ thuật & đánh giá v2:** Xem [`way_a_browser_onnx_v2/MODEL_CARD.md`](way_a_browser_onnx_v2/MODEL_CARD.md) và [`way_b_backend_api_v2/MODEL_CARD.md`](way_b_backend_api_v2/MODEL_CARD.md).
