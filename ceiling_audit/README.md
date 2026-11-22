# Laya Address VN — Bộ Công Cụ Đo Trần & Gán Nhãn 300 Mẫu (Offline 100%)

Thư mục này chứa trọn bộ công cụ chạy **hoàn toàn offline trên máy bạn** (không gọi API ngoài, không đưa chuỗi khách thật ra khỏi máy):
- **Bước 1 (`ceiling_eval.py`):** Đo song song **Trần Tiền-Goong (Pre-Goong)** và **Trần Hậu-Goong (Post-Goong Consistency)** trên lát cắt đơn đã lọt qua Luật D, kèm **Kiểm tra nhiễu vùng miền/lặp SĐT (Anti-Confounder Audit)**.
- **Bước 2 (`sample_300_for_labeling.py` + `score_300_labels.py`):** Lấy 300 mẫu phân tầng bất đồng + gán nhãn mù trên **HTML offline (`label_300_ui.html`)** hoặc **Excel**, rồi chấm **IPW Population-Weighted** trên toàn bộ tập đơn và xuất file **`anonymized_v3_brief.json` (ẩn danh 100%)**.

---

## 1. Cài đặt (chỉ cần nếu chạy từ môi trường mới)

```powershell
cd "C:\Users\Xin Chao\Pictures\Google Automation\laya_address_vn\ceiling_audit"
.\.venv\Scripts\python.exe -c "import sklearn, pandas, openpyxl, onnxruntime; print('READY')"
```

---

## 2. Chạy Bước 1 — Đo trần trên file DEV thật (34.263 đơn)

### Trường hợp A: File CSV/XLSX có tên cột chuẩn (`raw_text`, `outcome`, `rule_d_pass`, `goong_ward`, ...)
```powershell
cd "C:\Users\Xin Chao\Pictures\Google Automation\laya_address_vn\ceiling_audit"
.\.venv\Scripts\python.exe ceiling_eval.py `
  --data "D:\path\to\dev_orders.csv" `
  --output-dir "outputs\step1_dev" `
  --target-ratio 1.5 `
  --min-caught-returns 30
```

### Trường hợp B: Truyền tên cột cụ thể + chưa chấm Way A/B v2 sẵn (`--score-v2`)
```powershell
.\.venv\Scripts\python.exe ceiling_eval.py `
  --data "D:\path\to\dev_orders.xlsx" `
  --text-col "dia_chi" `
  --outcome-col "trang_thai" `
  --rule-d-col "luat_d_pass" `
  --phone-col "sdt" `
  --time-col "created_at" `
  --goong-ward-col "goong_xa" `
  --goong-district-col "goong_huyen" `
  --goong-province-col "goong_tinh" `
  --score-v2 `
  --output-dir "outputs\step1_dev"
```
> *Lưu ý:* Nếu cột Luật D của bạn mang nghĩa `1 = BỊ CHẶN` (thay vì `1 = CHO QUA`), thêm cờ `--rule-d-is-block`.

### Đầu ra Bước 1 (`outputs\step1_dev\`)
- **`comparison_sheet.xlsx`** (4 sheet: `Summary_Comparison`, `Operating_Curves`, `Confounder_Audit`, `Top_Blocked_Preview`)
- **`comparison_sheet.csv`**
- **`ceiling_report.md`** (tóm tắt kết quả + Kết luận tự động `GO_POST_GOONG` / `CAUTION_CONFOUNDER_SUSPECTED` / `NO_GO_FOCUS_FORM_PREVENTION`)
- **`ceiling_report.json`** (ẩn danh 100%, an toàn để gửi đi)
- **`oof_predictions.csv`** (dùng làm đầu vào cho Bước 2)

---

## 3. Chạy Bước 2 — Lấy 300 mẫu phân tầng, gán nhãn mù & chấm điểm IPW

### Bước 2a: Trích 300 mẫu mù phân tầng trên lát cắt đã qua Luật D
```powershell
.\.venv\Scripts\python.exe sample_300_for_labeling.py `
  --data "D:\path\to\dev_orders.csv" `
  --oof-csv "outputs\step1_dev\oof_predictions.csv" `
  --output-dir "outputs\step2_300"
```
> **Cách gán nhãn tiện nhất:** Mở đúp file **`outputs\step2_300\label_300_ui.html`** bằng Chrome/Edge (chạy offline qua `file://`, có phím tắt `1/2/3`, `Q/W/E/R`, `←/→`, tự lưu vào trình duyệt), xong bấm **Xuất `labeled_300.csv`**. Hoặc điền trực tiếp file Excel `blind_label_sheet_300.xlsx` (có dropdown sẵn).

### Bước 2b: Chấm độ chính xác thật & xuất tóm tắt ẩn danh cho v3
```powershell
.\.venv\Scripts\python.exe score_300_labels.py `
  --labels "outputs\step2_300\labeled_300.csv" `
  --key "outputs\step2_300\sample_300_key.csv" `
  --ceiling-report "outputs\step1_dev\ceiling_report.json" `
  --output-dir "outputs\step2_300"
```

### Đầu ra Bước 2 (`outputs\step2_300\`)
- **`label_300_eval_report.md`** — Báo cáo phân rã % hoàn do địa chỉ/Goong vs hoàn nền, độ chính xác từng cổng theo **IPW ước lượng toàn tập**.
- **`anonymized_v3_brief.json`** — **Cam kết 0% PII**: chỉ chứa bảng đếm tổng hợp, ma trận lỗi và đường cong trần — tuyệt đối không chứa chuỗi khách hàng, SĐT hay mã đơn.

---

## 4. Tự kiểm tra bằng bộ dữ liệu demo tổng hợp + Pytest

```powershell
cd "C:\Users\Xin Chao\Pictures\Google Automation\laya_address_vn\ceiling_audit"
.\.venv\Scripts\pytest.exe -v --basetemp=".pytest_tmp" tests\test_ceiling_audit.py
```
