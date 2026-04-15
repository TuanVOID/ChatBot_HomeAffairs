# CrawlData (Camoufox 3 Workers)

Tài liệu này mô tả luồng crawl hiện tại cho `thuvienphapluat.vn` bằng Camoufox + Playwright.

## 1) Script chính

- Crawler đơn worker: `08_camoufox_crawl_by_org.py`
- Điều phối custom 3 workers song song: `09_parallel_camoufox_custom_batch.py`
- Batch chạy nhanh (Windows): `09_start_camoufox_custom_batch_3workers.bat`

## 2) Tính năng chính

- 3 worker chạy song song (`w1`, `w2`, `w3`), mỗi worker có plan riêng.
- Hỗ trợ proxy theo worker.
- Hỗ trợ OCR captcha TVPL.
- Hỗ trợ xử lý Cloudflare (auto click + chờ manual fallback).
- Delay click dạng người dùng: random quanh `--delay`.
- Resume tiến độ theo worker.
- Chế độ strict chống miss dữ liệu:
  - Ghi `PAGE_MANIFEST` cho từng trang listing.
  - Ghi `DOC_STATUS` cho từng URL chi tiết (`ok/seen/failed`).
  - Trước khi kết thúc task, đối soát `manifest - success`.
  - Tự backfill trang thiếu cho đến khi đủ hoặc hết `--verify-max-rounds`.

## 3) Cấu trúc dữ liệu output

- Mặc định ghi vào `scripts/CrawlData/output/`
- File theo org/range, ví dụ:
  - `org_022_chinh-phu_p001.jsonl`
  - `org_026_quoc-hoi_p001.jsonl`
- Mỗi dòng JSONL có schema:
  - `url`
  - `title`
  - `content`
  - `meta`
  - `document_number`
  - `source`
  - `crawl_time`

## 4) State, log, resume

- Resume state:
  - `scripts/CrawlData/state/custom_batch_resume/w1.json`
  - `scripts/CrawlData/state/custom_batch_resume/w2.json`
  - `scripts/CrawlData/state/custom_batch_resume/w3.json`
- Log:
  - `scripts/CrawlData/logs/camoufox_custom_batch/w1.log`
  - `scripts/CrawlData/logs/camoufox_custom_batch/w2.log`
  - `scripts/CrawlData/logs/camoufox_custom_batch/w3.log`

`RESET_RESUME=1` chỉ reset tiến độ worker, không xóa dữ liệu trong `output/*.jsonl`.

## 5) Chạy bằng file BAT (khuyến nghị)

File: `09_start_camoufox_custom_batch_3workers.bat`

Bạn cấu hình:

- `W1_TASK_*`, `W2_TASK_*`, `W3_TASK_*` theo format:
  - `so_bai,link`
- `PROXY_W1`, `PROXY_W2`, `PROXY_W3`
- `RESET_RESUME`:
  - `0`: resume bình thường
  - `1`: reset tiến độ và crawl đối soát lại

Script tự tính số trang theo công thức:

- `pages = ceil(so_bai / 20)`

## 6) Chạy bằng CLI trực tiếp

```powershell
cd scripts/CrawlData
python 09_parallel_camoufox_custom_batch.py `
  --delay 11 `
  --viewport-width 1600 `
  --viewport-height 900 `
  --resume-state-dir state/custom_batch_resume `
  --output-dir output `
  --verify-max-rounds 4 `
  --plan-w1 "360,https://...org=22&page=1" `
  --plan-w2 "200,https://...org=26&page=1" `
  --plan-w3 "400,https://...org=33&page=1"
```

## 7) Giải thích skip worker (w1/w2 dừng ngay)

Nếu log có:

- `already completed from resume state, skip.`

thì worker đang đọc state cũ đã `completed=true`.

Cách xử lý:

- Đặt `RESET_RESUME=1` trong file `.bat` (chạy 1 lần), sau đó trả lại `0`.

## 8) Giới hạn hiện tại

- Một số trang chi tiết dạng PDF viewer/iframe có thể không trích được `content` HTML.
- Khi đó URL có thể bị `failed` (thường do `empty_content`) và cần bổ sung luồng đọc PDF nếu muốn coverage cao hơn.

## 9) Files liên quan nhanh

- `08_camoufox_crawl_by_org.py`: crawl 1 worker, parse document.
- `09_parallel_camoufox_custom_batch.py`: orchestration + strict verify + backfill.
- `09_start_camoufox_custom_batch_3workers.bat`: template chạy hằng ngày.

