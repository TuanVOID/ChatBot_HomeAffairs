# Phase 07: Test & Deploy
Status: ⬜ Pending
Dependencies: Phase 06

## Objective
Kiểm tra toàn bộ hệ thống end-to-end, fix bugs, optimize performance,
deploy qua ngrok để demo.

## Requirements
### Functional
- [ ] Test 10 câu hỏi (1 per domain) → verify câu trả lời chính xác
- [ ] Test citation: trích dẫn đúng nguồn
- [ ] Test domain filter: không trả kết quả ngoài lĩnh vực
- [ ] Test error handling: Ollama offline, query rỗng
- [ ] Git commit + push lên GitHub

### Non-Functional
- [ ] Response time <5s cho BM25-only
- [ ] Response time <10s cho hybrid search
- [ ] RAM usage <8GB (server + indexes)

## Test Scenarios

| # | Domain | Câu hỏi | Expected |
|---|--------|---------|----------|
| 1 | CBCCVC | "Điều kiện thi tuyển công chức?" | Luật CBCC 80/2025 |
| 2 | Tiền lương | "Mức lương cơ sở mới nhất?" | NĐ 73/2024 |
| 3 | CQĐP | "Bỏ cấp huyện từ khi nào?" | Luật CQĐP 72/2025 |
| 4 | Thi đua | "Ai có thẩm quyền tặng Huân chương?" | Luật TĐKT 06/2022 |
| 5 | Tôn giáo | "Thủ tục đăng ký hoạt động tôn giáo?" | Luật 02/2016 |
| 6 | Văn thư | "Quy trình quản lý văn bản đi?" | NĐ 30/2020 |
| 7 | BHXH | "Điều kiện hưởng lương hưu?" | Luật BHXH 2024 |
| 8 | Bộ máy | "Cơ cấu tổ chức Bộ Nội vụ mới?" | NĐ 25/2025 |
| 9 | CCHC | "Chỉ số PAR INDEX là gì?" | NQ 76/NQ-CP |
| 10 | Thanh niên | "Quyền và nghĩa vụ thanh niên?" | Luật TN 57/2020 |

## Implementation Steps
1. [ ] Chạy test 10 câu hỏi, ghi kết quả
2. [ ] Fix bugs nếu có
3. [ ] Optimize: tune RRF weights, top_k
4. [ ] Git commit tất cả thay đổi
5. [ ] Deploy: `.\start.ps1 -Ngrok` → share URL demo

## Deliverables
- [ ] Test report (pass/fail cho 10 scenarios)
- [ ] README.md cập nhật (chuyên ngành Nội vụ)
- [ ] PROGRESS.md cập nhật
- [ ] GitHub repo cập nhật
- [ ] Ngrok URL để demo

---
✅ PLAN COMPLETE
