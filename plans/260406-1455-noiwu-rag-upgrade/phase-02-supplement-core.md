# Phase 02: Bổ sung văn bản "xương sống"
Status: ⬜ Pending
Dependencies: Phase 01

## Objective
Bổ sung ~60 văn bản QPPL gốc cấp Trung ương mà dataset có thể thiếu,
đặc biệt các văn bản mới ban hành 2025-2026.

## Requirements
### Functional
- [ ] Tạo danh mục ~60 VB xương sống (10 lĩnh vực × ~6 VB)
- [ ] Kiểm tra từng VB: đã có trong filtered data chưa?
- [ ] Với VB thiếu: crawl từ thuvienphapluat.vn hoặc nhập thủ công
- [ ] Merge vào filtered_noiwu_docs.jsonl
- [ ] Đánh dấu VB xương sống bằng metadata `priority: "core"`

### Danh mục VB cần kiểm tra (từ yêu cầu user)

**Nền tảng chung:**
- Hiến pháp 2013 (sửa đổi NQ 203/2025)
- NĐ 25/2025/NĐ-CP (chức năng Bộ Nội vụ)

**1. Tổ chức bộ máy:**
- Luật 63/2025/QH15 (Tổ chức Chính phủ)
- NĐ 303/2025, 158/2018, 120/2020, 62/2020, 106/2020

**2. Chính quyền địa phương:**
- Luật 72/2025/QH15 (CQĐP)
- NQ 35/2023/UBTVQH15
- TT 27/2025/TT-BNV

**3. Cán bộ, công chức, viên chức:**
- Luật 80/2025/QH15 (CBCC)
- Luật 129/2025/QH15 (Viên chức)
- NĐ 170/2025, 334/2025, 251/2025, 171/2025, 27/2026, 33/2023
- TT 22/2025/TT-BNV, TT 03/2026/TT-BNV

**4. Tiền lương:**
- NQ 27-NQ/TW 2018
- NĐ 07/2026, 73/2024
- TT 01/2026, 23/2025, 24/2025/TT-BNV

**5. CCHC & Dân chủ:**
- NQ 76/NQ-CP 2021
- Luật Dân chủ cơ sở 2022
- NĐ 59/2023

**6. Thi đua, khen thưởng:**
- Luật 06/2022/QH15
- NĐ 152/2025
- TT 20/2025/TT-BNV

**7. Tôn giáo:**
- Luật 02/2016/QH14
- NĐ 95/2023

**8. Văn thư, lưu trữ:**
- Luật 33/2024/QH15
- NĐ 31/2026, 113/2025, 30/2020
- TT 06/2025/TT-BNV

**9. Thanh niên & Hội, Quỹ:**
- Luật 57/2020/QH14
- NĐ 03/2026, 45/2010, 13/2021

**10. LĐ-TB&XH (sáp nhập):**
- Luật BHXH 2024
- NĐ 372/2025, 158/2025, 85/2026, 338/2025, 374/2025
- TT 25/2025, 04/2026, 02/2026/TT-BNV

## Implementation Steps
1. [ ] Tạo `scripts/00b_check_core_docs.py` — kiểm tra VB nào đã có
2. [ ] Tạo `data/core_docs_manifest.json` — danh sách VB + URL TVPL
3. [ ] Tạo `scripts/00c_crawl_missing.py` — crawl VB thiếu từ TVPL
4. [ ] Merge vào filtered data, đánh dấu `priority: "core"`
5. [ ] Verify: tất cả 60 VB đều có trong output

## Files to Create/Modify
- `scripts/00b_check_core_docs.py`
- `scripts/00c_crawl_missing.py`
- `data/core_docs_manifest.json`
- `processed/filtered_noiwu_docs.jsonl` (append)

## Test Criteria
- [ ] 100% VB xương sống có trong data (~60 VB)
- [ ] Mỗi VB có content đầy đủ (>500 chars)
- [ ] Metadata đúng (doc_type, issuer, date)

---
Next Phase: → Phase 03 (Data Pipeline)
