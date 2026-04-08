# Phase 01: Lọc data Nội vụ từ dataset
Status: ⬜ Pending
Dependencies: Không

## Objective
Viết script lọc ~30-50K docs liên quan Nội vụ từ dataset 518K docs,
thay vì full ingestion lãng phí 90% data không liên quan.

## Requirements
### Functional
- [ ] Đọc metadata parquet (streaming, không load hết RAM)
- [ ] Lọc theo `legal_sectors` (10 lĩnh vực Nội vụ)
- [ ] Lọc theo `issuing_authority` (Quốc hội, Chính phủ, Bộ Nội vụ, Bộ LĐ-TB&XH...)
- [ ] Lọc theo `doc_type` (Luật, NĐ, QĐ, TT, NQ — bỏ Công văn, Thông báo)
- [ ] Join với content files để lấy nội dung đầy đủ
- [ ] Output: `filtered_noiwu_docs.jsonl`

### Non-Functional
- [ ] Chạy trong <10 phút (streaming parquet)
- [ ] Không OOM (batch processing 5000 rows)

## Bộ lọc chi tiết

### Legal Sectors (lĩnh vực pháp luật)
```python
NOIWU_SECTORS = [
    "Bộ máy hành chính",
    "Lao động - Tiền lương",
    "Cán bộ - Công chức - Viên chức",
    "Bảo hiểm",
    "Việc làm",
    "Thi đua - Khen thưởng - các danh hiệu",
    "Văn hóa - Xã hội",       # Tôn giáo, Thanh niên
    "Lĩnh vực khác",          # Văn thư, Lưu trữ
    "Quyền dân sự",           # Dân chủ cơ sở
]
```

### Issuing Authorities (cơ quan ban hành)
```python
NOIWU_ISSUERS = [
    "Quốc hội", "Ủy ban thường vụ Quốc hội",
    "Chính phủ", "Thủ tướng Chính phủ",
    "Bộ Nội vụ",
    "Bộ Lao động - Thương binh và Xã hội",
    "Bộ Tài chính",           # Tiền lương, BHXH
]
```

### Document Types (loại văn bản)
```python
PRIORITY_TYPES = [
    "Luật", "Bộ luật",
    "Nghị quyết",
    "Nghị định",
    "Quyết định",
    "Thông tư",
    "Thông tư liên tịch",
    "Chỉ thị",
    "Hiến pháp",
]
```

## Implementation Steps
1. [ ] Tạo `scripts/00_filter_noiwu.py`
2. [ ] Implement streaming parquet reader (batch 5000)
3. [ ] Implement multi-criteria filter (sectors OR issuers match)
4. [ ] Join filtered IDs với content parquet files
5. [ ] Output filtered_noiwu_docs.jsonl + stats
6. [ ] Test chạy: verify count, spot-check nội dung

## Files to Create/Modify
- `scripts/00_filter_noiwu.py` — Script lọc chính
- `config/noiwu_filters.py` — Định nghĩa bộ lọc (sectors, issuers, types)

## Test Criteria
- [ ] Output >10,000 docs (ước tính 30-50K)
- [ ] Spot-check 10 docs random → đúng liên quan Nội vụ
- [ ] Các VB gốc TW (NĐ, Luật) có trong output
- [ ] Không chứa docs hoàn toàn không liên quan (VD: Giáo dục thuần túy)

---
Next Phase: → Phase 02 (Bổ sung VB xương sống)
