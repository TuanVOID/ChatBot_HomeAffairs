# Phase 01: Import Raw-Data → documents.jsonl
Status: ⬜ Pending
Dependencies: None

## Objective
Parse 55 file .txt từ `data/raw-data/` thành `processed/documents.jsonl` chuẩn.

## Format đầu vào (.txt)
```
Dòng 1: URL thuvienphapluat.vn (optional)
Dòng 2+: Nội dung văn bản (copy từ web)
```

## Format đầu ra (documents.jsonl)
Mỗi dòng 1 JSON:
```json
{
  "doc_id": "luat-can-bo-cong-chuc-so-80-2025",
  "title": "Luật Cán bộ, công chức",
  "document_number": "80/2025/QH15",
  "doc_type": "Luật",
  "issuer": "Quốc hội",
  "issue_date": "2025-06-24",
  "effective_date": "2025-07-01",
  "source_url": "https://thuvienphapluat.vn/...",
  "sectors": ["Cán bộ công chức", "Bộ máy hành chính"],
  "content": "Toàn bộ nội dung văn bản..."
}
```

## Implementation Steps
1. [ ] Viết `scripts/import_raw_data.py` — parse metadata từ nội dung text
   - Extract: số hiệu, loại VB, cơ quan ban hành, ngày ban hành
   - Regex detect: "Luật số:", "Nghị định", "Thông tư", "Nghị quyết"
   - Tải sector từ URL (nếu có)
2. [ ] Parse tất cả 55 files → `processed/documents.jsonl`  
3. [ ] Log stats: tổng docs, phân bố theo loại VB, kích thước
4. [ ] Validate: mỗi doc có đủ fields required
5. [ ] Xóa data cũ (chunks.jsonl, chunks_tokenized.jsonl)

## Files to Create/Modify
- `scripts/import_raw_data.py` — NEW: parser chính
- `processed/documents.jsonl` — OUTPUT

## Test Criteria
- [ ] 55 documents parsed thành công
- [ ] Mỗi doc có: doc_id, title, doc_type, content
- [ ] Không có doc trùng lặp
