# Phase 05: Backend chuyên ngành Nội vụ
Status: ⬜ Pending
Dependencies: Phase 04

## Objective
Nâng cấp backend: prompt chuyên ngành, API lọc theo lĩnh vực,
metadata-aware search, và cải thiện chất lượng trả lời.

## Requirements
### Functional
- [ ] System prompt chuyên ngành Nội vụ (context rõ 10 lĩnh vực)
- [ ] API filter theo lĩnh vực: `/api/chat?domain=tien_luong`
- [ ] Metadata-aware ranking: ưu tiên VB gốc TW > VB địa phương
- [ ] Citation format chuẩn: "Theo Điều X, Khoản Y, NĐ Z/2025/NĐ-CP"
- [ ] API endpoint mới: `/api/domains` — trả danh sách 10 lĩnh vực
- [ ] Tăng context window: top-5 → top-8 chunks

## Implementation Steps
1. [ ] Sửa `src/llm/prompt_builder.py`: prompt chuyên ngành
2. [ ] Sửa `src/retrieval/hybrid.py`: thêm domain filter
3. [ ] Sửa `src/retrieval/hybrid.py`: ưu tiên priority="core"
4. [ ] Thêm `/api/domains` endpoint trong `server.py`
5. [ ] Sửa `/api/chat`: nhận param `domain` (optional)
6. [ ] Tăng `top_k` và cải thiện RRF scoring

## Files to Modify
- `src/llm/prompt_builder.py` — System prompt mới
- `src/retrieval/hybrid.py` — Domain filter + priority boost
- `server.py` — API mới + filter param

## Test Criteria
- [ ] Query "chế độ nghỉ hưu" → trả về VB BHXH, không trả VB Giáo dục
- [ ] Query + domain filter → chỉ trả VB trong lĩnh vực đó
- [ ] VB xương sống luôn ranked cao hơn VB địa phương cùng nội dung
- [ ] Citation format đúng chuẩn

---
Next Phase: → Phase 06 (UI chuyên ngành)
