# Phase 04: Build Indexes (BM25 + FAISS)
Status: ⬜ Pending
Dependencies: Phase 03

## Objective
Build BM25 index và FAISS vector index cho toàn bộ chunks đã xử lý.
Đây là bước quan trọng nhất để hybrid search hoạt động chính xác.

## Requirements
### Functional
- [ ] BM25 index với Vietnamese tokenization (underthesea)
- [ ] FAISS index với Qwen3-Embedding-0.6B
- [ ] Cài đặt underthesea cho Vietnamese word segmentation
- [ ] Cache embeddings để không phải tính lại

### Non-Functional
- [ ] BM25 build: <5 phút
- [ ] FAISS build: ~1-2 giờ (tùy số chunks, GPU accelerated)
- [ ] VRAM usage < 8GB khi embedding

## Implementation Steps
1. [ ] Cài đặt underthesea: `pip install underthesea`
2. [ ] Sửa BM25 searcher: tích hợp underthesea tokenizer
3. [ ] Build BM25: `python scripts/03_index_bm25.py`
4. [ ] Pull embedding model: `ollama pull qwen3-embedding:0.6b`
5. [ ] Build FAISS: `python scripts/04_index_vector.py`

## Files to Modify
- `src/retrieval/bm25_searcher.py` — Tích hợp underthesea
- `scripts/03_index_bm25.py` — Rebuild cho data mới
- `scripts/04_index_vector.py` — Build vector index

## Test Criteria
- [ ] BM25 search: "quyền sử dụng đất" → trả về VB liên quan
- [ ] Vector search: "chế độ thai sản BHXH" → trả về NĐ BHXH
- [ ] Hybrid search: kết hợp cả 2 → ranking chính xác hơn
- [ ] Search "Điều 128 Luật Đất đai" → trả về đúng Điều 128

## Expected Output
- `indexes/bm25/` — Whoosh index
- `indexes/faiss/` — FAISS index + ID mapping
- `indexes/embeddings_cache/` — Cached embeddings (pickle)

---
Next Phase: → Phase 05 (Backend chuyên ngành)
