# Phase 03: Data Pipeline — Ingest + Chunk
Status: ⬜ Pending
Dependencies: Phase 02

## Objective
Chạy pipeline xử lý data đã lọc: normalize → chunk theo cấu trúc pháp luật
(Chương/Điều/Khoản/Điểm).

## Requirements
### Functional
- [ ] Sửa `01_ingest.py` đọc từ `filtered_noiwu_docs.jsonl` (thay vì parquet)
- [ ] Hoặc tạo mode mới: `--source jsonl`
- [ ] Chunk theo cấu trúc Điều/Khoản (đã có trong `02_chunk.py`)
- [ ] Thêm metadata chunk: `domain` = lĩnh vực Nội vụ (1-10)

### Non-Functional
- [ ] Chạy trong <30 phút cho 30-50K docs
- [ ] Output chunks.jsonl < 500MB

## Implementation Steps
1. [ ] Sửa `config/settings.py`: thêm path cho filtered data
2. [ ] Sửa `scripts/01_ingest.py`: hỗ trợ đọc JSONL trực tiếp
3. [ ] Chạy ingestion: `python scripts/01_ingest.py --source filtered`
4. [ ] Chạy chunking: `python scripts/02_chunk.py`

## Files to Modify
- `config/settings.py` — Thêm FILTERED_DATA_PATH
- `scripts/01_ingest.py` — Thêm mode đọc JSONL
- `scripts/02_chunk.py` — Thêm domain metadata vào chunks

## Test Criteria
- [ ] documents.jsonl chứa đúng số docs đã lọc
- [ ] chunks.jsonl: mỗi chunk có `domain` field
- [ ] Spot-check: chunk content đúng cấu trúc Điều/Khoản

## Expected Output
- `processed/documents.jsonl` — ~30-50K docs (~1-3 GB)
- `processed/chunks.jsonl` — ~500K-1M chunks (~200-500 MB)
- `processed/manifest.json` — stats

---
Next Phase: → Phase 04 (Build Indexes)
