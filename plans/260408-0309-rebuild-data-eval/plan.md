# Plan: Rebuild Data từ Raw-Data + Hệ thống Evaluation
Created: 2026-04-08T03:09
Status: 🟡 In Progress

## Overview
Thay thế toàn bộ data cũ (518K docs từ HuggingFace → lọc 176K) bằng **55 văn bản xương sống** 
từ `data/raw-data/`. Xây dựng hệ thống evaluation có cấu trúc để đánh giá, cải thiện liên tục.

## Ưu điểm của data mới
- **Chất lượng cao**: VB chọn lọc thủ công, đúng trọng tâm Nội vụ
- **Mới nhất**: Nhiều Luật 2025, NĐ 2026 (data HuggingFace chỉ đến 2023)
- **Size nhỏ hơn 1000x**: ~4MB text vs 4.3GB → index nhanh, search nhanh
- **Dễ mở rộng**: User chỉ cần thêm file .txt vào raw-data/

## Phases

| Phase | Name | Status | Tasks | Est. Time |
|-------|------|--------|-------|-----------|
| 01 | Import raw-data → documents.jsonl | ⬜ Pending | 5 | 15 min |
| 02 | Chunk documents | ⬜ Pending | 3 | 5 min |
| 03 | Build BM25 + FAISS indexes | ⬜ Pending | 4 | 10 min |
| 04 | Backend + Config update | ⬜ Pending | 3 | 10 min |
| 05 | Evaluation framework | ⬜ Pending | 6 | 30 min |
| 06 | Golden test queries | ⬜ Pending | 3 | 15 min |
| 07 | Test & Deploy | ⬜ Pending | 4 | 15 min |

**Tổng:** ~1.5 giờ (so với ~8h trước cho 176K docs)

## Quick Commands
- Start Phase 1: `/code phase-01`
- Check progress: `/next`

## So sánh Data cũ vs mới

| Metric | Data cũ (HF) | Data mới (raw) |
|--------|-------------|----------------|
| Sources | 518K docs HuggingFace | 55 VB chọn lọc thủ công |
| Relevant docs | ~176K sau filter | 55 (100% relevant) |
| Chunks | ~3.9M | ~5-10K (dự kiến) |
| BM25 index | ~35GB temp, 5h build | ~50MB, <5 min |
| FAISS index | Chưa build | ~20MB, <10 min |
| Newest VB | 2023 | NĐ 07/2026, NĐ 85/2026 |
