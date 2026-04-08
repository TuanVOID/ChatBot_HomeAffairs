# Plan: Nâng cấp Legal RAG → Chuyên ngành Nội vụ
Created: 2026-04-06 14:55
Status: 🟡 In Progress

## Overview
Biến chatbot tra cứu pháp luật tổng quát thành chatbot chuyên ngành Nội vụ,
bao gồm 10 lĩnh vực: Tổ chức bộ máy, Cán bộ công chức, Chính quyền địa phương,
Tiền lương, CCHC, Thi đua khen thưởng, Tôn giáo, Văn thư lưu trữ, Thanh niên,
và LĐ-TB&XH (sáp nhập theo NĐ 25/2025).

## Tech Stack (Giữ nguyên)
- Backend: FastAPI (Python)
- LLM: Ollama → Qwen2.5-7B-Instruct
- Embedding: Qwen3-Embedding-0.6B
- Search: BM25 (Whoosh) + FAISS (Vector)
- Frontend: Vanilla HTML/JS/CSS
- Network: pyngrok

## Phases

| Phase | Name | Status | Tasks | Est. Time |
|-------|------|--------|-------|-----------|
| 01 | Lọc data Nội vụ từ dataset | ✅ Done | 6 | 5 min |
| 02 | Bổ sung VB "xương sống" | ⏩ Skipped | 5 | (later) |
| 03 | Data pipeline (Ingest + Chunk) | ✅ Done | 4 | 14 min |
| 04 | Build indexes (BM25 + FAISS) | 🟡 In Progress | 5 | ~1.5 hrs |
| 05 | Backend chuyên ngành | ⬜ Pending | 6 | 1 hr |
| 06 | UI chuyên ngành Nội vụ | ⬜ Pending | 5 | 1 hr |
| 07 | Test & Deploy | ⬜ Pending | 5 | 30 min |

**Tổng:** 36 tasks | Ước tính: 5-8 sessions

## Quick Commands
- Start Phase 1: `/code phase-01`
- Check progress: `/next`
- Save context: `/save-brain`
