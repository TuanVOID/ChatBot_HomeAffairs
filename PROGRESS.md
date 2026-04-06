# 📝 PROGRESS — Development Log

---

## [2026-04-05] Phase 0 & Phase 1: Setup + Ingestion

### Phase 0: Thiết lập môi trường
- ✅ Xác nhận Python 3.11.9, pyarrow 23.0.1, fastapi 0.135.2 đã sẵn có
- ✅ Cài thêm: `pandas 3.0.2`, `loguru 0.7.3`, `python-dotenv 1.2.2`
- ✅ Tạo cấu trúc thư mục dự án: `config/`, `scripts/utils/`, `src/`, `web/`, `processed/`, `indexes/`, `logs/`
- ✅ Tạo `requirements.txt` với đầy đủ dependencies
- ✅ Tạo `.env` + `.env.example` cho config
- ✅ Tạo `config/settings.py` — centralized config đọc từ .env

### Phase 1: Ingestion & Normalize
- ✅ Khảo sát dataset `vietnamese-legal-documents`: 518,255 docs
  - Metadata: 1 parquet file (~82MB) — columns: id, document_number, title, url, legal_type, legal_sectors, issuing_authority, issuance_date, signers
  - Content: 11 parquet files (~3.6GB tổng) — columns: id, content
  - **Phát hiện:** Metadata config KHÔNG có cột `content` → phải join với content files
- ✅ Tạo `scripts/utils/text_utils.py` — normalize Unicode NFC, strip HTML, collapse whitespace
- ✅ Tạo `scripts/utils/parquet_reader.py` — stream reader cho parquet files (hỗ trợ batching)
- ✅ Tạo `scripts/01_ingest.py` — main ingestion script
  - Join metadata + content streaming
  - Clean & normalize text
  - Filter docs < 50 chars
  - Extract metadata fields
  - Output: `processed/documents.jsonl`
- ✅ Test thành công với `--limit 100`:
  - 100 documents processed trong 20.3s
  - Avg content length: 24,850 chars
  - Doc types: Quyết định (71), Nghị quyết (12), Công văn (6), Lệnh (3), ...
  - Top issuers: Tỉnh Vĩnh Long (15), TP.HCM (10), ...
- ✅ Tạo các file tài liệu: `README.md`, `PROGRESS.md`, `ARCHITECTURE.md`, `plan.md`

### Quyết định thiết kế
- Bỏ OpenClaw (TypeScript agent framework) → dùng **FastAPI** làm orchestration
- Dùng `faiss-cpu` thay `faiss-gpu` ban đầu (dễ cài trên Windows, chuyển GPU sau)
- Content join strategy: streaming batches (5K rows/batch) để không đầy RAM

---

## Next Steps

### Phase 2: Parse Structure & Chunk
- [ ] Tạo `scripts/utils/legal_parser.py` — regex parse Chương/Mục/Điều/Khoản/Điểm
- [ ] Tạo `scripts/02_chunk.py` — chunk theo Khoản (hoặc Điều nếu không có Khoản)
- [ ] Test chunk quality trên 100 docs sample

### Phase 3: Build Indexes
- [ ] Cài `whoosh`, `underthesea`
- [ ] Tạo `scripts/03_index_bm25.py` — BM25 index with Vietnamese tokenization
- [ ] Tạo `scripts/04_index_vector.py` — FAISS vector index via Ollama embeddings

### Phase 4+: Retrieval, API, UI
- [ ] Hybrid retrieval engine (BM25 + Vector + RRF)
- [ ] FastAPI endpoints (/chat, /search, /health)
- [ ] Web UI (dark theme, streaming chat)
- [ ] Evaluation pipeline

## [2026-04-05 13:44] Phase 2 & Phase 3: Chunking + Indexing

### Phase 2: Parse Structure & Chunk ✅
- ✅ Tạo `scripts/utils/legal_parser.py`:
  - Regex patterns cho Phần/Chương/Mục/Điều/Khoản/Điểm
  - `split_into_articles()`: tách content thành danh sách Điều
  - `parse_clauses()`: tìm Khoản bên trong mỗi Điều
  - `parse_points()`: tìm Điểm bên trong mỗi Khoản
  - `build_breadcrumb()`: tạo path `Luật X > Chương Y > Điều Z > Khoản W`
  - `detect_footer_start()`: nhận diện phần ký tên cuối văn bản
- ✅ Tạo `scripts/02_chunk.py`:
  - Strategy: chunk theo Khoản, fallback sliding window cho docs không có cấu trúc
  - Mỗi chunk chứa context header (tiêu đề Điều)
  - Auto-split chunks quá dài (>1024 tokens)
  - Tạo `text_for_keyword` field cho BM25 search
- ✅ Test thành công `--limit 100`:
  - 100 docs → **3,762 chunks** trong 0.8s
  - 89/100 docs nhận diện cấu trúc Điều, 11 docs dùng sliding window
  - Avg 209 tokens/chunk
  - Token distribution: <50 (568), 50-200 (2333), 200-500 (541), 500-1024 (276), >1024 (44)

### Phase 3a: BM25 Index (Whoosh) ✅
- ✅ Cài `whoosh 2.7.4`, `faiss-cpu 1.13.2`, `httpx`
- ✅ Tạo `scripts/03_index_bm25.py`:
  - Custom VietnameseAnalyzer wrapper (underthesea)
  - Schema: chunk_id, doc_id, title, doc_type, issuer, content
  - Fallback simple tokenizer khi underthesea chưa cài
- ✅ Build index thành công: **3,762 chunks indexed** trong 6.0s
- ✅ Test queries đều cho kết quả relevant:
  - "quyền sử dụng đất" → 23 results (top: QĐ về đất đai)
  - "thuế thu nhập cá nhân" → 7 results (top: NQ phân cấp nguồn thu)
  - "xử phạt vi phạm hành chính" → 296 results (top: NĐ 125/2020)

### Phase 3b: Vector Index (FAISS) ✅ (script ready)
- ✅ Tạo `scripts/04_index_vector.py`:
  - Embed via Ollama API (`/api/embed`)
  - Batch processing (configurable batch_size)
  - FAISS Flat index (cosine similarity via L2 normalize + Inner Product)
  - Auto IVF cho >50K vectors
  - Lưu metadata mapping song song
  - **Chưa chạy được** vì Ollama chưa start → chạy khi `ollama serve` sẵn sàng

---

## Next Steps (cũ — đã hoàn thành)

> ~~Phase 3b, Phase 4, Phase 5, Phase 6~~ → ĐÃ HOÀN THÀNH ✅

## [2026-04-06 08:39] Phase 4, 5, 6: Retrieval + API + UI + Ngrok

### Phase 4: Hybrid Retrieval Engine ✅
- ✅ `src/retrieval/bm25_searcher.py` — BM25 search wrapper (Whoosh)
- ✅ `src/retrieval/vector_searcher.py` — Vector search wrapper (FAISS + Ollama embed)
- ✅ `src/retrieval/hybrid.py` — Hybrid engine:
  - RRF (Reciprocal Rank Fusion) kết hợp BM25 + Vector
  - Graceful fallback: nếu vector index chưa build → BM25-only mode
  - Dedup by chunk_id

### Phase 5: FastAPI Server + Ollama Integration ✅
- ✅ `src/llm/prompt_builder.py`:
  - System prompt cho legal assistant (tiếng Việt)
  - RAG prompt builder: system + context chunks + query + history
  - Sources summary builder cho UI citations
- ✅ `server.py` — Main server:
  - `GET /` — Serve frontend HTML
  - `GET /api/health` — Health check (retriever + Ollama status)
  - `POST /api/search` — Retrieval-only (không gọi LLM)
  - `POST /api/chat` — **SSE streaming** chat (retrieve → prompt → Ollama stream)
  - `POST /api/clear` — Clear session history
  - Session management (multi-session, history trim)
  - **Ngrok integration** (reuse pattern từ Meeting-trans)

### Phase 6: Web UI ✅
- ✅ `web/index.html` — ChatGPT-style dark theme:
  - Sidebar: nút "Cuộc trò chuyện mới", status indicator
  - Welcome screen: tiêu đề, mô tả, 4 example queries
  - Chat area: user/bot messages, markdown rendering
  - **SSE streaming**: token-by-token display
  - Sources/citations: expandable panel
  - Typing indicator animation
  - Responsive design (mobile-friendly)
  - Google Fonts (Inter)

### Test Results
- ✅ Server chạy thành công trên port 8899
- ✅ BM25 Retriever loaded: 3,762 docs
- ✅ UI hiển thị đẹp: dark theme, sidebar, welcome screen
- ✅ Health check: "Hệ thống sẵn sàng" (đèn xanh)
- ⚠️ Ollama chưa chạy nên chat chưa test end-to-end

---

## Hướng dẫn chạy

### Quick Start (BM25 only — không cần Ollama embedding)
```bash
# 1. Start Ollama (cho chat model)
ollama serve

# 2. Pull chat model (lần đầu)
ollama pull qwen2.5:7b-instruct

# 3. Start server
python server.py --port 8899

# 4. Mở browser: http://localhost:8899
```

### Full Hybrid (BM25 + Vector)
```bash
# Thêm bước build vector index
ollama pull qwen3-embedding:0.6b
python scripts/04_index_vector.py
```

### Demo qua Internet (Ngrok)
```bash
python server.py --port 8899 --ngrok
# URL sẽ hiển thị trên terminal
```

---

## Next Steps

### Tối ưu & Mở rộng
- [ ] Chạy full ingestion (`python scripts/01_ingest.py` — 518K docs)
- [ ] Chạy full chunking (`python scripts/02_chunk.py`)
- [ ] Rebuild BM25 index cho full dataset
- [ ] Build vector index (`python scripts/04_index_vector.py`)
- [ ] Test end-to-end với Ollama chat
- [ ] Cài `underthesea` để BM25 tokenize tiếng Việt tốt hơn
- [ ] Evaluation pipeline (precision, recall)
- [ ] Thêm favicon, loading states, better error handling
