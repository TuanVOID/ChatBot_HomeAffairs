# 🏗️ ARCHITECTURE — System Design

---

## Tổng Quan Các Layer

```
┌─────────────────────────────────────────┐
│          PRESENTATION LAYER             │
│  Web UI (HTML/CSS/JS) — Dark Theme      │
│  Chat bubbles, SSE streaming, Citations │
└────────────────┬────────────────────────┘
                 │ HTTP / SSE
┌────────────────▼────────────────────────┐
│            API LAYER                    │
│  FastAPI (Python 3.11)                  │
│  POST /api/chat — query + stream answer │
│  POST /api/search — retrieval only      │
│  GET  /api/documents — browse docs      │
│  GET  /api/health — system status       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          RETRIEVAL LAYER                │
│                                         │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ BM25 Search │  │  Vector Search   │  │
│  │ (Whoosh)    │  │  (FAISS)         │  │
│  └──────┬──────┘  └────────┬─────────┘  │
│         └────────┬─────────┘            │
│         ┌────────▼─────────┐            │
│         │  RRF Fusion      │            │
│         │  + Dedup + Filter│            │
│         └────────┬─────────┘            │
└──────────────────┼──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         LLM / INFERENCE LAYER           │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  Prompt Builder                 │    │
│  │  System + Context + Query       │    │
│  │  + Few-shot (from QA dataset)   │    │
│  └─────────────┬───────────────────┘    │
│                │                        │
│  ┌─────────────▼───────────────────┐    │
│  │  Ollama Server (localhost:11434)│    │
│  │  ├── Qwen2.5-7B-Instruct       │    │
│  │  └── Qwen3-Embedding-0.6B      │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           DATA LAYER                    │
│                                         │
│  Source:                                │
│  └── vietnamese-legal-documents/        │
│      ├── metadata/ (518K docs, 82MB)    │
│      └── content/  (11 files, 3.6GB)    │
│                                         │
│  Processed:                             │
│  ├── processed/documents.jsonl          │
│  ├── processed/chunks.jsonl             │
│  └── processed/manifest.json            │
│                                         │
│  Indexes:                               │
│  ├── indexes/bm25/    (Whoosh)          │
│  └── indexes/vector/  (FAISS + meta)    │
└─────────────────────────────────────────┘
```

---

## Data Pipeline Flow

```
[Parquet Files]
    │
    ▼  01_ingest.py
[Normalize + Join metadata & content]
    │
    ▼  Output: documents.jsonl
[Structured Documents with metadata]
    │
    ▼  02_chunk.py
[Parse Chương → Mục → Điều → Khoản → Điểm]
    │
    ▼  Output: chunks.jsonl
[Chunks with legal structure breadcrumb]
    │
    ├──────────────────────────┐
    ▼                          ▼
[03_index_bm25.py]       [04_index_vector.py]
    │                          │
    ▼                          ▼
[Whoosh BM25 Index]    [FAISS Vector Index]
    │                          │
    └────────────┬─────────────┘
                 ▼
        [Hybrid Search Ready]
```

---

## Query Flow (Runtime)

```
User query: "Quyền sử dụng đất theo Luật Đất đai?"
    │
    ▼
[Vietnamese word_tokenize (underthesea)]
    │
    ├─────────────────────┐
    ▼                     ▼
[BM25 Search]        [Vector Search]
  top-20               top-20
    │                     │
    └──────────┬──────────┘
               ▼
[Reciprocal Rank Fusion (k=60)]
               │
               ▼
[Dedup by Article + Metadata Filter]
               │
               ▼
         Top-10 chunks
               │
               ▼
[Prompt Builder]
  ┌─────────────────────────────┐
  │ System: "Bạn là trợ lý..." │
  │ Context: [top-10 chunks]    │
  │ Few-shot: [QA examples]     │
  │ Query: user question        │
  └─────────────┬───────────────┘
                │
                ▼
[Ollama API → Qwen2.5-7B-Instruct]
                │
                ▼ (streaming SSE)
[Response + Citations + Disclaimer]
```

---

## Auth Flow

```
Không có authentication — chạy localhost-only.
Nếu cần expose: thêm API key auth header middleware.

Client → FastAPI → No Auth (localhost)
Client → FastAPI → API-Key Header → Protected Routes (production)
```

---

## Key Design Patterns

### 1. Hybrid Retrieval (BM25 + Vector)
- **BM25**: Exact match — số hiệu VB, tên luật, từ khóa pháp lý
- **Vector**: Semantic — câu hỏi diễn đạt khác
- **RRF**: `score(d) = Σ 1/(k + rank(d))` — proven fusion method

### 2. Legal Structure Chunking
- Chunk unit = **Khoản** (clause) hoặc **Điều** (article) nếu không có Khoản
- Breadcrumb metadata: `Luật X > Chương Y > Điều Z > Khoản W`
- Cho phép citation chính xác đến từng Khoản/Điểm

### 3. Streaming Response
- FastAPI SSE (Server-Sent Events) 
- Token-by-token streaming từ Ollama
- Citations trả kèm cuối response

### 4. Config-driven
- Tất cả settings tập trung tại `config/settings.py`
- Đọc từ `.env` file
- Dễ thay đổi model, paths, hyperparameters

---

## Resource Allocation (RTX 5060 Ti 16GB)

| Component | VRAM | RAM | Ghi chú |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 5.5 GB | 1 GB | Chat model |
| Qwen3-Embedding-0.6B | 0.8 GB | 0.5 GB | Embedding |
| FAISS index | 2 GB | 4 GB | ~500K vectors |
| Whoosh BM25 | 0 | 2 GB | Pure CPU |
| FastAPI | 0 | 0.5 GB | Server |
| **Total** | **8.3 GB** | **8 GB** | Dư 7.7GB VRAM |
