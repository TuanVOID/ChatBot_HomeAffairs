# 📋 Legal RAG Chatbot – Kế Hoạch Triển Khai Chi Tiết

> **Dự án:** Vietnamese Legal RAG Chatbot  
> **Mục tiêu:** Chatbot tra cứu luật Việt Nam chạy hoàn toàn local  
> **Ngày lập:** 2026-04-05  
> **Hardware:** Ryzen 9 3900x · 32GB DDR4-3200 · RTX 5060 Ti 16GB VRAM

---

## 📐 Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────┐
│                    🖥️ Web UI (HTML/JS)                   │
│            Chat Interface + Citation Display             │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────┐
│              🔧 API Layer (FastAPI - Python)             │
│         Orchestration · Session · Prompt Builder         │
│                                                         │
│  ┌──────────────┐    ┌─────────────────────────────┐    │
│  │  /api/chat   │    │  /api/search (debug)        │    │
│  │  /api/health │    │  /api/documents (browse)    │    │
│  └──────┬───────┘    └─────────────┬───────────────┘    │
│         │                          │                     │
│  ┌──────▼──────────────────────────▼───────────────┐    │
│  │         🔍 Hybrid Retrieval Engine               │    │
│  │                                                  │    │
│  │  ┌─────────────┐     ┌────────────────────┐     │    │
│  │  │ BM25 Search │     │ Vector Search      │     │    │
│  │  │ (Whoosh)    │     │ (FAISS + Qwen3     │     │    │
│  │  │             │     │  Embedding-0.6B)   │     │    │
│  │  └──────┬──────┘     └─────────┬──────────┘     │    │
│  │         │                      │                 │    │
│  │         └──────────┬───────────┘                 │    │
│  │                    │                             │    │
│  │         ┌──────────▼──────────┐                  │    │
│  │         │  RRF / Score Fusion │                  │    │
│  │         │  + Re-ranking       │                  │    │
│  │         └──────────┬──────────┘                  │    │
│  └────────────────────┼────────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────▼────────────────────────────┐    │
│  │         🧠 Prompt Builder                        │    │
│  │  System Prompt + Retrieved Context + Query       │    │
│  │  + Few-shot Examples (from QA dataset)           │    │
│  └────────────────────┬────────────────────────────┘    │
└───────────────────────┼─────────────────────────────────┘
                        │ HTTP (localhost:11434)
┌───────────────────────▼─────────────────────────────────┐
│              🤖 Ollama Server (Local)                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Qwen2.5-7B-Instruct (Q4_K_M / Q5_K_M)         │   │
│  │  ~5-7 GB VRAM · Context: 8192 tokens            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Qwen3-Embedding-0.6B (embedding model)         │   │
│  │  ~0.6 GB VRAM · dim: 1024                       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              💾 Data Layer                               │
│                                                         │
│  📁 data/                                               │
│  ├── vietnamese-legal-documents/  (518K docs, ~3.6GB)   │
│  │   ├── content/   (11 parquet files)                  │
│  │   └── metadata/  (1 parquet file, ~82MB)             │
│  ├── vietnamese-legal-qa/         (9.7K QA pairs)       │
│  │   └── data/train-*.parquet                           │
│  └── data-luat/                   (data_tong.json)      │
│                                                         │
│  📁 indexes/                                            │
│  ├── bm25/          (Whoosh index)                      │
│  └── vector/        (FAISS index + metadata)            │
│                                                         │
│  📁 processed/                                          │
│  ├── documents.jsonl (normalized docs)                  │
│  ├── chunks.jsonl    (structured chunks)                │
│  └── manifest.json   (processing metadata)              │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```
f:\SpeechToText-indti\ChatBot2_Opus\
│
├── plan.md                          # ← BẠN ĐANG ĐỌC FILE NÀY
│
├── config/
│   ├── settings.py                  # Tất cả config tập trung
│   └── prompts.py                   # System prompts, templates
│
├── scripts/
│   ├── 01_ingest.py                 # Phase 1: Đọc parquet → normalize
│   ├── 02_chunk.py                  # Phase 2: Parse structure → chunk
│   ├── 03_index_bm25.py            # Phase 3a: Build BM25 index
│   ├── 04_index_vector.py          # Phase 3b: Build FAISS vector index
│   ├── 05_eval.py                   # Phase 7: Evaluation pipeline
│   └── utils/
│       ├── legal_parser.py          # Parse chương/điều/khoản/điểm
│       ├── text_utils.py            # Normalize Unicode, clean text
│       └── parquet_reader.py        # Đọc parquet files
│
├── src/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_searcher.py         # BM25 keyword search
│   │   ├── vector_searcher.py       # FAISS vector search
│   │   ├── hybrid.py                # Hybrid retrieval + RRF
│   │   └── reranker.py              # Optional cross-encoder rerank
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py         # Ollama API wrapper
│   │   ├── prompt_builder.py        # Context → prompt assembly
│   │   └── response_parser.py       # Parse citations, warnings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── chat.py              # POST /api/chat
│   │   │   ├── search.py            # POST /api/search
│   │   │   ├── documents.py         # GET /api/documents
│   │   │   └── health.py            # GET /api/health
│   │   └── models.py                # Pydantic request/response
│   │
│   └── embeddings/
│       ├── __init__.py
│       └── embedding_service.py     # Ollama embedding wrapper
│
├── web/
│   ├── index.html                   # Main chat UI
│   ├── css/
│   │   └── style.css                # Premium dark theme
│   └── js/
│       └── app.js                   # Chat logic, streaming
│
├── data/                            # Symlink → ../ChatBot/data/
│   └── (sử dụng data từ workspace ChatBot/data)
│
├── indexes/                         # Generated indexes
│   ├── bm25/
│   └── vector/
│
├── processed/                       # Processed JSONL files
│   ├── documents.jsonl
│   ├── chunks.jsonl
│   └── manifest.json
│
├── logs/                            # Application logs
│
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── run.py                           # One-click start script
```

---

## 🔧 Technology Stack

| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| **Chat Model** | Qwen2.5-7B-Instruct (via Ollama) | Hỗ trợ tiếng Việt tốt, chạy vừa 16GB VRAM |
| **Embedding Model** | Qwen3-Embedding-0.6B (via Ollama) | Nhẹ (~0.6GB), multilingual, dim=1024 |
| **Model Server** | Ollama | Đơn giản, local, hỗ trợ GGUF quantized |
| **Keyword Search** | Whoosh (Python) | Pure Python, không cần external service |
| **Vector Search** | FAISS (faiss-gpu) | GPU-accelerated, proven at scale |
| **API Framework** | FastAPI | Async, fast, auto-docs, streaming SSE |
| **Web UI** | Vanilla HTML/CSS/JS | Lightweight, no build step needed |
| **Data Format** | JSONL + Parquet | Streaming-friendly, standard formats |
| **Orchestration** | FastAPI (thay vì OpenClaw) | Xem ghi chú bên dưới |

### Về OpenClaw

OpenClaw là một **TypeScript-based autonomous AI agent framework** chuyên kết nối qua messaging platforms (WhatsApp, Telegram, Discord...). Nó **không phải** một Python chatbot/RAG framework.

Trong kế hoạch này, tôi đề xuất dùng **FastAPI (Python)** làm lớp orchestration chính vì:
1. Toàn bộ data pipeline (parquet → chunking → indexing) đều viết bằng Python
2. FAISS, Whoosh, transformers đều là Python libraries
3. Ollama có Python SDK chính thức (`ollama` package)
4. Dễ integrate và debug hơn so với TypeScript ↔ Python bridge

Nếu bạn **vẫn muốn dùng OpenClaw**, ta có thể integrate bằng cách:
- Chạy FastAPI server như một "tool/skill" mà OpenClaw gọi
- OpenClaw xử lý messaging channel, FastAPI xử lý RAG pipeline
- Cần cài thêm Node.js + OpenClaw setup riêng

---

## 📊 Ước Tính Tài Nguyên (RTX 5060 Ti 16GB)

| Thành phần | VRAM | RAM | Disk |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M | ~5.5 GB | ~1 GB | ~4.5 GB |
| Qwen3-Embedding-0.6B | ~0.8 GB | ~0.5 GB | ~1.2 GB |
| FAISS index (ước tính 500K chunks) | ~2 GB | ~4 GB | ~2 GB |
| Whoosh BM25 index | 0 | ~2 GB | ~500 MB |
| FastAPI server | 0 | ~500 MB | - |
| **Tổng** | **~8.3 GB / 16 GB** | **~8 GB / 32 GB** | **~8 GB** |

> Còn dư ~7.7GB VRAM → có thể nâng lên Q5_K_M hoặc thậm chí Q6_K cho chất lượng tốt hơn.
> Hoặc dùng context window lớn hơn (16K tokens) nếu cần.

---

## 🚀 Các Phase Triển Khai

### Phase 0: Setup Môi Trường
**Thời gian ước tính:** 30 phút

| # | Công việc | Chi tiết |
|---|---|---|
| 0.1 | Cài Ollama | Download từ ollama.com, verify GPU detection |
| 0.2 | Pull models | `ollama pull qwen2.5:7b-instruct` + embedding model |
| 0.3 | Tạo Python venv | Python 3.11+, venv trong ChatBot2_Opus |
| 0.4 | Cài dependencies | `pip install -r requirements.txt` |
| 0.5 | Symlink data | Link `data/` → `../ChatBot/data/` |

**requirements.txt:**
```
# Core
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-dotenv>=1.0.0
pydantic>=2.6.0

# Data Processing
pandas>=2.2.0
pyarrow>=15.0.0

# Search & Retrieval
whoosh>=2.7.4
faiss-gpu>=1.7.4
numpy>=1.26.0

# LLM Integration
ollama>=0.4.0
httpx>=0.27.0

# Text Processing
underthesea>=6.8.0
regex>=2024.0.0

# Evaluation
rouge-score>=0.1.2
scikit-learn>=1.4.0

# Logging
loguru>=0.7.0
```

---

### Phase 1: Ingestion & Normalize
**Thời gian ước tính:** 2-3 giờ coding, ~30 phút chạy  
**Script:** `scripts/01_ingest.py`

| # | Công việc | Chi tiết |
|---|---|---|
| 1.1 | Đọc metadata parquet | Load 518K records từ `metadata/data-*.parquet` |
| 1.2 | Đọc content parquet | Stream load từ `content/data-*.parquet` (~3.6GB) |
| 1.3 | Join metadata + content | Merge trên `id` column |
| 1.4 | Normalize text | Unicode NFC, loại HTML tags, fix whitespace |
| 1.5 | Filter documents | Loại docs quá ngắn (<50 chars), trùng lặp |
| 1.6 | Extract metadata | `legal_type`, `issuing_authority`, `issuance_date`, `legal_sectors` |
| 1.7 | Xuất documents.jsonl | Mỗi doc 1 dòng JSON với full metadata |

**Output schema (documents.jsonl):**
```json
{
  "doc_id": "doc_123456",
  "title": "Luật Đất đai số 31/2024/QH15",
  "document_number": "31/2024/QH15",
  "doc_type": "Luật",
  "issuer": "Quốc hội",
  "issue_date": "18/01/2024",
  "legal_sectors": ["Đất đai", "Bất động sản"],
  "status": "active",
  "content": "Chương I. NHỮNG QUY ĐỊNH CHUNG\nĐiều 1. Phạm vi điều chỉnh\n..."
}
```

**Nguồn dữ liệu chính:** `vietnamese-legal-documents` (518K docs) - đây là **nguồn chân lý**.  
**QA dataset:** `vietnamese-legal-qa` (9.7K) - chỉ dùng cho **few-shot examples** và **evaluation**, KHÔNG phải nguồn retrieval.

---

### Phase 2: Parse Structure & Chunk
**Thời gian ước tính:** 3-4 giờ coding, ~1 giờ chạy  
**Script:** `scripts/02_chunk.py`

| # | Công việc | Chi tiết |
|---|---|---|
| 2.1 | Legal structure parser | Regex/rules parse Chương → Mục → Điều → Khoản → Điểm |
| 2.2 | Chunk strategy | Mỗi chunk = 1 Khoản (hoặc 1 Điều nếu ko có Khoản) |
| 2.3 | Context window | Chunk size 512-1024 tokens, overlap context (Điều header) |
| 2.4 | Metadata enrichment | Gắn breadcrumb: `Luật X > Chương Y > Điều Z > Khoản W` |
| 2.5 | text_for_keyword | Tạo field riêng cho BM25: title + path + text |
| 2.6 | Xuất chunks.jsonl | Mỗi chunk 1 dòng với full metadata |

**Chunk schema (chunks.jsonl):**
```json
{
  "chunk_id": "doc_123456_d1_k2",
  "doc_id": "doc_123456",
  "title": "Luật Đất đai số 31/2024/QH15",
  "document_number": "31/2024/QH15",
  "doc_type": "Luật",
  "issuer": "Quốc hội",
  "issue_date": "18/01/2024",
  "chapter": "Chương I",
  "section": null,
  "article": "Điều 1",
  "clause": "Khoản 2",
  "point": null,
  "path": "Chương I > Điều 1 > Khoản 2",
  "text": "Điều 1\n2. Luật này áp dụng đối với...",
  "text_for_keyword": "Luật Đất đai 31/2024/QH15 Điều 1 Khoản 2 ...",
  "token_count": 245
}
```

**Chiến lược parse cấu trúc pháp luật:**
```
Pattern Detection:
  Chương [I-XX]:    ^Chương\s+[IVXLCDM]+[\.\s]
  Mục [1-99]:       ^Mục\s+\d+[\.\s]
  Điều [1-999]:     ^Điều\s+\d+[\.\s]
  Khoản [1-99]:     ^\d+\.\s
  Điểm [a-z]:       ^[a-zđ]\)\s
```

---

### Phase 3a: Build BM25 Keyword Index
**Thời gian ước tính:** 1-2 giờ coding, ~20 phút chạy  
**Script:** `scripts/03_index_bm25.py`

| # | Công việc | Chi tiết |
|---|---|---|
| 3a.1 | Define Whoosh schema | Fields: chunk_id, text_for_keyword, doc_type, issuer |
| 3a.2 | Vietnamese tokenizer | Dùng `underthesea.word_tokenize` cho BM25 |
| 3a.3 | Build index | Index toàn bộ chunks.jsonl vào Whoosh |
| 3a.4 | Test queries | Verify search quality với vài câu mẫu |

---

### Phase 3b: Build Vector Index (FAISS)
**Thời gian ước tính:** 2-3 giờ coding, ~2-4 giờ chạy (embedding 500K chunks)  
**Script:** `scripts/04_index_vector.py`

| # | Công việc | Chi tiết |
|---|---|---|
| 3b.1 | Embedding service | Wrap Ollama embedding API cho Qwen3-Embedding-0.6B |
| 3b.2 | Batch embedding | Embed chunks theo batch (batch_size=64) |
| 3b.3 | FAISS index | Build IVF index (nlist=2048), train + add |
| 3b.4 | Save metadata | Lưu chunk_id mapping song song với FAISS index |
| 3b.5 | Test similarity | Verify embedding quality |

> **Lưu ý:** Embedding 500K chunks sẽ mất 2-4 giờ trên GPU. Nên triển khai **incremental indexing** để không phải re-embed khi thêm data mới. Có thể bắt đầu với subset nhỏ (50K docs) để test trước.

---

### Phase 4: Hybrid Retrieval Engine
**Thời gian ước tính:** 3-4 giờ

| # | Công việc | Chi tiết |
|---|---|---|
| 4.1 | BM25 searcher | Wrapper class cho Whoosh query |
| 4.2 | Vector searcher | Wrapper class cho FAISS search |
| 4.3 | Score fusion (RRF) | Reciprocal Rank Fusion: `1/(k+rank)` |
| 4.4 | Metadata filter | Filter theo `doc_type`, `issuer`, `date_range` |
| 4.5 | Result dedup | Loại duplicate chunks cùng Điều |
| 4.6 | Top-K selection | Trả về top 5-10 chunks + scores |

**Hybrid Search Flow:**
```python
def hybrid_search(query: str, top_k: int = 10, alpha: float = 0.5):
    # 1. BM25 search
    bm25_results = bm25_searcher.search(query, top_k=top_k*2)
    
    # 2. Vector search  
    query_embedding = embed(query)
    vector_results = vector_searcher.search(query_embedding, top_k=top_k*2)
    
    # 3. RRF fusion
    fused = reciprocal_rank_fusion(
        bm25_results, vector_results, 
        k=60, alpha=alpha
    )
    
    # 4. Dedup + filter
    results = dedup_by_article(fused)
    
    return results[:top_k]
```

---

### Phase 5: API Layer (FastAPI)
**Thời gian ước tính:** 4-5 giờ

| # | Công việc | API | Chi tiết |
|---|---|---|---|
| 5.1 | Chat endpoint | `POST /api/chat` | Nhận query → retrieve → generate |
| 5.2 | Streaming | SSE via `/api/chat` | Stream tokens real-time |
| 5.3 | Search debug | `POST /api/search` | Chỉ retrieval, không generate |
| 5.4 | Document browse | `GET /api/documents` | Xem documents theo filter |
| 5.5 | Health check | `GET /api/health` | Check Ollama + indexes status |
| 5.6 | Session memory | In-memory store | Lưu conversation history (last 5 turns) |

**Chat Request/Response:**
```json
// Request
{
  "query": "Quyền của người sử dụng đất theo Luật Đất đai?",
  "session_id": "abc123",
  "filters": {
    "doc_type": "Luật",
    "date_from": "2024-01-01"
  }
}

// Response (streamed)
{
  "answer": "Theo Điều 27 Luật Đất đai 2024...",
  "citations": [
    {
      "chunk_id": "doc_123_d27_k1",
      "title": "Luật Đất đai 2024",
      "article": "Điều 27",
      "clause": "Khoản 1",
      "text": "Người sử dụng đất có các quyền...",
      "score": 0.87
    }
  ],
  "warning": "Đây chỉ là thông tin tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.",
  "metadata": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 1200,
    "model": "qwen2.5:7b-instruct"
  }
}
```

---

### Phase 6: Prompt Engineering
**Thời gian ước tính:** 2-3 giờ (iterative)

**System Prompt:**
```
Bạn là trợ lý pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi 
dựa CHÍNH XÁC vào các văn bản pháp luật được cung cấp bên dưới.

## Quy tắc:
1. CHỈ trả lời dựa trên nội dung trong phần [TÀI LIỆU THAM KHẢO]
2. LUÔN trích dẫn nguồn: tên văn bản, số hiệu, Điều/Khoản/Điểm cụ thể
3. Nếu thông tin không có trong tài liệu → nói rõ "Không tìm thấy thông tin"
4. KHÔNG tự suy luận hoặc bịa thông tin pháp luật
5. Kết thúc bằng cảnh báo: đây chỉ là tham khảo, không thay thế tư vấn chuyên nghiệp
6. Trả lời bằng tiếng Việt

## Định dạng trả lời:
- Trả lời ngắn gọn, rõ ràng
- Dùng bullet points cho nhiều điểm
- Trích dẫn: [Điều X, Khoản Y - Tên văn bản (Số hiệu)]
```

**Few-shot từ QA dataset:**
```
Chọn 2-3 QA pairs từ vietnamese-legal-qa dataset có:
- question_type = "factual" hoặc "application"
- difficulty = "medium"
- Liên quan đến loại câu hỏi đang được hỏi
```

---

### Phase 7: Web UI
**Thời gian ước tính:** 4-5 giờ

| # | Công việc | Chi tiết |
|---|---|---|
| 7.1 | Chat interface | Dark theme, glassmorphism, chat bubbles |
| 7.2 | Message streaming | SSE streaming hiển thị từng token |
| 7.3 | Citation panel | Sidebar hiển thị nguồn trích dẫn |
| 7.4 | Filter controls | Lọc theo loại VB, cơ quan ban hành, năm |
| 7.5 | History sidebar | Lịch sử hội thoại |
| 7.6 | Responsive design | Desktop + Tablet + Mobile |
| 7.7 | Loading states | Skeleton loading, typing indicator |

**Design specs:**
- Bảng màu: Dark mode (#0a0a1a nền, #1e1e3f cards, #7c3aed accent)
- Font: Inter (Google Fonts)
- Micro-animations: fade-in messages, slide citations
- Glassmorphism cards cho citation blocks

---

### Phase 8: Evaluation Pipeline
**Thời gian ước tính:** 2-3 giờ  
**Script:** `scripts/05_eval.py`

| # | Metric | Công cụ | Mục tiêu |
|---|---|---|---|
| 8.1 | Retrieval Recall@10 | QA dataset questions | > 70% |
| 8.2 | Retrieval MRR | QA dataset questions | > 0.5 |
| 8.3 | Answer Faithfulness | LLM-as-judge (Qwen) | > 80% |
| 8.4 | Citation Accuracy | Exact match chunk_id | > 85% |
| 8.5 | Latency P95 | Timer wrapper | < 5s end-to-end |

**Eval Flow:**
```
QA Dataset (9.7K pairs)
    → Sample 200 questions
    → For each question:
        1. Run hybrid retrieval
        2. Check if gold chunk in top-10 (Recall)
        3. Generate answer with Qwen
        4. Compare answer vs gold answer (ROUGE-L)
        5. Check citation accuracy
    → Aggregate metrics
    → Report
```

---

## ⏱️ Timeline Tổng Quan

| Phase | Thời gian | Ngày (ước tính) |
|---|---|---|
| Phase 0: Setup | 0.5 ngày | Ngày 1 |
| Phase 1: Ingestion | 1-2 ngày | Ngày 1-2 |
| Phase 2: Chunking | 2 ngày | Ngày 3-4 |
| Phase 3: Indexing | 2 ngày | Ngày 5-6 |
| Phase 4: Retrieval | 2 ngày | Ngày 7-8 |
| Phase 5: API | 2-3 ngày | Ngày 9-11 |
| Phase 6: Prompts | 1-2 ngày | Ngày 12-13 |
| Phase 7: Web UI | 2-3 ngày | Ngày 12-14 |
| Phase 8: Eval | 1-2 ngày | Ngày 14-15 |
| **Tổng** | **~15 ngày** | |

---

## 🔑 Quyết Định Thiết Kế Quan Trọng

### 1. Tại sao Hybrid Retrieval (BM25 + Vector)?
- **BM25**: Tốt cho exact match (số hiệu văn bản, tên luật cụ thể)
- **Vector**: Tốt cho semantic similarity (câu hỏi diễn đạt khác)
- **RRF fusion**: Kết hợp ưu điểm cả hai, proven effectiveness

### 2. Tại sao chunk theo cấu trúc pháp luật?
- Văn bản luật có cấu trúc rõ ràng (Chương → Điều → Khoản)
- Chunk theo Khoản đảm bảo mỗi chunk là 1 đơn vị ngữ nghĩa hoàn chỉnh
- Metadata (breadcrumb path) giúp citation chính xác

### 3. Tại sao Qwen2.5-7B thay vì model lớn hơn?
- 7B + Q4_K_M chỉ dùng ~5.5GB VRAM, còn dư cho embedding + FAISS
- Qwen2.5 có hỗ trợ tiếng Việt tốt nhất trong phân khúc 7B
- Có thể nâng lên 14B nếu chuyển embedding sang CPU

### 4. Tại sao FastAPI thay vì OpenClaw?
- OpenClaw là TypeScript agent framework, không phải RAG framework
- Toàn bộ stack xử lý data đều Python → tránh language bridge complexity
- FastAPI có built-in SSE streaming cho chat experience mượt mà
- Có thể thêm OpenClaw sau như messaging gateway layer

---

## 📋 Checklist Trước Khi Bắt Đầu

- [ ] Ollama đã cài và chạy (`ollama serve`)
- [ ] GPU được detect (`ollama run qwen2.5:7b-instruct "Xin chào"`)
- [ ] Python 3.11+ đã cài
- [ ] Có đủ disk space (~15GB cho indexes + models)
- [ ] Data đã có trong `f:\SpeechToText-indti\ChatBot\data\`
- [ ] Xác nhận dùng FastAPI (không dùng OpenClaw) hay cần integrate OpenClaw

---

## ⚠️ Lưu ý quan trọng

1. **Không dùng QA dataset làm nguồn retrieval** — QA dataset chỉ để eval và few-shot
2. **Luôn có disclaimer** — Chatbot KHÔNG thay thế tư vấn pháp lý chuyên nghiệp
3. **Cần test kỹ** — Pháp luật yêu cầu độ chính xác cao, cần eval kỹ trước khi dùng
4. **Data có thể outdated** — 518K docs cover 1924-2026, cần mechanism cập nhật
