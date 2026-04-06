# 🏛️ Legal RAG Chatbot — Vietnamese Legal Document Assistant

> Chatbot tra cứu pháp luật Việt Nam chạy hoàn toàn local, sử dụng Retrieval-Augmented Generation (RAG).

---

## 🎯 Mục tiêu

Hệ thống chatbot cho phép người dùng **hỏi đáp về pháp luật Việt Nam** bằng tiếng Việt. Hệ thống tìm kiếm trong **518K+ văn bản pháp luật** và trả lời kèm **trích dẫn chính xác** (tên văn bản, Điều, Khoản, Điểm).

**Tất cả chạy 100% local** — không gửi dữ liệu ra bên ngoài.

---

## 🏗️ Kiến trúc Tổng Thể

```
User (Browser)
    ↓
Web UI (HTML/CSS/JS)
    ↓  HTTP/SSE
FastAPI Server (Python)
    ├── Hybrid Retrieval Engine
    │   ├── BM25 Search (Whoosh)
    │   └── Vector Search (FAISS + Qwen3-Embedding-0.6B)
    │   └── RRF Score Fusion
    ├── Prompt Builder (Context + Query + Few-shot)
    └── Ollama Client
         └── Qwen2.5-7B-Instruct (Local LLM)

Data Layer:
    ├── 518K legal documents (Parquet → JSONL)
    ├── BM25 Index (Whoosh)
    └── Vector Index (FAISS)
```

---

## 🔧 Tech Stack

| Layer | Công nghệ |
|---|---|
| **LLM** | Qwen2.5-7B-Instruct (via Ollama, Q4_K_M) |
| **Embedding** | Qwen3-Embedding-0.6B (via Ollama) |
| **Model Server** | Ollama (local) |
| **Keyword Search** | Whoosh (BM25) |
| **Vector Search** | FAISS (cpu/gpu) |
| **API** | FastAPI + SSE streaming |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Data** | Parquet → JSONL, Whoosh Index, FAISS Index |
| **NLP** | underthesea (Vietnamese tokenizer) |

---

## 🔄 Flow Chính

### 1. Data Pipeline (Offline)
```
Parquet files (518K docs, ~3.6GB)
  → 01_ingest.py: normalize, clean → documents.jsonl
  → 02_chunk.py: parse legal structure → chunks.jsonl
  → 03_index_bm25.py: build BM25 index
  → 04_index_vector.py: embed + build FAISS index
```

### 2. Query Flow (Online)
```
User Query
  → Vietnamese tokenize
  → Parallel: BM25 search + Vector search
  → RRF fusion (Reciprocal Rank Fusion)
  → Top-K chunks with metadata
  → Prompt Builder (system + context + query + few-shot)
  → Qwen2.5-7B via Ollama (streaming)
  → Response with citations + disclaimer
```

---

## 🖥️ Hardware Target

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 3900X (12C/24T) |
| RAM | 32 GB DDR4-3200 |
| GPU | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| VRAM usage | ~8.3 GB / 16 GB |

---

## 📁 Cấu Trúc Thư Mục

```
ChatBot2_Opus/
├── config/             # Centralized settings
├── scripts/            # Data pipeline scripts (01-05)
│   └── utils/          # Parsers, text utils
├── src/                # Application source code
│   ├── retrieval/      # BM25 + Vector + Hybrid
│   ├── llm/            # Ollama client, prompt builder
│   ├── api/            # FastAPI routes
│   └── embeddings/     # Embedding service
├── web/                # Frontend (HTML/CSS/JS)
├── processed/          # Generated JSONL files
├── indexes/            # BM25 + FAISS indexes
├── logs/               # Application logs
├── plan.md             # Detailed implementation plan
├── PROGRESS.md         # Development log
├── ARCHITECTURE.md     # System architecture
└── requirements.txt    # Python dependencies
```

---

## ⚠️ Lưu Ý

- Chatbot **KHÔNG** thay thế tư vấn pháp lý chuyên nghiệp
- QA dataset chỉ dùng cho **evaluation** và **few-shot**, không phải nguồn retrieval
- Cần test kỹ trước khi sử dụng — pháp luật yêu cầu độ chính xác cao
