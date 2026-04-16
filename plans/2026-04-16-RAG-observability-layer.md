RAG observability layer
-----------------------------------------------
Mục tiêu
Lớp này phải trả lời được 5 câu hỏi sau cho mỗi query:

User hỏi gì?
BM25 trả về gì, vector trả về gì?
RRF/dedup/filter đã giữ và loại cái gì?
Prompt cuối cùng dùng context nào?
Answer sai là do retrieval, prompt, hay model?
Nguyên tắc

Không thay đổi pipeline cốt lõi.
Chỉ thêm:

trace_id cho mỗi request
structured logs cho từng stage
lưu retrieval snapshot
lưu prompt/answer/citation
API debug để xem lại từng trace
bộ eval chạy hàng loạt
Kiến trúc mới

Bạn giữ nguyên kiến trúc hiện tại, chỉ thêm một lớp cắt ngang:

┌─────────────────────────────────────────┐
│          PRESENTATION LAYER             │
│  Web UI + Debug UI                      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│            API LAYER                    │
│  FastAPI                                │
│  /api/chat                              │
│  /api/search                            │
│  /api/debug/traces/{trace_id}           │
│  /api/debug/search/{trace_id}           │
│  /api/evals/run                         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│       OBSERVABILITY LAYER (NEW)         │
│  Trace Middleware                       │
│  Structured Logger                      │
│  Retrieval Snapshot Recorder            │
│  Prompt/Response Recorder               │
│  Metrics Aggregator                     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          RETRIEVAL + LLM LAYER          │
│  underthesea → BM25 → Vector → RRF      │
│  Dedup/Filter → Prompt → Ollama         │
└─────────────────────────────────────────┘

Bạn cần thêm những gì
1) Trace ID middleware

Mỗi request tạo một trace_id.

Ví dụ:

req_20260416_ab12cd34
đi theo toàn bộ log, retrieval, answer
FastAPI middleware
# app/middleware/trace.py
import time
import uuid
from fastapi import Request

async def trace_middleware(request: Request, call_next):
    trace_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.trace_id = trace_id
    request.state.started_at = time.time()

    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response

Gắn vào app:

app.middleware("http")(trace_middleware)
2) Structured logging JSON

Không log kiểu text lung tung nữa.
Mỗi event là 1 JSON record.

Ví dụ:

{
  "trace_id": "req_ab12cd34",
  "stage": "bm25_search",
  "timestamp": "2026-04-16T10:21:15.123Z",
  "query_raw": "quyền sử dụng đất là gì",
  "query_tokenized": "quyền_sử_dụng_đất là gì",
  "top_k": 20,
  "hits": [
    {"chunk_id": "LAW123_C45", "score": 12.44, "rank": 1},
    {"chunk_id": "LAW999_C18", "score": 11.87, "rank": 2}
  ],
  "latency_ms": 23
}
Log stages tối thiểu
request_received
query_normalized
bm25_search
vector_search
rrf_fusion
dedup_filter
prompt_built
llm_started
llm_finished
response_sent
error
3) Retrieval snapshot recorder

Đây là phần quan trọng nhất.

Bạn phải lưu được:

query gốc
query sau tokenize
BM25 top-k
vector top-k
sau RRF top-k
chunk nào bị loại và vì sao
top chunks cuối được nhét vào prompt
Schema Python
from pydantic import BaseModel
from typing import List, Optional

class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    article_ref: Optional[str] = None
    score: float
    rank: int
    source: str  # bm25 | vector | rrf
    breadcrumb: Optional[str] = None

class FilteredHit(BaseModel):
    chunk_id: str
    reason: str  # duplicate_article | metadata_filter | too_long | low_score

class RetrievalSnapshot(BaseModel):
    trace_id: str
    query_raw: str
    query_tokenized: Optional[str] = None
    bm25_hits: List[SearchHit]
    vector_hits: List[SearchHit]
    rrf_hits: List[SearchHit]
    filtered_out: List[FilteredHit]
    final_context_chunk_ids: List[str]
4) Prompt registry

Nếu answer sai mà bạn không biết prompt version nào đã được dùng thì coi như debug mù.

Bạn cần version hóa:

system prompt
answer format instruction
citation instruction
few-shot pack
retrieval params
Ví dụ metadata lưu kèm
{
  "trace_id": "req_ab12cd34",
  "prompt_version": "legal_v3",
  "fewshot_version": "qa_pack_20260410",
  "retrieval_config_version": "hybrid_v2",
  "model_name": "qwen2.5-7b-instruct",
  "temperature": 0.2,
  "top_k_context": 10
}
5) Response recorder

Lưu:

answer text
citation list
grounding flag sơ bộ
latency tổng
token count nếu lấy được
error nếu có
class ResponseRecord(BaseModel):
    trace_id: str
    answer_text: str
    citations: list[str]
    grounded_flag: bool | None = None
    total_latency_ms: int
    llm_latency_ms: int
    error_message: str | None = None

DB dùng gì 
Schema DB đề xuất
Bảng trace_runs
CREATE TABLE trace_runs (
    trace_id TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    user_query TEXT NOT NULL,
    query_tokenized TEXT,
    prompt_version TEXT,
    retrieval_config_version TEXT,
    model_name TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,         -- success | error
    total_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    answer_text TEXT,
    citations_json TEXT,
    error_message TEXT
);

Bảng trace_events
CREATE TABLE trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
Bảng retrieval_hits
CREATE TABLE retrieval_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    stage TEXT NOT NULL,          -- bm25 | vector | rrf | final
    chunk_id TEXT NOT NULL,
    doc_id TEXT,
    title TEXT,
    article_ref TEXT,
    breadcrumb TEXT,
    score REAL,
    rank INTEGER,
    kept INTEGER NOT NULL DEFAULT 1,
    filter_reason TEXT
);
Bảng eval_cases
CREATE TABLE eval_cases (
    case_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    expected_doc_id TEXT,
    expected_article_ref TEXT,
    expected_keywords TEXT,
    gold_answer TEXT
);
Bảng eval_results
CREATE TABLE eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    trace_id TEXT,
    hit_expected_source INTEGER,
    hit_expected_article INTEGER,
    citation_correct INTEGER,
    grounded INTEGER,
    completeness_score INTEGER,
    hallucination INTEGER,
    total_latency_ms INTEGER,
    notes TEXT
);
File structure nên thêm
app/
  api/
    chat.py
    search.py
    debug.py
    evals.py

  middleware/
    trace.py

  observability/
    logger.py
    recorder.py
    models.py
    metrics.py
    storage.py

  services/
    retrieval_service.py
    prompt_service.py
    llm_service.py

  repositories/
    trace_repo.py
    eval_repo.py

***Cách gắn vào pipeline hiện tại

Bạn đang có flow:
query -> tokenize -> bm25 + vector -> rrf -> dedup/filter -> prompt -> ollama -> answer

Ta chỉ cần bọc từng stage.

Ví dụ service retrieval
class RetrievalService:
    def __init__(self, bm25_engine, vector_engine, recorder):
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        self.recorder = recorder

    def search(self, trace_id: str, query_raw: str, query_tokenized: str):
        bm25_hits = self.bm25_engine.search(query_tokenized, top_k=20)
        self.recorder.record_hits(trace_id, "bm25", bm25_hits)

        vector_hits = self.vector_engine.search(query_raw, top_k=20)
        self.recorder.record_hits(trace_id, "vector", vector_hits)

        rrf_hits = fuse_rrf(bm25_hits, vector_hits, k=60)
        self.recorder.record_hits(trace_id, "rrf", rrf_hits)

        final_hits, filtered = dedup_and_filter(rrf_hits)
        self.recorder.record_filtered(trace_id, filtered)
        self.recorder.record_hits(trace_id, "final", final_hits)

        return final_hits
API mới cần có
1) GET /api/debug/traces/{trace_id}

Trả toàn bộ:

query
events
retrieval
prompt version
answer
citations
latency
2) GET /api/debug/search/{trace_id}

Trả riêng retrieval:

BM25
vector
RRF
filtered
final context
3) GET /api/debug/recent

Danh sách 50 trace gần nhất:

trace_id
query
status
total_latency
created_at
4) POST /api/evals/run

Chạy benchmark trên bộ 50–100 câu

5) GET /api/evals/{run_id}

Trả summary:

Recall@5
Recall@10
citation correct rate
grounded rate
hallucination rate
P50/P95 latency
JSON response debug mẫu
{
  "trace_id": "req_ab12cd34",
  "query_raw": "theo luật đất đai, quyền sử dụng đất là gì?",
  "query_tokenized": "theo luật_đất_đai quyền_sử_dụng_đất là gì",
  "bm25_hits": [
    {"chunk_id": "c1", "score": 13.2, "rank": 1, "article_ref": "Điều 3"},
    {"chunk_id": "c2", "score": 11.8, "rank": 2, "article_ref": "Điều 5"}
  ],
  "vector_hits": [
    {"chunk_id": "c8", "score": 0.89, "rank": 1, "article_ref": "Điều 4"},
    {"chunk_id": "c1", "score": 0.87, "rank": 2, "article_ref": "Điều 3"}
  ],
  "rrf_hits": [
    {"chunk_id": "c1", "score": 0.0325, "rank": 1},
    {"chunk_id": "c8", "score": 0.0317, "rank": 2}
  ],
  "filtered_out": [
    {"chunk_id": "c2", "reason": "duplicate_article"}
  ],
  "final_context": ["c1", "c8"],
  "prompt_version": "legal_v3",
  "model_name": "qwen2.5-7b-instruct",
  "answer": "...",
  "citations": ["Luật Đất đai 2024 - Điều 3"],
  "latency_ms": {
    "bm25": 22,
    "vector": 47,
    "fusion": 3,
    "llm": 1640,
    "total": 1731
  }
}
Eval layer

Đây là phần để bạn không phải debug hoàn toàn bằng cảm giác.

Format 1 case
{
  "case_id": "land_001",
  "question": "Quyền sử dụng đất được hiểu như thế nào?",
  "expected_doc_id": "luat_dat_dai_2024",
  "expected_article_ref": "Điều 3",
  "gold_answer": "..."
}
Chỉ số nên tính
Recall@5
Recall@10
Expected article hit rate
Citation correct rate
Grounded rate
Hallucination rate
P50 latency
P95 latency
Rule chấm practical
Retrieval pass
top-5 có đúng điều luật nguồn: 1/0
top-10 có đúng điều luật nguồn: 1/0
Answer pass
citation đúng nguồn: 1/0
grounded: 1/0
đầy đủ: 1–5
hallucination: 1/0
Giao diện debug tối thiểu

Bạn không cần dashboard đẹp ngay.

Chỉ cần một trang web đơn giản:

Trang 1: Recent traces
thời gian
query
trace_id
status
total latency
Trang 2: Trace detail
query
BM25 top-10
vector top-10
RRF top-10
filtered items
final chunks
prompt preview
answer
citations

Chỉ cần nhìn trang này là dev biết lỗi nằm ở đâu.

Taxonomy lỗi

Mỗi query fail nên được gán 1 lỗi chính:

INGEST_PARSE_ERROR
CHUNK_BOUNDARY_BAD
BM25_MISS
VECTOR_MISS
RRF_BAD_MERGE
FILTER_DROPPED_GOOD_CHUNK
PROMPT_OVERGENERALIZED
MODEL_HALLUCINATED
CITATION_WRONG
NO_ANSWER_WHEN_CONTEXT_EXISTS

Sau 100 case, bạn sẽ thấy lỗi tập trung ở đâu.

Lộ trình triển khai
Phase 1 — 1 đến 2 ngày

Mục tiêu: có trace cơ bản

thêm middleware trace_id
thêm JSON logger
lưu trace_runs
lưu bm25/vector/rrf/final
thêm /api/debug/traces/{trace_id}
Phase 2 — 2 đến 3 ngày

Mục tiêu: debug retrieval thật sự

lưu prompt version
lưu filtered reasons
thêm /api/debug/search/{trace_id}
làm UI debug tối thiểu
Phase 3 — 2 đến 4 ngày

Mục tiêu: đo chất lượng

tạo eval_cases
tạo eval_results
chạy benchmark 50–100 câu
xuất report Recall@5 / citation / grounded / latency
Phase 4 — sau đó

Mục tiêu: production hơn

chuyển SQLite -> Postgres
thêm aggregated metrics
thêm alert khi latency tăng hoặc grounded giảm
Nên làm ngay cái gì trước

Nếu phải chọn 3 thứ quan trọng nhất để làm ngay, tôi chốt:

trace_id + trace_runs
retrieval snapshot đầy đủ
eval set 50 câu thật

Chỉ 3 món này thôi là hệ của bạn đã bớt mù đi rất nhiều.

Kết luận

Với pipeline hiện tại của bạn, cách làm đúng không phải là đổi sang framework khác, mà là:

giữ nguyên lõi hybrid legal RAG hiện tại
rồi bọc thêm observability layer như trên.

Đó là cách ít rủi ro nhất, sát bài toán luật Việt nhất, và đủ thực dụng để dev triển khai ngay trên FastAPI hiện có. Kiến trúc gốc của bạn vốn đã tách API, retrieval, LLM và data pipeline rất rõ, nên đây là chỗ rất thuận để cấy thêm tracing/debug/eval.
