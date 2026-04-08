# Phase 05: Evaluation Framework
Status: ⬜ Pending
Dependencies: Phase 03, 04

## Objective
Xây dựng hệ thống đánh giá có cấu trúc để đo lường và cải thiện chất lượng RAG pipeline.

## Evaluation Record Schema
Mỗi query tạo 1 record gồm 4 nhóm:

### Nhóm 1: Thông tin đầu vào
```json
{
  "query": "Điều kiện tuyển dụng công chức?",
  "query_type": "factual",       // factual | procedural | comparative | multi-hop
  "expected_sources": ["luat-can-bo-cong-chuc-80-2025"],
  "reference_note": "Điều 19, Khoản 1"
}
```

### Nhóm 2: Retrieval metrics
```json
{
  "retrieved_top5": ["chunk_001", "chunk_042", ...],
  "correct_source_found": true,
  "rank_of_first_correct": 1,
  "retrieval_latency_ms": 45
}
```

### Nhóm 3: Generation metrics  
```json
{
  "answer": "Theo Điều 19 Luật 80/2025/QH15...",
  "used_context": true,
  "citation_correct": true,
  "grounded": true,        // answer chỉ dùng info trong context
  "complete_score": 4,     // 1-5: đầy đủ thông tin
  "hallucination": false,
  "answer_latency_ms": 2100
}
```

### Nhóm 4: Hệ thống
```json
{
  "total_latency_ms": 2145,
  "error": null,
  "model_version": "qwen2.5:7b-instruct",
  "prompt_version": "v2.0-noiwu",
  "index_version": "20260408-raw55"
}
```

## Implementation Steps
1. [ ] Tạo `eval/eval_schema.py` — Pydantic models cho record
2. [ ] Tạo `eval/eval_runner.py` — chạy batch queries, ghi records
3. [ ] Tạo `eval/eval_metrics.py` — tính aggregate metrics:
   - Recall@5, MRR
   - Citation accuracy, Grounded rate
   - Hallucination rate
   - Avg / P50 / P95 latency
4. [ ] Tạo `eval/eval_analyzer.py` — phân nhóm lỗi:
   - Retrieval failures
   - Generation failures (retrieval OK but answer wrong)
   - Citation failures
   - Out-of-scope
5. [ ] Tạo `eval/eval_report.py` — xuất báo cáo markdown
6. [ ] Thêm `/api/eval` endpoint cho server

## Diagnostic Flow (Bước 4 — sửa đúng chỗ)
```
Lỗi retrieval → sửa: chunking, embedding, query rewrite, metadata, top-k, reranking
Lỗi generation → sửa: prompt, answer template, citation instruction, max context
Lỗi hệ thống  → sửa: GPU, batch, top-k, prompt length, quantization, caching
```

## Files to Create
- `eval/__init__.py`
- `eval/eval_schema.py` — Record models
- `eval/eval_runner.py` — Batch evaluation  
- `eval/eval_metrics.py` — Aggregate metrics
- `eval/eval_analyzer.py` — Error categorization
- `eval/eval_report.py` — Report generation

## Tương thích với Reranker
### Hiện tại CÓ THỂ làm ngay:
- Record tracking đầy đủ (prompt_version, index_version)
- So sánh A/B prompt, index, model

### Cần bổ sung cho Reranker (Phase sau):
- Cross-encoder model (bge-reranker-v2-m3)  
- `src/retrieval/reranker.py` — rerank top-30 → top-10
- Pipeline: BM25 top-50 → RRF → Reranker top-10 → LLM
