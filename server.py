"""
Legal RAG Chatbot — FastAPI Server
=====================================
Main server tích hợp: Hybrid Retrieval + Ollama Chat + SSE Streaming + Ngrok.

Usage:
    python server.py                      # Local HTTP (port 8000)
    python server.py --ngrok              # Expose qua ngrok
    python server.py --port 8080          # Custom port
"""

import argparse
import asyncio
import json
import re
import sys
import time
import os
import uuid
from pathlib import Path
from urllib import request as urllib_request
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from loguru import logger

# ── Project setup ──
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import cfg
from src.llm.prompt_builder import build_rag_prompt, build_search_summary, GREETING_SUGGESTIONS
from src.observability.metrics import summarize_eval_results
from src.observability.models import SearchHit, FilteredHit, RetrievalSnapshot, EvalCase
from src.observability.recorder import ObservabilityRecorder
from src.retrieval.text_tokenizer import tokenize_for_query

# ── Logger ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")

# ── FastAPI App ──
app = FastAPI(title="Trợ lý Pháp luật Nội vụ", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
_retriever = None
_query_preprocessor = None
_chat_sessions: dict[str, list] = {}  # session_id -> history
_obs = ObservabilityRecorder(
    db_path=cfg.LOG_DIR / "observability.sqlite3",
    events_log_path=cfg.LOG_DIR / "observability.events.jsonl",
)

PROMPT_VERSION = "legal_v3"
RETRIEVAL_CONFIG_VERSION = "hybrid_v2"
FEWSHOT_VERSION = "none"


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or f"req_{uuid.uuid4().hex[:12]}"
    request.state.trace_id = trace_id
    request.state.started_at = time.time()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


def _trace_id_from_request(request: Request) -> str:
    return getattr(request.state, "trace_id", f"req_{uuid.uuid4().hex[:12]}")


def _request_started_at(request: Request) -> float:
    return float(getattr(request.state, "started_at", time.time()))


def _load_index_version() -> str:
    versions: list[str] = []
    for prefix, path in [
        ("vec", cfg.VECTOR_INDEX_DIR / "config.json"),
        ("bm25", cfg.BM25_INDEX_DIR / "config.json"),
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            versions.append(
                data.get("index_version")
                or data.get("updated_at")
                or data.get("created_at")
                or prefix
            )
        except Exception:
            versions.append(prefix)
    return "|".join(versions) if versions else "unknown"


def _build_article_ref(hit: dict[str, Any]) -> str | None:
    article = str(hit.get("article", "")).strip()
    clause = str(hit.get("clause", "")).strip()
    if article and clause:
        return f"{article} - {clause}"
    if article:
        return article
    if clause:
        return clause
    return None


def _to_search_hit(hit: dict[str, Any], source: str, rank: int | None = None) -> SearchHit:
    score = hit.get("rrf_score", hit.get("score", 0.0))
    return SearchHit(
        chunk_id=str(hit.get("chunk_id", "")),
        doc_id=str(hit.get("doc_id", "")),
        title=str(hit.get("title", "")),
        article_ref=_build_article_ref(hit),
        score=float(score or 0.0),
        rank=int(hit.get("rank", 0) if rank is None else rank),
        source=source,
        breadcrumb=str(hit.get("path", "")).strip() or None,
    )


def _compact_hits(hits: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    out = []
    for hit in hits[:limit]:
        out.append({
            "chunk_id": hit.get("chunk_id"),
            "score": hit.get("rrf_score", hit.get("score")),
            "rank": hit.get("rank"),
            "doc_id": hit.get("doc_id"),
            "article_ref": _build_article_ref(hit),
        })
    return out


def _citations_from_contexts(contexts: list[dict], limit: int = 5) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for ctx in contexts:
        title = str(ctx.get("title", "")).strip()
        doc_num = str(ctx.get("document_number", "")).strip()
        article = str(ctx.get("article", "")).strip()
        raw = " - ".join([p for p in [doc_num, article, title] if p]).strip(" -")
        if not raw:
            raw = str(ctx.get("chunk_id", "")).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        citations.append(raw)
        if len(citations) >= limit:
            break
    return citations


def _extract_citations_from_answer(answer: str) -> list[str]:
    if not answer:
        return []
    patterns = [
        r"(Điều\s+\d+[^\n.;]*)",
        r"(\d{1,4}/\d{4}/[A-Z0-9\-]+)",
    ]
    citations: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, answer, flags=re.IGNORECASE):
            text = str(match).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            citations.append(text)
    return citations[:10]


def _build_retrieval_snapshot(
    trace_id: str,
    query_raw: str,
    query_tokenized: str,
    retrieval: dict[str, Any],
) -> RetrievalSnapshot:
    bm25_hits = [_to_search_hit(hit, "bm25") for hit in retrieval.get("bm25_hits", [])]
    vector_hits = [_to_search_hit(hit, "vector") for hit in retrieval.get("vector_hits", [])]
    rrf_hits = [_to_search_hit(hit, "rrf") for hit in retrieval.get("rrf_hits", [])]

    filtered_out: list[FilteredHit] = []
    for row in retrieval.get("filtered_out", []):
        filtered_out.append(
            FilteredHit(
                chunk_id=str(row.get("chunk_id", "")),
                reason=str(row.get("reason", "unknown")),
                detail=json.dumps(row, ensure_ascii=False),
            )
        )

    final_context_chunk_ids = [
        str(row.get("chunk_id", ""))
        for row in retrieval.get("final_results", [])
        if row.get("chunk_id")
    ]
    latencies = dict(retrieval.get("latencies_ms", {}))
    if "total" not in latencies:
        latencies["total"] = int(sum(v for v in latencies.values() if isinstance(v, int)))

    return RetrievalSnapshot(
        trace_id=trace_id,
        query_raw=query_raw,
        query_tokenized=query_tokenized,
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        rrf_hits=rrf_hits,
        filtered_out=filtered_out,
        final_context_chunk_ids=final_context_chunk_ids,
        latency_ms=latencies,
    )


def _record_retrieval(trace_id: str, snapshot: RetrievalSnapshot) -> None:
    _obs.record_retrieval_snapshot(snapshot)
    _obs.record_event(trace_id, "bm25_search", {
        "top_k": len(snapshot.bm25_hits),
        "hits": [h.model_dump() for h in snapshot.bm25_hits[:10]],
        "latency_ms": snapshot.latency_ms.get("bm25", 0),
    })
    _obs.record_event(trace_id, "vector_search", {
        "top_k": len(snapshot.vector_hits),
        "hits": [h.model_dump() for h in snapshot.vector_hits[:10]],
        "latency_ms": snapshot.latency_ms.get("vector", 0),
    })
    _obs.record_event(trace_id, "rrf_fusion", {
        "top_k": len(snapshot.rrf_hits),
        "hits": [h.model_dump() for h in snapshot.rrf_hits[:10]],
        "latency_ms": snapshot.latency_ms.get("rrf", 0),
    })
    _obs.record_event(trace_id, "dedup_filter", {
        "filtered_out": [h.model_dump() for h in snapshot.filtered_out[:50]],
        "final_context_chunk_ids": snapshot.final_context_chunk_ids,
        "latency_ms": snapshot.latency_ms.get("dedup", 0),
    })


def _serialize_messages_for_debug(messages: list[dict]) -> str:
    """Format toàn bộ chat messages để hiển thị debug trên UI."""
    blocks = []
    for idx, msg in enumerate(messages, 1):
        role = (msg.get("role") or "unknown").upper()
        content = (msg.get("content") or "").strip()
        blocks.append(f"[{idx}] {role}\n{content}")
    return "\n\n".join(blocks)


def _init_retriever():
    """Khởi tạo retrieval engine + query preprocessor."""
    global _retriever, _query_preprocessor
    try:
        from src.retrieval.hybrid import HybridRetriever
        _retriever = HybridRetriever(
            bm25_index_dir=cfg.BM25_INDEX_DIR,
            vector_index_dir=cfg.VECTOR_INDEX_DIR,
            chunks_path=cfg.PROCESSED_DIR / "chunks.jsonl",
            ollama_url=cfg.OLLAMA_BASE_URL,
            embedding_model=cfg.EMBEDDING_MODEL,
            rrf_k=cfg.RRF_K,
        )
        logger.info(f"Retriever initialized: mode={_retriever.mode}")
    except Exception as e:
        logger.error(f"Retriever init failed: {e}")
        _retriever = None

    try:
        from src.llm.query_preprocessor import QueryPreprocessor
        _query_preprocessor = QueryPreprocessor(
            ollama_url=cfg.OLLAMA_BASE_URL,
            chat_model=cfg.CHAT_MODEL,
        )
    except Exception as e:
        logger.warning(f"QueryPreprocessor init failed: {e}")
        _query_preprocessor = None


# ── API Routes ──

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve frontend HTML."""
    html_path = _PROJECT_ROOT / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Legal RAG Chatbot</h1><p>web/index.html not found</p>")


@app.get("/api/health")
async def health_check():
    """System health check."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "retriever": _retriever.mode if _retriever else "not_loaded",
        "ollama": "online" if ollama_ok else "offline",
        "chat_model": cfg.CHAT_MODEL,
        "embedding_model": cfg.EMBEDDING_MODEL,
    }


@app.get("/api/suggestions")
async def get_suggestions():
    """Câu hỏi gợi ý cho Nội vụ."""
    return {"suggestions": GREETING_SUGGESTIONS}


@app.post("/api/search")
async def search_endpoint(request: Request):
    """Retrieval-only search (không gọi LLM)."""
    trace_id = _trace_id_from_request(request)
    request_started_at = _request_started_at(request)
    data = await request.json()
    query = data.get("query", "").strip()
    top_k = data.get("top_k", cfg.HYBRID_TOP_K)

    _obs.start_trace(
        trace_id=trace_id,
        endpoint="/api/search",
        user_query=query,
        prompt_version=PROMPT_VERSION,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        index_version=_load_index_version(),
        model_name=cfg.CHAT_MODEL,
        model_version=cfg.CHAT_MODEL,
    )
    _obs.record_event(trace_id, "request_received", {
        "endpoint": "/api/search",
        "query_raw": query,
        "top_k": top_k,
    })

    if not query:
        _obs.record_event(trace_id, "error", {"error": "Query is empty"})
        _obs.complete_trace_error(
            trace_id=trace_id,
            error_message="Query is empty",
            total_latency_ms=round((time.time() - request_started_at) * 1000),
        )
        return JSONResponse({"error": "Query is empty"}, status_code=400)

    if not _retriever:
        _obs.record_event(trace_id, "error", {"error": "Retriever not loaded"})
        _obs.complete_trace_error(
            trace_id=trace_id,
            error_message="Retriever not loaded",
            total_latency_ms=round((time.time() - request_started_at) * 1000),
        )
        return JSONResponse({"error": "Retriever not loaded"}, status_code=503)

    try:
        query_tokenized = tokenize_for_query(query)
        _obs.update_query_tokenized(trace_id, query_tokenized)
        _obs.record_event(trace_id, "query_normalized", {
            "query_raw": query,
            "query_tokenized": query_tokenized,
        })

        retrieval = _retriever.search_with_snapshot(
            query=query,
            bm25_top_k=cfg.BM25_TOP_K,
            vector_top_k=cfg.VECTOR_TOP_K,
            final_top_k=top_k,
        )
        results = retrieval.get("final_results", [])
        snapshot = _build_retrieval_snapshot(trace_id, query, query_tokenized, retrieval)
        _record_retrieval(trace_id, snapshot)

        total_elapsed = round((time.time() - request_started_at) * 1000)
        retrieval_ms = int(snapshot.latency_ms.get("total", 0))
        _obs.complete_trace_success(
            trace_id=trace_id,
            total_latency_ms=total_elapsed,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=0,
            answer_text=None,
            citations=_citations_from_contexts(results),
            used_context=[r.get("chunk_id", "") for r in results if r.get("chunk_id")],
        )
        _obs.record_event(trace_id, "response_sent", {
            "result_count": len(results),
            "total_latency_ms": total_elapsed,
        })

        return {
            "trace_id": trace_id,
            "query": query,
            "query_tokenized": query_tokenized,
            "results": results,
            "total": len(results),
            "mode": _retriever.mode,
            "elapsed_ms": total_elapsed,
        }
    except Exception as e:
        error_message = f"search_failed: {type(e).__name__}: {e}"
        _obs.record_event(trace_id, "error", {"error": error_message})
        _obs.complete_trace_error(
            trace_id=trace_id,
            error_message=error_message,
            total_latency_ms=round((time.time() - request_started_at) * 1000),
        )
        logger.error(error_message)
        return JSONResponse({"error": str(e), "trace_id": trace_id}, status_code=500)


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Chat endpoint với SSE streaming.
    Trả response dạng text/event-stream.
    """
    trace_id = _trace_id_from_request(request)
    request_started_at = _request_started_at(request)
    data = await request.json()
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")

    _obs.start_trace(
        trace_id=trace_id,
        endpoint="/api/chat",
        user_query=query,
        prompt_version=PROMPT_VERSION,
        retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
        index_version=_load_index_version(),
        model_name=cfg.CHAT_MODEL,
        model_version=cfg.CHAT_MODEL,
    )
    _obs.record_event(trace_id, "request_received", {
        "endpoint": "/api/chat",
        "session_id": session_id,
        "query_raw": query,
    })

    if not query:
        _obs.record_event(trace_id, "error", {"error": "Query is empty"})
        _obs.complete_trace_error(
            trace_id=trace_id,
            error_message="Query is empty",
            total_latency_ms=round((time.time() - request_started_at) * 1000),
        )
        return JSONResponse({"error": "Query is empty"}, status_code=400)

    # Get/create session history
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []
    history = _chat_sessions[session_id]

    async def event_generator():
        nonlocal query
        trace_closed = False
        search_query = query
        query_tokenized = tokenize_for_query(query)
        contexts: list[dict] = []
        retrieval_ms = 0
        llm_ms = 0
        full_response = ""

        def finalize_success(
            *,
            answer_text: str,
            citations: list[str],
            used_context: list[str],
            grounded_flag: int | None,
        ) -> None:
            nonlocal trace_closed
            if trace_closed:
                return
            total_ms = round((time.time() - request_started_at) * 1000)
            _obs.complete_trace_success(
                trace_id=trace_id,
                total_latency_ms=total_ms,
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=llm_ms,
                answer_text=answer_text,
                citations=citations,
                used_context=used_context,
                grounded_flag=grounded_flag,
            )
            _obs.record_event(trace_id, "response_sent", {
                "status": "success",
                "total_latency_ms": total_ms,
                "retrieval_latency_ms": retrieval_ms,
                "llm_latency_ms": llm_ms,
            })
            trace_closed = True

        def finalize_error(error_message: str) -> None:
            nonlocal trace_closed
            if trace_closed:
                return
            total_ms = round((time.time() - request_started_at) * 1000)
            _obs.record_event(trace_id, "error", {"error": error_message})
            _obs.complete_trace_error(
                trace_id=trace_id,
                error_message=error_message,
                total_latency_ms=total_ms,
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=llm_ms,
            )
            trace_closed = True

        try:
            # Step 0: Preprocess query (thêm dấu, detect ngôn ngữ)
            if _query_preprocessor:
                pp = _query_preprocessor.process(query)
                search_query = pp["processed"] if pp.get("processed") else query
                query_tokenized = tokenize_for_query(search_query)
                _obs.update_query_tokenized(trace_id, query_tokenized)
                _obs.record_event(trace_id, "query_normalized", {
                    "query_raw": query,
                    "query_tokenized": query_tokenized,
                    "lang": pp.get("lang"),
                    "enriched": pp.get("enriched", False),
                    "rejected": pp.get("rejected", False),
                })

                # Gửi enrichment info về UI
                if pp["enriched"]:
                    yield f"data: {json.dumps({'type': 'enrichment', 'original': pp['original'], 'processed': pp['processed'], 'lang': pp['lang']}, ensure_ascii=False)}\n\n"

                # Nếu rejected (tiếng Anh, empty) → trả thông báo
                if pp["rejected"]:
                    yield f"data: {json.dumps({'type': 'token', 'content': pp['reject_message']}, ensure_ascii=False)}\n\n"
                    finalize_success(
                        answer_text=pp["reject_message"],
                        citations=[],
                        used_context=[],
                        grounded_flag=1,
                    )
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
            else:
                _obs.update_query_tokenized(trace_id, query_tokenized)
                _obs.record_event(trace_id, "query_normalized", {
                    "query_raw": query,
                    "query_tokenized": query_tokenized,
                    "lang": "unknown",
                    "enriched": False,
                    "rejected": False,
                })

            # Step 1: Retrieve context
            if _retriever:
                retrieval_data = _retriever.search_with_snapshot(
                    query=search_query,
                    bm25_top_k=cfg.BM25_TOP_K,
                    vector_top_k=cfg.VECTOR_TOP_K,
                    final_top_k=cfg.HYBRID_TOP_K,
                )
                contexts = retrieval_data.get("final_results", [])
                snapshot = _build_retrieval_snapshot(
                    trace_id=trace_id,
                    query_raw=query,
                    query_tokenized=query_tokenized,
                    retrieval=retrieval_data,
                )
                retrieval_ms = int(snapshot.latency_ms.get("total", 0))
                _record_retrieval(trace_id, snapshot)

                # Send sources info
                sources = build_search_summary(contexts)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'retrieval_ms': retrieval_ms}, ensure_ascii=False)}\n\n"
            else:
                _obs.record_event(trace_id, "dedup_filter", {
                    "filtered_out": [],
                    "final_context_chunk_ids": [],
                    "latency_ms": 0,
                })
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'retrieval_ms': 0}, ensure_ascii=False)}\n\n"

            # Step 2: Build prompt
            messages = build_rag_prompt(search_query, contexts, history)
            prompt_debug_text = _serialize_messages_for_debug(messages)
            _obs.record_event(trace_id, "prompt_built", {
                "prompt_version": PROMPT_VERSION,
                "fewshot_version": FEWSHOT_VERSION,
                "retrieval_config_version": RETRIEVAL_CONFIG_VERSION,
                "model_name": cfg.CHAT_MODEL,
                "top_k_context": len(contexts),
                "message_count": len(messages),
            })
            yield f"data: {json.dumps({'type': 'prompt_debug', 'content': prompt_debug_text}, ensure_ascii=False)}\n\n"

            # Step 3: Stream from Ollama
            _obs.record_event(trace_id, "llm_started", {
                "model_name": cfg.CHAT_MODEL,
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 2048,
            })
            llm_t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{cfg.OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": cfg.CHAT_MODEL,
                            "messages": messages,
                            "stream": True,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": 2048,
                            },
                        },
                        timeout=120,
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    token = chunk["message"]["content"]
                                    full_response += token
                                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    ConnectionRefusedError, OSError) as e:
                logger.warning(f"Ollama connection failed: {type(e).__name__}: {e}")
                finalize_error(f"Ollama connection failed: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Ollama server không phản hồi. Hãy chạy: ollama serve && ollama pull ' + cfg.CHAT_MODEL}, ensure_ascii=False)}\n\n"
                return
            except Exception as e:
                logger.error(f"Ollama chat error: {type(e).__name__}: {e}")
                finalize_error(f"Ollama chat error: {type(e).__name__}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Lỗi LLM: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
                return

            llm_ms = round((time.perf_counter() - llm_t0) * 1000)
            _obs.record_event(trace_id, "llm_finished", {
                "llm_latency_ms": llm_ms,
                "answer_chars": len(full_response),
            })

            # Step 4: Save to history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": full_response})

            # Trim history
            if len(history) > cfg.MAX_HISTORY_TURNS * 2:
                _chat_sessions[session_id] = history[-(cfg.MAX_HISTORY_TURNS * 2):]

            citations = _extract_citations_from_answer(full_response)
            if not citations:
                citations = _citations_from_contexts(contexts)
            grounded_flag = 1 if full_response.strip() else 0
            used_context = [c.get("chunk_id", "") for c in contexts if c.get("chunk_id")]
            finalize_success(
                answer_text=full_response,
                citations=citations,
                used_context=used_context,
                grounded_flag=grounded_flag,
            )

            # Done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            finalize_error(f"Chat error: {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
async def clear_session(request: Request):
    """Clear chat history."""
    data = await request.json()
    session_id = data.get("session_id", "default")
    _chat_sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {"id": sid, "turns": len(h) // 2}
            for sid, h in _chat_sessions.items()
        ]
    }


# ── Ngrok integration (from Meeting-trans) ──

@app.get("/api/debug/recent")
async def debug_recent(limit: int = 50):
    traces = _obs.get_recent(limit=limit)
    return {"total": len(traces), "traces": traces}


@app.get("/api/debug/traces/{trace_id}")
async def debug_trace_detail(trace_id: str):
    detail = _obs.get_trace_detail(trace_id)
    if not detail:
        return JSONResponse({"error": "Trace not found", "trace_id": trace_id}, status_code=404)
    return detail


@app.get("/api/debug/search/{trace_id}")
async def debug_search_detail(trace_id: str):
    detail = _obs.get_search_detail(trace_id)
    if _obs.store.get_trace(trace_id) is None:
        return JSONResponse({"error": "Trace not found", "trace_id": trace_id}, status_code=404)
    return detail


def _normalize_text_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _hit_matches_expected_doc(hit: dict[str, Any], expected_doc_id: str | None) -> bool:
    if not expected_doc_id:
        return False
    expected = _normalize_text_for_match(expected_doc_id)
    doc_id = _normalize_text_for_match(hit.get("doc_id", ""))
    doc_num = _normalize_text_for_match(hit.get("document_number", ""))
    title = _normalize_text_for_match(hit.get("title", ""))
    path = _normalize_text_for_match(hit.get("path", ""))
    return expected in doc_id or expected in doc_num or expected in title or expected in path


def _hit_matches_expected_article(hit: dict[str, Any], expected_article_ref: str | None) -> bool:
    if not expected_article_ref:
        return False
    expected = _normalize_text_for_match(expected_article_ref)
    article = _normalize_text_for_match(hit.get("article", ""))
    clause = _normalize_text_for_match(hit.get("clause", ""))
    path = _normalize_text_for_match(hit.get("path", ""))
    return expected in article or expected in clause or expected in path


def _answer_contains_expected(answer: str, case: EvalCase) -> bool:
    answer_norm = _normalize_text_for_match(answer)
    if not answer_norm:
        return False
    checks: list[bool] = []
    if case.expected_doc_id:
        checks.append(_normalize_text_for_match(case.expected_doc_id) in answer_norm)
    if case.expected_article_ref:
        checks.append(_normalize_text_for_match(case.expected_article_ref) in answer_norm)
    if not checks:
        return False
    return any(checks)


def _completeness_score(answer: str) -> int:
    length = len((answer or "").strip())
    if length == 0:
        return 1
    if length < 120:
        return 2
    if length < 300:
        return 3
    if length < 700:
        return 4
    return 5


def _load_eval_cases_from_path(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if not path.exists():
        return cases
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("case_id"):
                    row["case_id"] = f"case_{idx:04d}"
                cases.append(EvalCase(**row))
            except Exception:
                continue
    return cases


async def _run_llm_non_stream(messages: list[dict[str, Any]]) -> tuple[str, int]:
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": cfg.CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 2048,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    answer = ((payload.get("message") or {}).get("content") or "").strip()
    latency_ms = round((time.perf_counter() - t0) * 1000)
    return answer, latency_ms


@app.post("/api/evals/run")
async def run_evals(request: Request):
    if not _retriever:
        return JSONResponse({"error": "Retriever not loaded"}, status_code=503)

    data = await request.json()
    run_id = f"eval_{uuid.uuid4().hex[:10]}"
    with_generation = bool(data.get("with_generation", False))
    limit = int(data.get("limit", 50))
    final_top_k = max(int(data.get("final_top_k", 10)), 10)

    cases: list[EvalCase] = []
    for row in data.get("cases", []) or []:
        try:
            cases.append(EvalCase(**row))
        except Exception:
            continue

    if not cases and data.get("cases_path"):
        cases.extend(_load_eval_cases_from_path(Path(str(data.get("cases_path")))))
    if not cases:
        stored = _obs.store.list_eval_cases(limit=limit)
        for row in stored:
            try:
                cases.append(EvalCase(**row))
            except Exception:
                continue
    if not cases:
        return JSONResponse(
            {
                "error": "No eval cases provided/found",
                "hint": "Pass body.cases or body.cases_path, or insert into eval_cases table first.",
            },
            status_code=400,
        )

    cases = cases[:limit]
    _obs.store.upsert_eval_cases([c.model_dump() for c in cases])
    _obs.store.create_eval_run(run_id, total_cases=len(cases))

    results: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            trace_id = f"{run_id}_{idx:04d}"
            start_ts = time.time()
            _obs.start_trace(
                trace_id=trace_id,
                endpoint="/api/evals/run",
                user_query=case.question,
                prompt_version=PROMPT_VERSION,
                retrieval_config_version=RETRIEVAL_CONFIG_VERSION,
                index_version=_load_index_version(),
                model_name=cfg.CHAT_MODEL,
                model_version=cfg.CHAT_MODEL,
            )
            _obs.record_event(trace_id, "request_received", {
                "endpoint": "/api/evals/run",
                "run_id": run_id,
                "case_id": case.case_id,
                "query_raw": case.question,
            })

            search_query = case.question
            if _query_preprocessor:
                pp = _query_preprocessor.process(case.question)
                search_query = pp.get("processed") or case.question
            query_tokenized = tokenize_for_query(search_query)
            _obs.update_query_tokenized(trace_id, query_tokenized)
            _obs.record_event(trace_id, "query_normalized", {
                "query_raw": case.question,
                "query_tokenized": query_tokenized,
            })

            retrieval = _retriever.search_with_snapshot(
                query=search_query,
                bm25_top_k=cfg.BM25_TOP_K,
                vector_top_k=cfg.VECTOR_TOP_K,
                final_top_k=final_top_k,
            )
            snapshot = _build_retrieval_snapshot(trace_id, case.question, query_tokenized, retrieval)
            _record_retrieval(trace_id, snapshot)

            ranked = retrieval.get("reranked_hits", [])
            first_rank = None
            for r_idx, hit in enumerate(ranked, 1):
                if _hit_matches_expected_doc(hit, case.expected_doc_id):
                    first_rank = r_idx
                    break

            hit_top5 = int(any(_hit_matches_expected_doc(hit, case.expected_doc_id) for hit in ranked[:5]))
            hit_top10 = int(any(_hit_matches_expected_doc(hit, case.expected_doc_id) for hit in ranked[:10]))
            hit_expected_article = int(any(
                _hit_matches_expected_doc(hit, case.expected_doc_id)
                and _hit_matches_expected_article(hit, case.expected_article_ref)
                for hit in ranked[:10]
            ))

            contexts = retrieval.get("final_results", [])[:cfg.HYBRID_TOP_K]
            answer_text = ""
            answer_latency_ms = 0
            citation_correct = None
            grounded = None
            completeness = None
            hallucination = None

            if with_generation:
                messages = build_rag_prompt(search_query, contexts, history=[])
                _obs.record_event(trace_id, "prompt_built", {
                    "prompt_version": PROMPT_VERSION,
                    "fewshot_version": FEWSHOT_VERSION,
                    "retrieval_config_version": RETRIEVAL_CONFIG_VERSION,
                    "model_name": cfg.CHAT_MODEL,
                    "top_k_context": len(contexts),
                    "message_count": len(messages),
                })
                _obs.record_event(trace_id, "llm_started", {"model_name": cfg.CHAT_MODEL})
                answer_text, answer_latency_ms = await _run_llm_non_stream(messages)
                _obs.record_event(trace_id, "llm_finished", {
                    "answer_chars": len(answer_text),
                    "llm_latency_ms": answer_latency_ms,
                })
                citations = _extract_citations_from_answer(answer_text) or _citations_from_contexts(contexts)
                citation_correct = int(_answer_contains_expected(answer_text, case))
                grounded = int(bool(answer_text.strip()) and (hit_top10 == 1 or citation_correct == 1))
                completeness = _completeness_score(answer_text)
                hallucination = int(bool(answer_text.strip()) and hit_top10 == 0 and citation_correct == 0)
                _obs.complete_trace_success(
                    trace_id=trace_id,
                    retrieval_latency_ms=int(snapshot.latency_ms.get("total", 0)),
                    llm_latency_ms=answer_latency_ms,
                    total_latency_ms=round((time.time() - start_ts) * 1000),
                    answer_text=answer_text,
                    citations=citations,
                    used_context=[c.get("chunk_id", "") for c in contexts if c.get("chunk_id")],
                    grounded_flag=grounded,
                )
            else:
                _obs.complete_trace_success(
                    trace_id=trace_id,
                    retrieval_latency_ms=int(snapshot.latency_ms.get("total", 0)),
                    llm_latency_ms=0,
                    total_latency_ms=round((time.time() - start_ts) * 1000),
                    answer_text="",
                    citations=[],
                    used_context=[c.get("chunk_id", "") for c in contexts if c.get("chunk_id")],
                    grounded_flag=None,
                )

            total_latency_ms = round((time.time() - start_ts) * 1000)
            row = {
                "run_id": run_id,
                "case_id": case.case_id,
                "trace_id": trace_id,
                "hit_top5": hit_top5,
                "hit_top10": hit_top10,
                "rank_of_first_correct": first_rank,
                "hit_expected_source": hit_top5,
                "hit_expected_article": hit_expected_article,
                "citation_correct": citation_correct,
                "grounded": grounded,
                "completeness_score": completeness,
                "hallucination": hallucination,
                "retrieval_latency_ms": int(snapshot.latency_ms.get("total", 0)),
                "answer_latency_ms": answer_latency_ms,
                "total_latency_ms": total_latency_ms,
                "notes": {
                    "expected_doc_id": case.expected_doc_id,
                    "expected_article_ref": case.expected_article_ref,
                    "with_generation": with_generation,
                },
            }
            _obs.store.insert_eval_result(row)
            results.append(row)

        summary = summarize_eval_results(results)
        _obs.store.finish_eval_run(run_id, status="success", summary=summary)
        return {
            "run_id": run_id,
            "with_generation": with_generation,
            "total_cases": len(cases),
            "summary": summary,
        }
    except Exception as e:
        _obs.store.finish_eval_run(
            run_id,
            status="error",
            summary={"total_cases": len(results)},
            error_message=f"{type(e).__name__}: {e}",
        )
        logger.error(f"Eval run failed {run_id}: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": str(e), "run_id": run_id, "processed_cases": len(results)},
            status_code=500,
        )


@app.get("/api/evals/{run_id}")
async def get_eval_run(run_id: str):
    run = _obs.store.get_eval_run(run_id)
    if not run:
        return JSONResponse({"error": "Eval run not found", "run_id": run_id}, status_code=404)
    results = _obs.store.list_eval_results(run_id)
    summary = run.get("summary_json") or summarize_eval_results(results)
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "total_cases": run.get("total_cases"),
        "summary": summary,
        "results": results,
        "error_message": run.get("error_message"),
    }


def _normalize_ngrok_addr(addr) -> str:
    if addr is None:
        return ""
    addr = str(addr).strip()
    for prefix in ("http://", "https://"):
        if addr.startswith(prefix):
            addr = addr[len(prefix):]
    return addr.rstrip("/")


def _desired_ngrok_addrs(port: int) -> set:
    return {str(port), f"localhost:{port}", f"127.0.0.1:{port}", f"0.0.0.0:{port}"}


def _list_local_ngrok_tunnels() -> list:
    req = urllib_request.Request("http://127.0.0.1:4040/api/tunnels", method="GET")
    try:
        with urllib_request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("tunnels", []) or []
    except Exception:
        return []


def _delete_local_ngrok_tunnel(uri: str) -> bool:
    if not uri:
        return False
    req = urllib_request.Request(f"http://127.0.0.1:4040{uri}", method="DELETE")
    try:
        with urllib_request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _find_local_ngrok_tunnel_for_port(port: int):
    desired = _desired_ngrok_addrs(port)
    for tunnel in _list_local_ngrok_tunnels():
        addr = _normalize_ngrok_addr((tunnel.get("config") or {}).get("addr"))
        if addr in desired:
            return tunnel
    return None


def _close_conflicting_tunnels(port: int) -> int:
    desired = _desired_ngrok_addrs(port)
    closed = 0
    for tunnel in _list_local_ngrok_tunnels():
        addr = _normalize_ngrok_addr((tunnel.get("config") or {}).get("addr"))
        if addr in desired:
            continue
        if _delete_local_ngrok_tunnel(tunnel.get("uri", "")):
            closed += 1
    return closed


def start_server(host: str, port: int, use_ngrok: bool = False):
    """Khởi động server."""
    import time as _time

    cfg.ensure_dirs()

    # Init retriever
    _init_retriever()

    print("=" * 60)
    print("  🏛️  LEGAL RAG CHATBOT")
    print(f"  Mode: {'HYBRID (BM25 + Vector)' if _retriever and _retriever.mode == 'hybrid' else 'BM25 ONLY'}")
    print(f"  Chat model: {cfg.CHAT_MODEL}")
    print(f"  Embedding: {cfg.EMBEDDING_MODEL}")
    print(f"  Local: http://{host}:{port}")
    print("=" * 60)

    # Ngrok tunnel
    if use_ngrok:
        try:
            from pyngrok import ngrok, conf as ngrok_conf
            tunnel = None

            # Set authtoken từ .env
            ngrok_token = os.getenv("ngrok_token", "").strip()
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
                print(f"[NGROK] Auth token set từ .env")
            else:
                print("[NGROK] Không tìm thấy ngrok_token trong .env, dùng config mặc định")

            # Đóng tất cả tunnel cũ trên port này để tránh xung đột
            existing = _find_local_ngrok_tunnel_for_port(port)
            if existing:
                print(f"[NGROK] Đóng tunnel cũ trên port {port}...")
                try:
                    ngrok.disconnect(existing.get("public_url", ""))
                except Exception:
                    _close_conflicting_tunnels(port)
                _time.sleep(1)

            # Tạo tunnel mới
            try:
                tunnel = ngrok.connect(port, "http")
            except Exception as e:
                err_msg = str(e)
                if "ERR_NGROK_334" in err_msg or "already online" in err_msg:
                    print("[NGROK] Tunnel bị xung đột, đang dọn dẹp...")
                    _close_conflicting_tunnels(port)
                    _time.sleep(2)
                    tunnel = ngrok.connect(port, "http")
                else:
                    raise

            print(f"\n{'=' * 60}")
            print(f"  🌐 NGROK TUNNEL ACTIVE")
            print(f"  Public URL: {tunnel.public_url}")
            print(f"  Chia sẻ URL này để demo!")
            print(f"{'=' * 60}\n")
        except ImportError:
            print("[LỖI] Cần cài pyngrok: pip install pyngrok")
        except Exception as e:
            print(f"[LỖI] Không thể tạo ngrok tunnel: {e}")
            print("       Kiểm tra ngrok_token trong .env")

    # Filter noisy logs
    import logging

    class _Filter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage() if hasattr(record, 'getMessage') else ""
            return "/api/health" not in msg

    logging.getLogger("uvicorn.access").addFilter(_Filter())

    uvicorn.run(app=app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal RAG Chatbot Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8899,
                        help="Port (mặc định 8899 để tránh xung đột với apps khác)")
    parser.add_argument("--ngrok", action="store_true",
                        help="Tạo ngrok tunnel (truy cap từ Internet)")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, use_ngrok=args.ngrok)
