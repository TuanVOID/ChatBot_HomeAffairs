from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.observability.logger import JsonlStructuredLogger
from src.observability.models import RetrievalSnapshot
from src.observability.storage import SQLiteObservabilityStore, utc_now_iso


class ObservabilityRecorder:
    def __init__(self, db_path: Path, events_log_path: Path):
        self._store = SQLiteObservabilityStore(db_path=db_path)
        self._json_logger = JsonlStructuredLogger(events_log_path)

    @property
    def store(self) -> SQLiteObservabilityStore:
        return self._store

    def start_trace(
        self,
        *,
        trace_id: str,
        endpoint: str,
        user_query: str,
        query_tokenized: str | None = None,
        prompt_version: str | None = None,
        retrieval_config_version: str | None = None,
        index_version: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self._store.upsert_trace_start(
            trace_id=trace_id,
            endpoint=endpoint,
            user_query=user_query,
            query_tokenized=query_tokenized,
            prompt_version=prompt_version,
            retrieval_config_version=retrieval_config_version,
            index_version=index_version,
            model_name=model_name,
            model_version=model_version,
            status="running",
        )

    def complete_trace_success(
        self,
        *,
        trace_id: str,
        total_latency_ms: int | None = None,
        retrieval_latency_ms: int | None = None,
        llm_latency_ms: int | None = None,
        answer_text: str | None = None,
        citations: list[str] | None = None,
        used_context: list[str] | None = None,
        grounded_flag: int | None = None,
    ) -> None:
        self._store.update_trace(
            trace_id,
            status="success",
            finished_at=utc_now_iso(),
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            answer_text=answer_text,
            citations_json=json.dumps(citations or [], ensure_ascii=False),
            used_context_json=json.dumps(used_context or [], ensure_ascii=False),
            grounded_flag=grounded_flag,
            error_message=None,
        )

    def complete_trace_error(
        self,
        *,
        trace_id: str,
        error_message: str,
        total_latency_ms: int | None = None,
        retrieval_latency_ms: int | None = None,
        llm_latency_ms: int | None = None,
    ) -> None:
        self._store.update_trace(
            trace_id,
            status="error",
            finished_at=utc_now_iso(),
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            error_message=error_message,
        )

    def update_query_tokenized(self, trace_id: str, query_tokenized: str) -> None:
        self._store.update_trace(trace_id, query_tokenized=query_tokenized)

    def update_prompt_metadata(
        self,
        trace_id: str,
        *,
        prompt_version: str | None = None,
        retrieval_config_version: str | None = None,
        index_version: str | None = None,
    ) -> None:
        updates: dict[str, Any] = {}
        if prompt_version is not None:
            updates["prompt_version"] = prompt_version
        if retrieval_config_version is not None:
            updates["retrieval_config_version"] = retrieval_config_version
        if index_version is not None:
            updates["index_version"] = index_version
        if updates:
            self._store.update_trace(trace_id, **updates)

    def record_event(self, trace_id: str, stage: str, payload: dict[str, Any]) -> None:
        record = {
            "trace_id": trace_id,
            "stage": stage,
            "timestamp": utc_now_iso(),
            **payload,
        }
        self._store.insert_event(trace_id, stage, record)
        self._json_logger.write(record)

    def record_retrieval_snapshot(self, snapshot: RetrievalSnapshot) -> None:
        self._store.update_trace(
            snapshot.trace_id,
            query_tokenized=snapshot.query_tokenized,
            retrieval_latency_ms=snapshot.latency_ms.get("total"),
            used_context_json=json.dumps(snapshot.final_context_chunk_ids, ensure_ascii=False),
        )

        bm25_hits = [h.model_dump() for h in snapshot.bm25_hits]
        vector_hits = [h.model_dump() for h in snapshot.vector_hits]
        rrf_hits = [h.model_dump() for h in snapshot.rrf_hits]
        rrf_by_chunk = {h["chunk_id"]: h for h in rrf_hits}
        final_hits: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(snapshot.final_context_chunk_ids):
            base = dict(rrf_by_chunk.get(chunk_id, {
                "chunk_id": chunk_id,
                "doc_id": "",
                "title": "",
                "article_ref": None,
                "score": 0.0,
                "rank": idx,
                "source": "final",
                "breadcrumb": None,
            }))
            base["rank"] = idx
            final_hits.append(base)
        filtered = [f.model_dump() for f in snapshot.filtered_out]

        self._store.insert_hits(snapshot.trace_id, "bm25", bm25_hits, kept=1)
        self._store.insert_hits(snapshot.trace_id, "vector", vector_hits, kept=1)
        self._store.insert_hits(snapshot.trace_id, "rrf", rrf_hits, kept=1)
        self._store.insert_hits(snapshot.trace_id, "final", final_hits, kept=1)

        filtered_rows = []
        for rank, row in enumerate(filtered):
            filtered_rows.append({
                "chunk_id": row.get("chunk_id", ""),
                "doc_id": "",
                "title": "",
                "article_ref": None,
                "breadcrumb": None,
                "score": 0.0,
                "rank": rank,
                "filter_reason": row.get("reason", "unknown"),
            })
        self._store.insert_hits(
            snapshot.trace_id,
            "filtered",
            filtered_rows,
            kept=0,
            default_filter_reason="filtered",
        )

    def get_recent(self, limit: int = 50) -> list[dict]:
        return self._store.list_recent_traces(limit=limit)

    def get_trace_detail(self, trace_id: str) -> dict | None:
        run = self._store.get_trace(trace_id)
        if not run:
            return None

        events = self._store.get_events(trace_id)
        for ev in events:
            raw = ev.get("payload_json")
            try:
                ev["payload"] = json.loads(raw) if raw else {}
            except Exception:
                ev["payload"] = {}
            ev.pop("payload_json", None)

        grouped = self.get_search_detail(trace_id)
        citations = run.get("citations_json")
        used_context = run.get("used_context_json")
        try:
            run["citations"] = json.loads(citations) if citations else []
        except Exception:
            run["citations"] = []
        try:
            run["used_context"] = json.loads(used_context) if used_context else []
        except Exception:
            run["used_context"] = []

        return {
            "trace": run,
            "events": events,
            "retrieval": grouped,
        }

    def get_search_detail(self, trace_id: str) -> dict:
        hits = self._store.get_hits(trace_id)
        grouped: dict[str, list[dict]] = {
            "bm25_hits": [],
            "vector_hits": [],
            "rrf_hits": [],
            "final_hits": [],
            "filtered_out": [],
        }
        for row in hits:
            stage = row.get("stage")
            payload = {
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "title": row.get("title"),
                "article_ref": row.get("article_ref"),
                "breadcrumb": row.get("breadcrumb"),
                "score": row.get("score"),
                "rank": row.get("rank"),
            }
            if stage == "bm25":
                grouped["bm25_hits"].append(payload)
            elif stage == "vector":
                grouped["vector_hits"].append(payload)
            elif stage == "rrf":
                grouped["rrf_hits"].append(payload)
            elif stage == "final":
                grouped["final_hits"].append(payload)
            elif stage == "filtered":
                grouped["filtered_out"].append({
                    "chunk_id": payload["chunk_id"],
                    "reason": row.get("filter_reason"),
                    "rank": payload["rank"],
                })

        run = self._store.get_trace(trace_id) or {}
        return {
            "trace_id": trace_id,
            "query_raw": run.get("user_query"),
            "query_tokenized": run.get("query_tokenized"),
            "bm25_hits": grouped["bm25_hits"],
            "vector_hits": grouped["vector_hits"],
            "rrf_hits": grouped["rrf_hits"],
            "filtered_out": grouped["filtered_out"],
            "final_context": [h["chunk_id"] for h in grouped["final_hits"]],
            "prompt_version": run.get("prompt_version"),
            "retrieval_config_version": run.get("retrieval_config_version"),
            "index_version": run.get("index_version"),
            "model_name": run.get("model_name"),
        }
