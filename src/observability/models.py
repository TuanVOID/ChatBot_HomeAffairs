from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str = ""
    title: str = ""
    article_ref: str | None = None
    score: float = 0.0
    rank: int = 0
    source: str = ""
    breadcrumb: str | None = None


class FilteredHit(BaseModel):
    chunk_id: str
    reason: str
    detail: str | None = None


class RetrievalSnapshot(BaseModel):
    trace_id: str
    query_raw: str
    query_tokenized: str | None = None
    bm25_hits: list[SearchHit] = Field(default_factory=list)
    vector_hits: list[SearchHit] = Field(default_factory=list)
    rrf_hits: list[SearchHit] = Field(default_factory=list)
    filtered_out: list[FilteredHit] = Field(default_factory=list)
    final_context_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: dict[str, int] = Field(default_factory=dict)


class EvalCase(BaseModel):
    case_id: str
    question: str
    expected_doc_id: str | None = None
    expected_article_ref: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    gold_answer: str | None = None


class EvalResultRecord(BaseModel):
    run_id: str
    case_id: str
    trace_id: str | None = None
    hit_top5: int | None = None
    hit_top10: int | None = None
    rank_of_first_correct: int | None = None
    hit_expected_source: int | None = None
    hit_expected_article: int | None = None
    citation_correct: int | None = None
    grounded: int | None = None
    completeness_score: int | None = None
    hallucination: int | None = None
    retrieval_latency_ms: int | None = None
    answer_latency_ms: int | None = None
    total_latency_ms: int | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

