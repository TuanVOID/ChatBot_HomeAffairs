"""
Hybrid Retrieval Engine: BM25 + Vector + RRF fusion.
"""

from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import cfg
from src.retrieval.legal_query_parser import (
    extract_doc_short,
    extract_numeric_ref,
    normalize_chapter_ref,
    normalize_doc_number,
    parse_legal_refs,
)
from src.retrieval.text_tokenizer import tokenize_for_query

_INTRA_DOC_CONCEPT_PHRASES: dict[str, tuple[str, ...]] = {
    "pham_vi_dieu_chinh": (
        "pham vi dieu chinh",
        "doi tuong ap dung",
        "ap dung doi voi",
    ),
    "doi_tuong_ap_dung": (
        "doi tuong ap dung",
        "pham vi ap dung",
        "ap dung doi voi",
    ),
    "hieu_luc_thi_hanh": (
        "hieu luc thi hanh",
        "co hieu luc",
        "ngay co hieu luc",
    ),
    "trach_nhiem": (
        "trach nhiem",
        "co trach nhiem",
        "chiu trach nhiem",
    ),
    "nhiem_vu_quyen_han": (
        "nhiem vu quyen han",
        "quyen va nghia vu",
    ),
    "quyen_va_nghia_vu": (
        "quyen va nghia vu",
        "nghia vu cua",
        "quyen cua",
    ),
    "tham_quyen_xu_phat": (
        "tham quyen xu phat",
        "xu phat vi pham hanh chinh",
        "nguoi co tham quyen",
        "quyet dinh xu phat",
    ),
    "xu_ly_ky_luat": (
        "xu ly ky luat",
        "hinh thuc ky luat",
    ),
    "dieu_kien_tuyen_dung": (
        "dieu kien tuyen dung",
        "tieu chuan tuyen dung",
    ),
    "don_phuong_cham_dut_hop_dong": (
        "don phuong cham dut hop dong",
        "thoi han bao truoc",
    ),
    "trinh_tu_thu_tuc": (
        "trinh tu thu tuc",
        "ho so thu tuc",
    ),
    "thoi_han_giai_quyet": (
        "thoi han giai quyet",
        "thoi han xu ly",
    ),
}

_FOCUS_ACTION_HINT_TOKENS = {
    "tham",
    "quyen",
    "xu",
    "phat",
    "dieu",
    "kien",
    "tuyen",
    "dung",
    "cham",
    "dut",
    "quyen",
    "nghia",
    "vu",
    "trach",
    "nhiem",
    "hieu",
    "luc",
    "thu",
    "tuc",
    "trinh",
    "tu",
}

_FOCUS_ACTOR_HINT_TOKENS = {
    "cong",
    "chuc",
    "can",
    "bo",
    "vien",
    "lao",
    "dong",
    "nguoi",
    "co",
    "quan",
    "to",
    "chuc",
    "ca",
    "nhan",
}

_CONCEPT_BM25_CORE_TOP_K = 20
_CONCEPT_BM25_EXPANDED_TOP_K = 20
_CONCEPT_BM25_TITLE_TOP_K = 12
_CONCEPT_VECTOR_TOP_K = 20
_CONCEPT_MAX_CANDIDATE_DOCS = 6
_CONCEPT_MAX_CANDIDATE_FAMILIES = 3
_CONCEPT_MAX_FOCUS_CHUNKS = 24

_CONCEPT_DOC_ROLE_PRIOR: dict[str, list[str]] = {
    "pham_vi_dieu_chinh": ["law_core", "consolidated", "implementation", "implementation_sanction"],
    "dieu_kien_tuyen_dung": ["implementation", "consolidated", "law_core"],
    "xu_ly_ky_luat": ["implementation", "consolidated", "law_core"],
    "tham_quyen_xu_phat": ["implementation_sanction", "implementation", "consolidated", "law_core"],
    "don_phuong_cham_dut_hop_dong": ["law_core", "implementation", "consolidated"],
    "quyen_va_nghia_vu": ["law_core", "consolidated", "implementation"],
    "nhiem_vu_quyen_han": ["law_core", "implementation", "consolidated"],
    "hieu_luc_thi_hanh": ["consolidated", "law_core", "implementation"],
}

_TOPIC_DOC_ROLE_PRIOR: dict[str, list[str]] = {
    "labor": ["law_core", "implementation", "consolidated"],
    "civil_service": ["implementation", "law_core", "consolidated"],
    "administrative_sanction": ["implementation_sanction", "implementation", "consolidated", "law_core"],
    "social_insurance": ["implementation", "consolidated", "law_core"],
    "banking": ["law_core", "implementation", "consolidated"],
    "general_legal": ["law_core", "consolidated", "implementation"],
}

_IMPLEMENTATION_ROLES = {"implementation", "implementation_sanction", "consolidated"}

_GENERIC_DOC_TITLE_MARKERS = (
    "chuc nang nhiem vu quyen han",
    "co cau to chuc",
    "nhiem vu quyen han",
    "quy dinh chuc nang",
    "quy che lam viec",
)

_GARBAGE_FAMILY_PATTERNS = (
    r"\bdang nhap\b",
    r"\bdang ky\b",
    r"\bmat khau\b",
    r"\btin tuc\b",
    r"\blien he\b",
    r"\btrang chu\b",
    r"\bmenu\b",
    r"\blogin\b",
)

_FAMILY_LEGAL_MORPH_PATTERN = re.compile(
    r"\b(luat|bo\s+luat|nghi\s+dinh|thong\s+tu|vbhn|van\s+ban\s+hop\s+nhat)\b"
)


@dataclass
class CandidateDocScore:
    doc_id: str
    family_key: str
    doc_role: str = ""
    best_chunk_id: str | None = None
    bm25_core_rank: int | None = None
    bm25_expanded_rank: int | None = None
    bm25_title_rank: int | None = None
    vector_rank: int | None = None
    bm25_doc_score: float = 0.0
    vector_doc_score: float = 0.0
    title_anchor_score: float = 0.0
    concept_coverage_score: float = 0.0
    actor_action_score: float = 0.0
    doc_type_prior_score: float = 0.0
    doc_role_prior_score: float = 0.0
    wrong_doc_role_penalty: float = 0.0
    genericity_penalty: float = 0.0
    family_identity_score: float = 0.0
    concept_alignment_score: float = 0.0
    bm25_support_score: float = 0.0
    title_support_score: float = 0.0
    vector_support_score: float = 0.0
    family_support_source_count: int = 0
    final_doc_score: float = 0.0


class HybridRetriever:
    """
    Hybrid search combines BM25 (keyword) + vector search + RRF.
    Falls back to BM25-only when vector index is unavailable.
    """

    def __init__(
        self,
        bm25_index_dir: Path,
        vector_index_dir: Path,
        chunks_path: Path,
        ollama_url: str,
        embedding_model: str,
        rrf_k: int = 60,
        legal_lookup_service: Any | None = None,
        document_lookup_service: Any | None = None,
    ):
        from src.retrieval.bm25_searcher import BM25Searcher

        self._rrf_k = rrf_k
        self._bm25 = BM25Searcher(bm25_index_dir, chunks_path)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")

        self._vector = None
        try:
            if (vector_index_dir / "faiss.index").exists():
                from src.retrieval.vector_searcher import VectorSearcher

                self._vector = VectorSearcher(
                    vector_index_dir,
                    ollama_url,
                    embedding_model,
                    chunks_map=self._bm25.chunks_map,
                )
                logger.info("HybridRetriever: BM25 + Vector (FULL HYBRID)")
            else:
                logger.warning("HybridRetriever: Vector index missing -> BM25 ONLY mode")
        except Exception as exc:
            logger.warning(f"Vector searcher init failed: {exc} -> BM25 ONLY mode")

        self._legal_lookup = legal_lookup_service
        if self._legal_lookup is None:
            try:
                from src.retrieval.legal_lookup_service import LegalLookupService

                self._legal_lookup = LegalLookupService(self._bm25.chunks_map)
            except Exception as exc:
                self._legal_lookup = None
                logger.warning(f"LegalLookupService init failed: {exc}")

        self._document_lookup = document_lookup_service
        if self._document_lookup is None:
            try:
                from src.retrieval.document_lookup_service import DocumentLookupService

                available_doc_ids = {
                    str(row.get("doc_id", "")).strip()
                    for row in self._bm25.chunks_map.values()
                    if str(row.get("doc_id", "")).strip()
                }
                self._document_lookup = DocumentLookupService(available_doc_ids=available_doc_ids)
            except Exception as exc:
                self._document_lookup = None
                logger.warning(f"DocumentLookupService init failed: {exc}")

        self._query_rewrite = None
        if bool(getattr(cfg, "QUERY_REWRITE_ENABLED", True)):
            try:
                from src.retrieval.query_rewrite_service import QueryRewriteService

                self._query_rewrite = QueryRewriteService()
            except Exception as exc:
                self._query_rewrite = None
                logger.warning(f"QueryRewriteService init failed: {exc}")

    def search(
        self,
        query: str,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        final_top_k: int = 10,
    ) -> list[dict]:
        snapshot = self.search_with_snapshot(
            query=query,
            bm25_top_k=bm25_top_k,
            vector_top_k=vector_top_k,
            final_top_k=final_top_k,
        )
        return snapshot["final_results"]

    def search_with_snapshot(
        self,
        query: str,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        final_top_k: int = 10,
    ) -> dict:
        latencies_ms: dict[str, int] = {}
        t_total = time.perf_counter()

        refs = parse_legal_refs(query)
        route = self._decide_route(query, refs)
        initial_route = route
        effective_route = route

        document_lookup_used = False
        document_lookup_ms = 0
        document_lookup_matched = False
        document_lookup_match_type = ""
        document_lookup_confidence = 0.0
        document_lookup_doc_ids: list[str] = []
        document_lookup_rows: list[dict[str, Any]] = []
        document_lookup_fallback_to_loose = False
        document_lookup_top_matches: list[dict[str, Any]] = []
        document_lookup_distinct_top_matches: list[dict[str, Any]] = []
        document_lookup_selected_doc_numbers: list[str] = []
        document_lookup_top1_confidence = 0.0
        document_lookup_top2_confidence = 0.0
        document_lookup_margin = 0.0
        document_lookup_selected_route = ""
        document_lookup_selected_doc_ids: list[str] = []
        document_lookup_selected_reason = ""
        document_lookup_scope_reason_detail = ""
        document_lookup_candidate_doc_ids_original_count = 0
        document_lookup_candidate_doc_ids_trimmed = False
        document_lookup_parsed_law_name = ""
        document_lookup_parsed_topic_tail = ""
        document_lookup_law_specificity = 0.0
        document_lookup_soft_scope_applied = False
        low_specificity_generic = False
        very_low_specificity = False
        fallback_retry_used = False
        fallback_from_route = ""
        fallback_to_route = ""
        fallback_retry_reason = ""
        allowed_doc_ids: set[str] = set()
        query_for_bm25 = query
        query_for_vector = query
        query_rewrite_used = False
        query_rewrite_ms = 0
        query_rewrite_confidence = 0.0
        query_rewrite_too_vague = False
        query_rewrite_clean = ""
        query_rewrite_lexical = ""
        query_rewrite_semantic = ""
        query_rewrite_focus_terms: list[str] = []
        query_rewrite_fillers_removed: list[str] = []
        query_rewrite_phrase_repairs: list[str] = []
        query_rewrite_token_classes: dict[str, Any] = {}
        query_rewrite_legal_anchor_guess = ""
        query_rewrite_doc_type_prior: list[str] = []
        query_rewrite_exclude_doc_type_hint: list[str] = []
        query_rewrite_topic_class = ""
        query_rewrite_subclass = "unknown"
        query_rewrite_legal_concept_tags: list[str] = []
        query_rewrite_actor_terms: list[str] = []
        query_rewrite_action_terms: list[str] = []
        query_rewrite_object_terms: list[str] = []
        query_rewrite_qualifier_terms: list[str] = []
        query_rewrite_legal_anchor_guess_list: list[str] = []
        query_rewrite_vagueness_level = "none"
        query_rewrite_concept_confidence = 0.0
        query_rewrite_is_concept_query = False
        query_rewrite_is_topic_broad = False
        query_rewrite_v3_used = False
        query_rewrite_lexical_core = ""
        query_rewrite_lexical_expanded = ""
        query_rewrite_concept_seed_query = ""
        query_rewrite_title_anchor_query = ""
        query_rewrite_risk = "medium"
        query_rewrite_lexical_token_count = 0
        query_rewrite_lexical_is_weak = False
        query_rewrite_weak_query_abort = False
        query_rewrite_weak_query_abort_reasons: list[str] = []
        query_rewrite_lexical_quality_flags: list[str] = []
        query_rewrite_intent_template_hits: list[str] = []
        query_rewrite_lexical_expansion_used: list[str] = []
        hybrid_default_bm25_query_fallback_used = False
        hybrid_default_bm25_query_fallback_reason = ""
        hybrid_anchor_scope_used = False
        hybrid_anchor_scope_confidence = 0.0
        hybrid_anchor_scope_doc_ids: list[str] = []
        hybrid_anchor_scope_reason = ""
        hybrid_default_hard_doc_type_filter = False
        hybrid_default_hard_doc_type_allowlist: list[str] = []
        hybrid_default_scope_fallback_used = False
        hybrid_default_scope_fallback_reason = ""
        hybrid_default_scope_fallback_hits = 0
        hybrid_default_scope_fallback_bm25_only_used = False
        hybrid_default_scope_fallback_bm25_only_hits = 0
        hybrid_default_scope_fallback_bm25_only_latency_ms = 0
        hybrid_default_scope_fallback_bm25_only_query = ""
        hybrid_default_scope_fallback_bm25_only_query_reason = ""
        hybrid_default_scope_fallback_vector_reused = False
        hybrid_default_scope_fallback_second_vector_called = False
        hybrid_default_scope_fallback_first_pass_hits = 0
        hybrid_default_scope_fallback_first_pass_top10: list[dict[str, Any]] = []
        hybrid_default_concept_policy_applied = False
        hybrid_default_concept_vector_optional = False
        hybrid_default_intra_doc_focus_rerank_used = False
        hybrid_default_intra_doc_focus_target_doc_ids: list[str] = []
        hybrid_default_intra_doc_focus_target_family_keys: list[str] = []
        hybrid_default_intra_doc_focus_rank_first_doc = 0
        hybrid_default_intra_doc_focus_rank_first_family = 0
        hybrid_default_intra_doc_focus_candidates = 0
        hybrid_default_intra_doc_focus_promoted = 0
        hybrid_default_mode = "legacy"
        hybrid_default_candidate_family_count = 0
        hybrid_default_candidate_doc_count = 0
        hybrid_default_title_anchor_used = False
        hybrid_default_title_anchor_hits = 0
        hybrid_default_doc_aggregation_used = False
        hybrid_default_doc_score_top1 = 0.0
        hybrid_default_doc_score_top2 = 0.0
        hybrid_default_doc_score_margin = 0.0
        hybrid_default_doc_top1_family = ""
        hybrid_default_doc_top2_family = ""
        hybrid_default_vector_used_for_doc_recall = False
        hybrid_default_vector_support_same_family = False
        hybrid_default_genericity_penalty_top1 = 0.0
        hybrid_default_genericity_penalty_applied_count = 0
        hybrid_default_focus_rerank_used = False
        hybrid_default_focus_rerank_candidate_chunks = 0
        hybrid_default_focus_heading_match_top1 = 0.0
        hybrid_default_focus_actor_action_top1 = 0.0
        hybrid_default_focus_concept_match_top1 = 0.0
        hybrid_default_selected_family_keys: list[str] = []
        hybrid_default_selected_doc_ids: list[str] = []
        hybrid_default_family_search_used = False
        hybrid_default_family_search_hits = 0
        hybrid_default_family_candidate_count_pre_filter = 0
        hybrid_default_family_candidate_count_post_filter = 0
        hybrid_default_family_score_top1 = 0.0
        hybrid_default_family_score_top2 = 0.0
        hybrid_default_family_score_margin = 0.0
        hybrid_default_family_top1_doc_role = ""
        hybrid_default_family_top1_is_implementation = False
        hybrid_default_family_recovery_used = False
        hybrid_default_family_recovery_promoted = 0
        hybrid_default_doc_role_prior_applied = False
        hybrid_default_doc_role_prior_top1 = ""
        hybrid_default_wrong_doc_role_penalty_applied = False
        hybrid_default_implementation_bridge_used = False
        hybrid_default_implementation_bridge_hits = 0
        hybrid_default_weak_query_abort_used = False
        hybrid_default_no_confident_family_candidates = False
        hybrid_default_garbage_family_rejected = False
        hybrid_default_garbage_family_reject_reason = ""
        hybrid_default_family_identity_score_top1 = 0.0
        hybrid_default_family_identity_score_top2 = 0.0
        hybrid_default_family_identity_margin = 0.0
        hybrid_default_trusted_family = False
        hybrid_default_shortlist_after_reject_count = 0
        hybrid_default_shortlist_after_reject_empty = False
        hybrid_default_focus_rerank_blocked_by_low_family_confidence = False
        hybrid_default_focus_rerank_stage_entered_with_family_confidence = 0.0
        rank_first_family_pre_focus = 0
        rank_first_family_post_focus = 0
        rank_first_doc_role_pre_focus = ""
        rank_first_doc_role_post_focus = ""
        candidate_route_early_exit_used = False
        candidate_route_early_exit_reason = ""
        candidate_route_early_exit_score_gap = 0.0

        if route == "law_anchored_hybrid":
            if refs.get("law_name") and self._document_lookup is not None:
                document_lookup_used = True
                t_doc_lookup = time.perf_counter()
                lookup = self._document_lookup.match_law_name(
                    str(refs.get("law_name", "")),
                    top_k=max(1, int(getattr(cfg, "DOCUMENT_LOOKUP_TOP_K", 5))),
                )
                document_lookup_ms = round((time.perf_counter() - t_doc_lookup) * 1000)
                latencies_ms["document_lookup"] = int(document_lookup_ms)

                document_lookup_matched = bool(lookup.get("matched", False))
                document_lookup_match_type = str(lookup.get("match_type", ""))
                document_lookup_confidence = float(lookup.get("confidence", 0.0))
                document_lookup_parsed_law_name = str(lookup.get("parsed_law_name", ""))
                document_lookup_parsed_topic_tail = str(lookup.get("parsed_topic_tail", ""))
                document_lookup_law_specificity = float(lookup.get("law_specificity", 0.0))
                document_lookup_doc_ids = [
                    str(doc_id)
                    for doc_id in (lookup.get("doc_ids") or [])
                    if str(doc_id).strip()
                ]
                document_lookup_rows = list(lookup.get("distinct_rows") or lookup.get("rows") or [])[:8]
                document_lookup_top_matches = [
                    {
                        "doc_id": str(row.get("doc_id", "")),
                        "confidence": float(row.get("confidence", 0.0)),
                        "match_type": str(row.get("match_type", "")),
                        "document_number": str(row.get("document_number", "")),
                        "title": str(row.get("title", "")),
                        "canonical_law_key": str(row.get("canonical_law_key", "")),
                    }
                    for row in document_lookup_rows[:5]
                ]
                seen_law_titles: set[str] = set()
                for match in document_lookup_top_matches:
                    title_key = " ".join(tokenize_for_query(str(match.get("title", ""))).split())
                    doc_num_key = normalize_doc_number(str(match.get("document_number", "")))
                    key = f"{title_key}|{doc_num_key}" if title_key else str(match.get("doc_id", ""))
                    if key in seen_law_titles:
                        continue
                    seen_law_titles.add(key)
                    document_lookup_distinct_top_matches.append(dict(match))

                if not document_lookup_distinct_top_matches:
                    document_lookup_distinct_top_matches = list(document_lookup_top_matches)

                if document_lookup_distinct_top_matches:
                    document_lookup_top1_confidence = float(document_lookup_distinct_top_matches[0].get("confidence", 0.0))
                if len(document_lookup_distinct_top_matches) >= 2:
                    document_lookup_top2_confidence = float(document_lookup_distinct_top_matches[1].get("confidence", 0.0))
                document_lookup_margin = max(0.0, document_lookup_top1_confidence - document_lookup_top2_confidence)

                exact_threshold = float(getattr(cfg, "DOCUMENT_LOOKUP_EXACT_CONFIDENCE_THRESHOLD", 0.92))
                candidate_threshold = float(getattr(cfg, "DOCUMENT_LOOKUP_CANDIDATE_CONFIDENCE_THRESHOLD", 0.75))
                margin_threshold = float(getattr(cfg, "DOCUMENT_LOOKUP_MIN_MARGIN", 0.10))
                margin_epsilon = max(0.0, float(getattr(cfg, "DOCUMENT_LOOKUP_MARGIN_EPSILON", 0.01)))
                candidate_max_docs = max(1, int(getattr(cfg, "DOCUMENT_LOOKUP_MAX_CANDIDATE_DOCS", 3)))
                top1_match_type = (
                    str(document_lookup_distinct_top_matches[0].get("match_type", ""))
                    if document_lookup_distinct_top_matches
                    else ""
                )
                top1_exactish = top1_match_type in {"exact_title_alias", "token_overlap_exactish"} or top1_match_type.startswith("exact")
                law_name_norm = tokenize_for_query(str(refs.get("law_name", "")))
                generic_law_name = bool(re.match(r"^(luat|bo luat)\s+ve\b", law_name_norm))
                has_equal_top = abs(document_lookup_top1_confidence - document_lookup_top2_confidence) <= margin_epsilon
                low_or_zero_margin = document_lookup_margin <= max(margin_epsilon, margin_threshold * 0.25)
                clean_law_name = bool(document_lookup_parsed_law_name) and not bool(document_lookup_parsed_topic_tail)
                multi_doc_candidates = len(document_lookup_doc_ids) > 1
                # "luat ve X" queries are often short; keep them eligible for scoped candidate retrieval
                # unless specificity is truly low.
                low_specificity_generic = generic_law_name and document_lookup_law_specificity < 0.60
                very_low_specificity = document_lookup_law_specificity < 0.50

                if document_lookup_matched and document_lookup_doc_ids:
                    if (
                        document_lookup_top1_confidence >= exact_threshold
                        and top1_exactish
                        and not has_equal_top
                        and document_lookup_margin >= margin_threshold
                        and not multi_doc_candidates
                        and clean_law_name
                        and document_lookup_law_specificity >= 0.75
                    ):
                        effective_route = "doc_scoped_hybrid_exact_doc"
                        document_lookup_selected_reason = "exact_confidence_margin"
                        document_lookup_selected_doc_ids = [document_lookup_doc_ids[0]]
                        document_lookup_scope_reason_detail = (
                            f"top1={document_lookup_top1_confidence:.3f},"
                            f"top2={document_lookup_top2_confidence:.3f},"
                            f"margin={document_lookup_margin:.3f},"
                            f"specificity={document_lookup_law_specificity:.2f}"
                        )
                        allowed_doc_ids = set(document_lookup_selected_doc_ids)
                    elif (
                        document_lookup_top1_confidence >= candidate_threshold
                        and not low_specificity_generic
                    ):
                        effective_route = "doc_scoped_hybrid_candidate_docs"
                        if has_equal_top or low_or_zero_margin:
                            document_lookup_selected_reason = "candidate_low_margin_or_tie"
                        elif multi_doc_candidates:
                            document_lookup_selected_reason = "candidate_multiple_docs"
                        else:
                            document_lookup_selected_reason = "candidate_confidence"
                        document_lookup_selected_doc_ids = document_lookup_doc_ids[:candidate_max_docs]
                        document_lookup_candidate_doc_ids_original_count = len(document_lookup_selected_doc_ids)
                        trim_top1_conf = float(
                            getattr(cfg, "DOC_SCOPED_CANDIDATE_TRIM_TOP1_CONFIDENCE", 0.90)
                        )
                        trim_top1_margin = float(
                            getattr(cfg, "DOC_SCOPED_CANDIDATE_TRIM_TOP1_MARGIN", 0.05)
                        )
                        trim_top2_conf = float(
                            getattr(cfg, "DOC_SCOPED_CANDIDATE_TRIM_TOP2_CONFIDENCE", 0.84)
                        )
                        trim_top2_margin = float(
                            getattr(cfg, "DOC_SCOPED_CANDIDATE_TRIM_TOP2_MARGIN", 0.02)
                        )
                        if (
                            not has_equal_top
                            and document_lookup_top1_confidence >= trim_top1_conf
                            and document_lookup_margin >= trim_top1_margin
                            and len(document_lookup_selected_doc_ids) > 1
                        ):
                            document_lookup_selected_doc_ids = document_lookup_selected_doc_ids[:1]
                            document_lookup_candidate_doc_ids_trimmed = True
                            document_lookup_selected_reason = (
                                f"{document_lookup_selected_reason}|candidate_trim_top1"
                            )
                        elif (
                            not has_equal_top
                            and document_lookup_top1_confidence >= trim_top2_conf
                            and document_lookup_margin >= trim_top2_margin
                            and len(document_lookup_selected_doc_ids) > 2
                        ):
                            document_lookup_selected_doc_ids = document_lookup_selected_doc_ids[:2]
                            document_lookup_candidate_doc_ids_trimmed = True
                            document_lookup_selected_reason = (
                                f"{document_lookup_selected_reason}|candidate_trim_top2"
                            )
                        document_lookup_scope_reason_detail = (
                            f"top1={document_lookup_top1_confidence:.3f},"
                            f"top2={document_lookup_top2_confidence:.3f},"
                            f"margin={document_lookup_margin:.3f},"
                            f"docs={len(document_lookup_doc_ids)}"
                        )
                        allowed_doc_ids = set(document_lookup_selected_doc_ids)
                    else:
                        effective_route = "law_anchored_hybrid_loose"
                        document_lookup_fallback_to_loose = True
                        if document_lookup_top1_confidence < candidate_threshold:
                            document_lookup_selected_reason = "confidence_too_low"
                        elif low_specificity_generic:
                            document_lookup_selected_reason = "specificity_too_low"
                        else:
                            document_lookup_selected_reason = "margin_too_low"
                        document_lookup_scope_reason_detail = (
                            f"top1={document_lookup_top1_confidence:.3f},"
                            f"top2={document_lookup_top2_confidence:.3f},"
                            f"margin={document_lookup_margin:.3f},"
                            f"top1_match={top1_match_type}"
                        )
                else:
                    effective_route = "law_anchored_hybrid_loose"
                    document_lookup_fallback_to_loose = True
                    document_lookup_selected_reason = "no_match"
                    document_lookup_scope_reason_detail = "lookup returned no matched doc ids"
            else:
                effective_route = "law_anchored_hybrid_loose"
                document_lookup_selected_reason = "lookup_unavailable_or_no_law_name"
                document_lookup_scope_reason_detail = "no law_name or lookup service unavailable"

            if (
                effective_route == "law_anchored_hybrid_loose"
                and document_lookup_matched
                and document_lookup_confidence >= float(
                    getattr(cfg, "LAW_ANCHORED_LOOSE_SOFT_SCOPE_CONFIDENCE", 0.78)
                )
                and document_lookup_doc_ids
                and not (low_specificity_generic and len(document_lookup_doc_ids) <= 1)
                and not very_low_specificity
            ):
                soft_top_k = max(1, int(getattr(cfg, "LAW_ANCHORED_LOOSE_SOFT_SCOPE_TOP_K", 5)))
                document_lookup_soft_scope_applied = True
                document_lookup_selected_doc_ids = document_lookup_doc_ids[:soft_top_k]
                if not document_lookup_selected_reason:
                    document_lookup_selected_reason = "loose_soft_scope"
                document_lookup_scope_reason_detail = (
                    f"soft_scope_docs={len(document_lookup_selected_doc_ids)},"
                    f"confidence={document_lookup_confidence:.3f}"
                )
                allowed_doc_ids = set(document_lookup_selected_doc_ids)
            elif (
                effective_route == "law_anchored_hybrid_loose"
                and (low_specificity_generic or very_low_specificity)
            ):
                # Avoid hard-scoping generic "luat ve ..." lookups to a single, often wrong document.
                document_lookup_soft_scope_applied = False
                document_lookup_selected_doc_ids = []
                allowed_doc_ids = set()
                if not document_lookup_selected_reason:
                    document_lookup_selected_reason = "loose_no_scope_low_specificity"
                detail_suffix = (
                    f"specificity={document_lookup_law_specificity:.2f},"
                    f"doc_ids={len(document_lookup_doc_ids)}"
                )
                if document_lookup_scope_reason_detail:
                    document_lookup_scope_reason_detail = (
                        f"{document_lookup_scope_reason_detail},{detail_suffix}"
                    )
                else:
                    document_lookup_scope_reason_detail = detail_suffix

            if (
                effective_route == "law_anchored_hybrid_loose"
                and document_lookup_confidence >= 0.90
                and document_lookup_law_specificity >= 0.80
                and len(document_lookup_doc_ids) == 1
            ):
                # Promote strong single-doc law anchors into scoped candidate retrieval.
                effective_route = "doc_scoped_hybrid_candidate_docs"
                document_lookup_selected_reason = "loose_promoted_single_doc_confident"
                document_lookup_selected_doc_ids = [document_lookup_doc_ids[0]]
                allowed_doc_ids = set(document_lookup_selected_doc_ids)

            document_lookup_selected_route = effective_route
            lookup_doc_num_map = {
                str(row.get("doc_id", "")).strip(): str(row.get("document_number", "")).strip()
                for row in document_lookup_rows
                if str(row.get("doc_id", "")).strip()
            }
            document_lookup_selected_doc_numbers = [
                lookup_doc_num_map.get(doc_id, "")
                for doc_id in document_lookup_selected_doc_ids
            ]

        if effective_route == "hybrid_default" and self._query_rewrite is not None:
            query_rewrite_used = True
            t_rewrite = time.perf_counter()
            rewritten = self._query_rewrite.rewrite(query, route=effective_route)
            query_rewrite_ms = round((time.perf_counter() - t_rewrite) * 1000)
            latencies_ms["query_rewrite"] = int(query_rewrite_ms)
            query_rewrite_clean = str(rewritten.get("query_clean", ""))
            query_rewrite_lexical = str(rewritten.get("lexical_query", ""))
            query_rewrite_semantic = str(rewritten.get("semantic_query", ""))
            query_rewrite_focus_terms = [str(t) for t in (rewritten.get("focus_terms") or []) if str(t).strip()]
            query_rewrite_fillers_removed = [str(t) for t in (rewritten.get("fillers_removed") or []) if str(t).strip()]
            query_rewrite_phrase_repairs = [str(t) for t in (rewritten.get("phrase_repairs") or []) if str(t).strip()]
            query_rewrite_token_classes = dict(rewritten.get("token_classes") or {})
            raw_anchor_guess = rewritten.get("legal_anchor_guess")
            if isinstance(raw_anchor_guess, list):
                query_rewrite_legal_anchor_guess_list = [
                    str(v).strip()
                    for v in raw_anchor_guess
                    if str(v).strip()
                ]
            else:
                query_rewrite_legal_anchor_guess_list = []
            if not query_rewrite_legal_anchor_guess_list:
                query_rewrite_legal_anchor_guess_list = [
                    str(v).strip()
                    for v in (rewritten.get("legal_anchor_guess_list") or [])
                    if str(v).strip()
                ]
            anchor_guess_text = str(rewritten.get("legal_anchor_guess_text", "")).strip()
            if not anchor_guess_text and query_rewrite_legal_anchor_guess_list:
                anchor_guess_text = query_rewrite_legal_anchor_guess_list[0]
            if not anchor_guess_text and isinstance(raw_anchor_guess, str):
                anchor_guess_text = str(raw_anchor_guess).strip()
            query_rewrite_legal_anchor_guess = anchor_guess_text
            query_rewrite_doc_type_prior = [
                str(t)
                for t in (rewritten.get("doc_type_prior") or [])
                if str(t).strip()
            ]
            query_rewrite_exclude_doc_type_hint = [
                str(t)
                for t in (rewritten.get("exclude_doc_type_hint") or [])
                if str(t).strip()
            ]
            query_rewrite_topic_class = str(rewritten.get("topic_class", ""))
            query_rewrite_subclass = str(rewritten.get("query_subclass", "unknown"))
            query_rewrite_legal_concept_tags = [
                str(v)
                for v in (rewritten.get("legal_concept_tags") or [])
                if str(v).strip()
            ]
            query_rewrite_actor_terms = [
                str(v)
                for v in (rewritten.get("actor_terms") or [])
                if str(v).strip()
            ]
            query_rewrite_action_terms = [
                str(v)
                for v in (rewritten.get("action_terms") or [])
                if str(v).strip()
            ]
            query_rewrite_object_terms = [
                str(v)
                for v in (rewritten.get("object_terms") or [])
                if str(v).strip()
            ]
            query_rewrite_qualifier_terms = [
                str(v)
                for v in (rewritten.get("qualifier_terms") or [])
                if str(v).strip()
            ]
            query_rewrite_concept_confidence = float(rewritten.get("concept_confidence", 0.0))
            query_rewrite_vagueness_level = str(rewritten.get("vagueness_level", "none"))
            query_rewrite_is_concept_query = bool(rewritten.get("is_concept_query", False))
            query_rewrite_is_topic_broad = bool(rewritten.get("is_topic_broad", False))
            query_rewrite_v3_used = bool(
                rewritten.get("query_rewrite_v3_used", True)
                or rewritten.get("lexical_core")
                or rewritten.get("lexical_expanded")
            )
            query_rewrite_lexical_core = str(rewritten.get("lexical_core", ""))
            query_rewrite_lexical_expanded = str(rewritten.get("lexical_expanded", ""))
            query_rewrite_concept_seed_query = str(rewritten.get("concept_seed_query", ""))
            query_rewrite_title_anchor_query = str(rewritten.get("title_anchor_query", ""))
            query_rewrite_risk = str(rewritten.get("rewrite_risk", "medium"))
            query_rewrite_confidence = float(rewritten.get("rewrite_confidence", 0.0))
            query_rewrite_too_vague = bool(rewritten.get("query_too_vague", False))
            query_rewrite_lexical_is_weak = bool(rewritten.get("lexical_is_weak", False))
            query_rewrite_weak_query_abort = bool(rewritten.get("weak_query_abort", False))
            query_rewrite_weak_query_abort_reasons = [
                str(v)
                for v in (rewritten.get("weak_query_abort_reasons") or [])
                if str(v).strip()
            ]
            query_rewrite_lexical_quality_flags = [
                str(v)
                for v in (rewritten.get("lexical_quality_flags") or [])
                if str(v).strip()
            ]
            query_rewrite_intent_template_hits = [
                str(v)
                for v in (rewritten.get("intent_template_hits") or [])
                if str(v).strip()
            ]
            query_rewrite_lexical_expansion_used = [
                str(v)
                for v in (rewritten.get("lexical_expansion_used") or [])
                if str(v).strip()
            ]
            query_rewrite_lexical_token_count = len(tokenize_for_query(query_rewrite_lexical).split())
            if not query_rewrite_lexical_core:
                query_rewrite_lexical_core = query_rewrite_lexical
            if not query_rewrite_lexical_expanded:
                query_rewrite_lexical_expanded = query_rewrite_lexical
            if not query_rewrite_concept_seed_query:
                query_rewrite_concept_seed_query = query_rewrite_lexical_core or query_rewrite_lexical
            if not query_rewrite_title_anchor_query:
                query_rewrite_title_anchor_query = " ".join(
                    tokenize_for_query(
                        " ".join(query_rewrite_legal_anchor_guess_list + query_rewrite_legal_concept_tags)
                    ).split()[:12]
                )
            if query_rewrite_weak_query_abort and query_rewrite_concept_seed_query:
                query_for_bm25 = query_rewrite_concept_seed_query
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "weak_query_abort_concept_seed"
                hybrid_default_weak_query_abort_used = True
            elif query_rewrite_subclass == "concept_generic" and query_rewrite_lexical:
                query_for_bm25 = query_rewrite_lexical
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "concept_generic_expanded_lexical"
            elif query_rewrite_too_vague:
                query_for_bm25 = query_rewrite_clean or query
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "query_too_vague"
            elif query_rewrite_lexical_is_weak and query_rewrite_lexical and query_rewrite_lexical_token_count >= 4:
                query_for_bm25 = query_rewrite_lexical
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "lexical_weak_expanded"
            elif query_rewrite_lexical_is_weak and query_rewrite_clean:
                query_for_bm25 = query_rewrite_clean
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "lexical_weak_clean"
            elif query_rewrite_lexical_token_count <= 2 and query_rewrite_clean:
                query_for_bm25 = query_rewrite_clean
                hybrid_default_bm25_query_fallback_used = True
                hybrid_default_bm25_query_fallback_reason = "lexical_too_short"
            else:
                query_for_bm25 = query_rewrite_lexical or query_rewrite_clean or query
            query_for_vector = query_rewrite_semantic or query_for_bm25
            if self._document_lookup is not None and query_rewrite_legal_anchor_guess:
                t_anchor_scope = time.perf_counter()
                lookup = self._document_lookup.match_law_name(
                    query_rewrite_legal_anchor_guess,
                    top_k=max(1, int(getattr(cfg, "HYBRID_DEFAULT_ANCHOR_SCOPE_TOP_K", 5))),
                    min_confidence=float(
                        getattr(cfg, "HYBRID_DEFAULT_ANCHOR_SCOPE_MIN_CONFIDENCE", 0.65)
                    ),
                )
                latencies_ms["hybrid_anchor_scope_lookup"] = round(
                    (time.perf_counter() - t_anchor_scope) * 1000
                )
                hybrid_anchor_scope_confidence = float(lookup.get("confidence", 0.0))
                anchor_scope_conf_threshold = float(
                    getattr(cfg, "HYBRID_DEFAULT_ANCHOR_SCOPE_CONFIDENCE", 0.86)
                )
                if bool(lookup.get("matched", False)) and hybrid_anchor_scope_confidence >= anchor_scope_conf_threshold:
                    candidate_max_docs = max(
                        1,
                        int(getattr(cfg, "HYBRID_DEFAULT_ANCHOR_SCOPE_MAX_DOCS", 3)),
                    )
                    hybrid_anchor_scope_doc_ids = [
                        str(doc_id).strip()
                        for doc_id in (lookup.get("doc_ids") or [])[:candidate_max_docs]
                        if str(doc_id).strip()
                    ]
                    if hybrid_anchor_scope_doc_ids:
                        allowed_doc_ids = set(hybrid_anchor_scope_doc_ids)
                        hybrid_anchor_scope_used = True
                        hybrid_anchor_scope_reason = "rewrite_legal_anchor_lookup"
                if not hybrid_anchor_scope_used:
                    hybrid_anchor_scope_reason = "lookup_not_confident_enough"

        bm25_mode = self._bm25_mode_for_route(effective_route)
        bm25_exec_top_k = int(bm25_top_k)
        vector_top_k_effective = int(vector_top_k)
        if effective_route == "hybrid_default":
            bm25_exec_top_k = max(
                int(bm25_top_k),
                int(getattr(cfg, "HYBRID_DEFAULT_BM25_TOP_K", 60)),
            )
        elif effective_route == "doc_scoped_hybrid_candidate_docs":
            bm25_exec_top_k = max(
                6,
                min(
                    int(bm25_top_k),
                    int(getattr(cfg, "DOC_SCOPED_CANDIDATE_BM25_TOP_K", 10)),
                ),
            )
            vector_top_k_effective = max(
                6,
                min(
                    int(vector_top_k),
                    int(getattr(cfg, "DOC_SCOPED_CANDIDATE_VECTOR_TOP_K", 8)),
                ),
            )

        vector_available = bool(self._vector and self._vector.available)
        executed_parallel = False
        vector_skipped_by_route = False
        vector_skipped_by_quality_gate = False
        vector_timed_out = False
        vector_timeout_ms = max(300, int(getattr(cfg, "VECTOR_TIMEOUT_MS", 2500)))
        vector_budget_ms = int(vector_timeout_ms)
        if effective_route == "doc_scoped_hybrid_candidate_docs":
            vector_timeout_ms = min(
                vector_timeout_ms,
                max(500, int(getattr(cfg, "DOC_SCOPED_CANDIDATE_VECTOR_TIMEOUT_MS", 1100))),
            )
            vector_budget_ms = int(vector_timeout_ms)
        vector_wall_ms = 0
        vector_deadline_hit = False
        vector_result_used = False
        vector_cancelled = False
        vector_join_wait_ms = 0
        hybrid_doc_gate_applied = False
        hybrid_doc_gate_bm25_before = 0
        hybrid_doc_gate_bm25_after = 0
        hybrid_doc_gate_vector_before = 0
        hybrid_doc_gate_vector_after = 0
        hybrid_doc_gate_removed_bm25 = 0
        hybrid_doc_gate_removed_vector = 0
        hybrid_doc_gate_override_kept_original = False
        hybrid_doc_gate_bm25_filtered_before_override = 0
        hybrid_doc_gate_vector_filtered_before_override = 0

        structured_lookup_used = False
        structured_lookup_hit_count = 0
        structured_lookup_fallback = False
        structured_lookup_match_type = ""
        structured_lookup_confidence = 0.0
        structured_lookup_candidates = 0

        bm25_debug: dict[str, Any] = {}
        bm25_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []

        if route == "structured_exact" and self._legal_lookup is not None:
            structured_lookup_used = True
            t_lookup = time.perf_counter()
            bm25_results = self._legal_lookup.lookup(
                refs,
                top_k=max(final_top_k, bm25_top_k),
                include_text=False,
            )
            latencies_ms["structured_lookup"] = round((time.perf_counter() - t_lookup) * 1000)
            structured_lookup_hit_count = len(bm25_results)

            if bm25_results:
                vector_skipped_by_route = True
                bm25_mode = "structured_lookup"
                top_lookup = bm25_results[0]
                structured_lookup_match_type = str(top_lookup.get("match_type", "doc_only_exact"))
                structured_lookup_confidence = float(top_lookup.get("match_confidence", 0.0))
                structured_lookup_candidates = len(bm25_results)
                bm25_debug = {
                    "mode": "structured_lookup",
                    "top_k": len(bm25_results),
                    "tokenized_query": tokenize_for_query(query),
                    "parsed_query": "structured_lookup",
                    "query_fields": [],
                    "is_multifield": False,
                    "include_text": False,
                    "legal_refs": refs,
                    "results_count": len(bm25_results),
                    "open_index_ms": 0,
                    "open_searcher_ms": 0,
                    "build_query_ms": 0,
                    "execute_search_ms": int(latencies_ms["structured_lookup"]),
                    "hydrate_hits_ms": 0,
                    "structured_lookup_match_type": structured_lookup_match_type,
                    "structured_lookup_confidence": structured_lookup_confidence,
                    "structured_lookup_candidates": structured_lookup_candidates,
                }
                latencies_ms["bm25"] = int(latencies_ms["structured_lookup"])
                latencies_ms["vector"] = 0
            else:
                structured_lookup_fallback = True
                effective_route = "narrow_bm25"
                bm25_mode = self._bm25_mode_for_route(effective_route)

        if effective_route == "hybrid_default":
            prior_buckets = {
                str(v).strip().lower()
                for v in query_rewrite_doc_type_prior
                if str(v).strip()
            }
            hybrid_default_hard_doc_type_filter = bool(
                query_rewrite_too_vague or query_rewrite_lexical_is_weak
            )
            topic_sensitive = {"civil_service", "labor", "administrative_sanction"}
            if query_rewrite_topic_class in topic_sensitive:
                hard_allow = {"bo_luat", "luat", "vbhn"}
            else:
                hard_allow = {"bo_luat", "luat", "vbhn", "nghi_dinh"}
            query_for_doc_hint = tokenize_for_query(" ".join([query, query_for_bm25]))
            if (
                "nghi_dinh" in prior_buckets
                and ("nghi dinh" in query_for_doc_hint or query_rewrite_topic_class not in topic_sensitive)
            ):
                hard_allow.add("nghi_dinh")
            if (
                "thong_tu" in prior_buckets
                and ("thong tu" in query_for_doc_hint or query_rewrite_topic_class in {"labor", "administrative_sanction"})
            ):
                hard_allow.add("thong_tu")
            if query_rewrite_subclass == "concept_generic":
                hybrid_default_concept_policy_applied = True
                hybrid_default_concept_vector_optional = True
                hybrid_default_hard_doc_type_filter = True
                hard_allow = {"luat", "bo_luat", "nghi_dinh", "vbhn"}
                if "thong_tu" in prior_buckets and query_rewrite_topic_class in {"labor", "administrative_sanction"}:
                    hard_allow.add("thong_tu")
                vector_timeout_ms = min(vector_timeout_ms, 1400)
                if query_rewrite_risk == "high":
                    vector_skipped_by_quality_gate = True
                vector_budget_ms = int(vector_timeout_ms)
            if query_rewrite_is_concept_query:
                hybrid_default_mode = "concept_aware"
                hybrid_default_concept_policy_applied = True
                hybrid_default_concept_vector_optional = True
                # For broad concept queries, keep doc-type prior soft and score-based.
                hybrid_default_hard_doc_type_filter = False
                hard_allow = set(query_rewrite_doc_type_prior or ["bo_luat", "luat", "vbhn", "nghi_dinh"])
                if query_rewrite_vagueness_level == "hard":
                    vector_skipped_by_quality_gate = True
                elif query_rewrite_concept_confidence < 0.55 and query_rewrite_lexical_is_weak:
                    vector_timeout_ms = min(vector_timeout_ms, 1200)
                vector_budget_ms = int(vector_timeout_ms)
            hybrid_default_hard_doc_type_allowlist = sorted(hard_allow)
            if query_rewrite_is_concept_query:
                if query_rewrite_vagueness_level == "hard":
                    vector_skipped_by_quality_gate = True
                elif query_rewrite_confidence < 0.75 and not query_rewrite_legal_anchor_guess:
                    vector_timeout_ms = min(vector_timeout_ms, 1500)
                    vector_budget_ms = int(vector_timeout_ms)
            else:
                if query_rewrite_too_vague or query_rewrite_lexical_is_weak:
                    vector_skipped_by_quality_gate = True
                elif query_rewrite_confidence < 0.75 and not query_rewrite_legal_anchor_guess:
                    vector_timeout_ms = min(vector_timeout_ms, 1500)
                    vector_budget_ms = int(vector_timeout_ms)

        def _run_bm25() -> tuple[list[dict], dict, int]:
            t0 = time.perf_counter()
            payload = self._bm25.search(
                query_for_bm25,
                bm25_exec_top_k,
                return_debug=True,
                mode=bm25_mode,
                legal_refs=refs,
                allowed_doc_ids=allowed_doc_ids or None,
                include_text=False,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            if isinstance(payload, dict):
                return (
                    list(payload.get("results", [])),
                    dict(payload.get("debug", {})),
                    elapsed_ms,
                )
            return list(payload or []), {}, elapsed_ms

        def _run_vector() -> dict[str, Any]:
            t0 = time.perf_counter()
            rows: list[dict] = []
            if vector_available and not vector_skipped_by_route:
                try:
                    rows = self._vector.search(
                        query_for_vector,
                        vector_top_k_effective,
                        allowed_doc_ids=allowed_doc_ids or None,
                    )
                except Exception as exc:
                    logger.warning(f"Vector search failed: {exc}")
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return {
                "rows": rows,
                "wall_ms": int(elapsed_ms),
            }

        if not bm25_results:
            concept_hybrid_default_enabled = (
                effective_route == "hybrid_default"
            )
            if concept_hybrid_default_enabled:
                vague_recovery_allowed = self._can_use_vague_concept_recovery(
                    lexical_core=query_rewrite_lexical_core or query_rewrite_lexical,
                    concept_tags=query_rewrite_legal_concept_tags,
                    actor_terms=query_rewrite_actor_terms,
                    action_terms=query_rewrite_action_terms,
                    object_terms=query_rewrite_object_terms,
                    title_anchor_query=query_rewrite_title_anchor_query,
                    weak_query_abort=query_rewrite_weak_query_abort,
                )
                vague_gate_enabled = bool(
                    getattr(cfg, "HYBRID_DEFAULT_V15_VAGUE_GATE_ENABLED", True)
                )
                if not vague_gate_enabled:
                    vague_recovery_allowed = True

                if query_rewrite_is_concept_query and not query_rewrite_weak_query_abort:
                    hybrid_default_mode = "concept_aware"
                elif vague_recovery_allowed:
                    hybrid_default_mode = "vague_concept_recovery"
                else:
                    hybrid_default_mode = "concept_aware_low_confidence"
                hybrid_default_concept_policy_applied = True
                hybrid_default_concept_vector_optional = True
                hybrid_default_doc_aggregation_used = True
                hybrid_default_weak_query_abort_used = bool(
                    query_rewrite_weak_query_abort
                    or (not query_rewrite_is_concept_query and not vague_recovery_allowed)
                )
                hybrid_default_family_search_used = not bool(
                    query_rewrite_weak_query_abort and vague_gate_enabled
                )
                hybrid_default_title_anchor_used = bool(
                    query_rewrite_title_anchor_query or query_rewrite_legal_anchor_guess_list
                )

                rewrite_plan = {
                    "concept_tags": query_rewrite_legal_concept_tags,
                    "actor_terms": query_rewrite_actor_terms,
                    "action_terms": query_rewrite_action_terms,
                    "object_terms": query_rewrite_object_terms,
                    "doc_type_prior": query_rewrite_doc_type_prior,
                    "legal_anchor_guess": query_rewrite_legal_anchor_guess_list,
                    "topic_class": query_rewrite_topic_class,
                    "concept_seed_query": query_rewrite_concept_seed_query,
                }
                hybrid_default_doc_role_prior_applied = bool(
                    self._build_doc_role_prior(rewrite_plan)
                )
                use_family_identity = bool(
                    getattr(cfg, "HYBRID_DEFAULT_V15_FAMILY_IDENTITY_ENABLED", True)
                )
                enable_garbage_rejection = bool(
                    getattr(cfg, "HYBRID_DEFAULT_V15_GARBAGE_REJECTION_ENABLED", True)
                )

                concept_core_query = (
                    query_rewrite_concept_seed_query
                    or query_rewrite_lexical_core
                    or query_rewrite_lexical
                    or query_rewrite_clean
                    or query_for_bm25
                    or query
                )
                concept_expanded_query = (
                    query_rewrite_lexical_expanded
                    or query_rewrite_lexical
                    or query_rewrite_clean
                    or query_for_bm25
                    or query
                )
                concept_title_query = (
                    query_rewrite_title_anchor_query
                    or " ".join(
                        tokenize_for_query(
                            " ".join(
                                query_rewrite_legal_anchor_guess_list
                                + query_rewrite_legal_concept_tags
                                + query_rewrite_actor_terms[:2]
                                + query_rewrite_object_terms[:2]
                            )
                        ).split()[:12]
                    )
                )
                weak_abort_gate = bool(query_rewrite_weak_query_abort and vague_gate_enabled)
                if weak_abort_gate:
                    concept_expanded_query = ""
                    concept_title_query = ""
                    hybrid_default_no_confident_family_candidates = False
                concept_core_top_k = (
                    max(6, int(getattr(cfg, "HYBRID_DEFAULT_WEAK_ABORT_CORE_TOP_K", 12)))
                    if weak_abort_gate
                    else max(
                        _CONCEPT_BM25_CORE_TOP_K,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_CONCEPT_BM25_CORE_TOP_K",
                                _CONCEPT_BM25_CORE_TOP_K,
                            )
                        ),
                    )
                )

                bm25_core_payload = self._bm25.search(
                    concept_core_query,
                    concept_core_top_k,
                    return_debug=True,
                    mode=bm25_mode,
                    legal_refs=refs,
                    allowed_doc_ids=allowed_doc_ids or None,
                    include_text=False,
                )
                bm25_core_hits = list(bm25_core_payload.get("results", []))
                bm25_core_debug = dict(bm25_core_payload.get("debug", {}))
                bm25_core_ms = int(
                    bm25_core_debug.get(
                        "execute_search_ms",
                        bm25_core_debug.get("total_ms", 0),
                    )
                )

                bm25_expanded_hits: list[dict[str, Any]] = []
                bm25_expanded_debug: dict[str, Any] = {}
                bm25_expanded_ms = 0
                if concept_expanded_query:
                    bm25_expanded_payload = self._bm25.search(
                        concept_expanded_query,
                        max(
                            _CONCEPT_BM25_EXPANDED_TOP_K,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_CONCEPT_BM25_EXPANDED_TOP_K",
                                    _CONCEPT_BM25_EXPANDED_TOP_K,
                                )
                            ),
                        ),
                        return_debug=True,
                        mode=bm25_mode,
                        legal_refs=refs,
                        allowed_doc_ids=allowed_doc_ids or None,
                        include_text=False,
                    )
                    bm25_expanded_hits = list(bm25_expanded_payload.get("results", []))
                    bm25_expanded_debug = dict(bm25_expanded_payload.get("debug", {}))
                    bm25_expanded_ms = int(
                        bm25_expanded_debug.get(
                            "execute_search_ms",
                            bm25_expanded_debug.get("total_ms", 0),
                        )
                    )

                bm25_title_hits: list[dict[str, Any]] = []
                bm25_title_debug: dict[str, Any] = {}
                bm25_title_ms = 0
                title_search_allowed = bool(
                    concept_title_query
                    and not weak_abort_gate
                    and (query_rewrite_is_concept_query or vague_recovery_allowed)
                )
                if title_search_allowed:
                    bm25_title_payload = self._bm25.search(
                        concept_title_query,
                        max(
                            _CONCEPT_BM25_TITLE_TOP_K,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_CONCEPT_BM25_TITLE_TOP_K",
                                    _CONCEPT_BM25_TITLE_TOP_K,
                                )
                            ),
                        ),
                        return_debug=True,
                        mode=bm25_mode,
                        legal_refs=refs,
                        allowed_doc_ids=allowed_doc_ids or None,
                        include_text=False,
                    )
                    bm25_title_hits = list(bm25_title_payload.get("results", []))
                    bm25_title_debug = dict(bm25_title_payload.get("debug", {}))
                    bm25_title_ms = int(
                        bm25_title_debug.get(
                            "execute_search_ms",
                            bm25_title_debug.get("total_ms", 0),
                        )
                    )
                hybrid_default_title_anchor_hits = len(bm25_title_hits)
                hybrid_default_family_search_hits = (
                    len(bm25_core_hits) + len(bm25_expanded_hits) + len(bm25_title_hits)
                )

                bm25_results = self._merge_rows_by_chunk(
                    self._merge_rows_by_chunk(
                        bm25_core_hits,
                        bm25_expanded_hits,
                        limit=72,
                    ),
                    bm25_title_hits,
                    limit=96,
                )
                bm25_debug = {
                    "mode": "concept_multi_pass",
                    "weak_abort_gate": bool(weak_abort_gate),
                    "vague_recovery_allowed": bool(vague_recovery_allowed),
                    "core_query": concept_core_query,
                    "expanded_query": concept_expanded_query,
                    "concept_seed_query": query_rewrite_concept_seed_query,
                    "title_query": concept_title_query,
                    "core_hits": len(bm25_core_hits),
                    "expanded_hits": len(bm25_expanded_hits),
                    "title_hits": len(bm25_title_hits),
                    "core_debug": bm25_core_debug,
                    "expanded_debug": bm25_expanded_debug,
                    "title_debug": bm25_title_debug,
                }
                latencies_ms["bm25"] = int(bm25_core_ms + bm25_expanded_ms + bm25_title_ms)

                concept_vector_hits: list[dict[str, Any]] = []
                candidate_scores = self._aggregate_concept_doc_candidates(
                    rewrite=rewrite_plan,
                    bm25_core_hits=bm25_core_hits,
                    bm25_expanded_hits=bm25_expanded_hits,
                    bm25_title_hits=bm25_title_hits,
                    vector_hits=[],
                    use_family_identity=use_family_identity,
                )
                resolve_payload = self._resolve_family_candidates(
                    candidates=candidate_scores,
                    max_families=max(
                        1,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_FAMILIES",
                                _CONCEPT_MAX_CANDIDATE_FAMILIES,
                            )
                        ),
                    ),
                    max_docs=max(
                        1,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_DOCS",
                                _CONCEPT_MAX_CANDIDATE_DOCS,
                            )
                        ),
                    ),
                    enable_garbage_rejection=enable_garbage_rejection,
                )
                selected_doc_ids = list(resolve_payload.get("selected_doc_ids", []))
                selected_family_keys = list(resolve_payload.get("selected_family_keys", []))
                hybrid_default_family_candidate_count_pre_filter = int(
                    len(resolve_payload.get("ranked_families", []))
                )
                hybrid_default_family_score_top1 = float(resolve_payload.get("family_score_top1", 0.0))
                hybrid_default_family_score_top2 = float(resolve_payload.get("family_score_top2", 0.0))
                hybrid_default_family_score_margin = float(resolve_payload.get("family_score_margin", 0.0))
                hybrid_default_family_identity_score_top1 = float(
                    resolve_payload.get("family_identity_score_top1", 0.0)
                )
                hybrid_default_family_identity_score_top2 = float(
                    resolve_payload.get("family_identity_score_top2", 0.0)
                )
                hybrid_default_family_identity_margin = float(
                    resolve_payload.get("family_identity_margin", 0.0)
                )
                hybrid_default_garbage_family_rejected = bool(
                    resolve_payload.get("garbage_family_rejected", False)
                )
                hybrid_default_garbage_family_reject_reason = str(
                    resolve_payload.get("garbage_family_reject_reason", "")
                )
                hybrid_default_trusted_family = bool(resolve_payload.get("trusted_family", False))
                hybrid_default_shortlist_after_reject_count = len(selected_family_keys)
                hybrid_default_shortlist_after_reject_empty = len(selected_family_keys) == 0

                need_family_recovery = (
                    len(selected_family_keys) == 0
                    or hybrid_default_family_score_margin < float(
                        getattr(cfg, "HYBRID_DEFAULT_FAMILY_MARGIN_RECOVERY_THRESHOLD", 0.12)
                    )
                    or hybrid_default_title_anchor_hits == 0
                )
                if need_family_recovery and not weak_abort_gate and vague_recovery_allowed:
                    hybrid_default_mode = "vague_concept_recovery"
                    hybrid_default_family_recovery_used = True
                    recovery_query = (
                        query_rewrite_concept_seed_query
                        or query_rewrite_title_anchor_query
                        or query_rewrite_clean
                        or concept_title_query
                        or query
                    )
                    recovery_payload = self._bm25.search(
                        recovery_query,
                        max(
                            _CONCEPT_BM25_TITLE_TOP_K,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_FAMILY_RECOVERY_TITLE_TOP_K",
                                    16,
                                )
                            ),
                        ),
                        return_debug=True,
                        mode=bm25_mode,
                        legal_refs=refs,
                        allowed_doc_ids=allowed_doc_ids or None,
                        include_text=False,
                    )
                    recovery_hits = list(recovery_payload.get("results", []))
                    bm25_title_hits = self._merge_rows_by_chunk(
                        bm25_title_hits,
                        recovery_hits,
                        limit=max(_CONCEPT_BM25_TITLE_TOP_K, 20),
                    )
                    bm25_results = self._merge_rows_by_chunk(
                        bm25_results,
                        recovery_hits,
                        limit=128,
                    )
                    hybrid_default_family_recovery_promoted = len(recovery_hits)
                    hybrid_default_title_anchor_hits = len(bm25_title_hits)

                    candidate_scores = self._aggregate_concept_doc_candidates(
                        rewrite=rewrite_plan,
                        bm25_core_hits=bm25_core_hits,
                        bm25_expanded_hits=bm25_expanded_hits,
                        bm25_title_hits=bm25_title_hits,
                        vector_hits=[],
                        use_family_identity=use_family_identity,
                    )
                    resolve_payload = self._resolve_family_candidates(
                        candidates=candidate_scores,
                        max_families=max(
                            1,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_FAMILIES",
                                    _CONCEPT_MAX_CANDIDATE_FAMILIES,
                                )
                            ),
                        ),
                        max_docs=max(
                            1,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_DOCS",
                                    _CONCEPT_MAX_CANDIDATE_DOCS,
                                )
                            ),
                        ),
                        enable_garbage_rejection=enable_garbage_rejection,
                    )
                    selected_doc_ids = list(resolve_payload.get("selected_doc_ids", []))
                    selected_family_keys = list(resolve_payload.get("selected_family_keys", []))
                    hybrid_default_family_score_top1 = float(resolve_payload.get("family_score_top1", 0.0))
                    hybrid_default_family_score_top2 = float(resolve_payload.get("family_score_top2", 0.0))
                    hybrid_default_family_score_margin = float(resolve_payload.get("family_score_margin", 0.0))
                    hybrid_default_family_identity_score_top1 = float(
                        resolve_payload.get("family_identity_score_top1", 0.0)
                    )
                    hybrid_default_family_identity_score_top2 = float(
                        resolve_payload.get("family_identity_score_top2", 0.0)
                    )
                    hybrid_default_family_identity_margin = float(
                        resolve_payload.get("family_identity_margin", 0.0)
                    )
                    hybrid_default_garbage_family_rejected = bool(
                        resolve_payload.get("garbage_family_rejected", False)
                    )
                    hybrid_default_garbage_family_reject_reason = str(
                        resolve_payload.get("garbage_family_reject_reason", "")
                    )
                    hybrid_default_trusted_family = bool(resolve_payload.get("trusted_family", False))
                    hybrid_default_shortlist_after_reject_count = len(selected_family_keys)
                    hybrid_default_shortlist_after_reject_empty = len(selected_family_keys) == 0

                concept_vector_enabled = (
                    vector_available
                    and not vector_skipped_by_route
                    and query_rewrite_vagueness_level != "hard"
                    and self._should_use_vector_for_concept_doc_recall(
                        concept_confidence=query_rewrite_concept_confidence,
                        lexical_quality_flags=query_rewrite_lexical_quality_flags,
                        topic_class=query_rewrite_topic_class,
                        bm25_title_hits=bm25_title_hits,
                        bm25_core_hits=bm25_core_hits,
                        bm25_expanded_hits=bm25_expanded_hits,
                        candidate_family_count=len(selected_family_keys),
                        family_score_margin=float(hybrid_default_family_score_margin),
                        weak_query_abort=bool(weak_abort_gate),
                    )
                    and (
                        len(selected_family_keys) == 0
                        or hybrid_default_family_score_margin < 0.12
                        or hybrid_default_title_anchor_hits == 0
                        or not hybrid_default_trusted_family
                    )
                )
                if concept_vector_enabled:
                    hybrid_default_vector_used_for_doc_recall = True
                    vector_query = (
                        query_rewrite_semantic
                        or query_for_vector
                        or concept_expanded_query
                    )
                    t_vector_doc = time.perf_counter()
                    try:
                        concept_vector_hits = self._vector.search(
                            vector_query,
                            min(
                                vector_top_k_effective,
                                int(
                                    getattr(
                                        cfg,
                                        "HYBRID_DEFAULT_CONCEPT_VECTOR_TOP_K",
                                        _CONCEPT_VECTOR_TOP_K,
                                    )
                                ),
                            ),
                            allowed_doc_ids=allowed_doc_ids or None,
                        )
                    except Exception as exc:
                        logger.warning(f"Concept-aware vector recall failed: {exc}")
                        concept_vector_hits = []
                    vector_wall_ms = round((time.perf_counter() - t_vector_doc) * 1000)
                    latencies_ms["vector"] = int(vector_wall_ms)
                else:
                    latencies_ms["vector"] = 0
                    vector_results = []

                candidate_scores = self._aggregate_concept_doc_candidates(
                    rewrite=rewrite_plan,
                    bm25_core_hits=bm25_core_hits,
                    bm25_expanded_hits=bm25_expanded_hits,
                    bm25_title_hits=bm25_title_hits,
                    vector_hits=concept_vector_hits,
                    use_family_identity=use_family_identity,
                )
                resolve_payload = self._resolve_family_candidates(
                    candidates=candidate_scores,
                    max_families=max(
                        1,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_FAMILIES",
                                _CONCEPT_MAX_CANDIDATE_FAMILIES,
                            )
                        ),
                    ),
                    max_docs=max(
                        1,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_DOCS",
                                _CONCEPT_MAX_CANDIDATE_DOCS,
                            )
                        ),
                    ),
                    enable_garbage_rejection=enable_garbage_rejection,
                )
                selected_doc_ids = list(resolve_payload.get("selected_doc_ids", []))
                selected_family_keys = list(resolve_payload.get("selected_family_keys", []))
                hybrid_default_family_candidate_count_post_filter = int(
                    len(resolve_payload.get("ranked_families", []))
                )
                hybrid_default_family_score_top1 = float(resolve_payload.get("family_score_top1", 0.0))
                hybrid_default_family_score_top2 = float(resolve_payload.get("family_score_top2", 0.0))
                hybrid_default_family_score_margin = float(resolve_payload.get("family_score_margin", 0.0))
                hybrid_default_family_identity_score_top1 = float(
                    resolve_payload.get("family_identity_score_top1", 0.0)
                )
                hybrid_default_family_identity_score_top2 = float(
                    resolve_payload.get("family_identity_score_top2", 0.0)
                )
                hybrid_default_family_identity_margin = float(
                    resolve_payload.get("family_identity_margin", 0.0)
                )
                hybrid_default_garbage_family_rejected = bool(
                    resolve_payload.get("garbage_family_rejected", False)
                )
                hybrid_default_garbage_family_reject_reason = str(
                    resolve_payload.get("garbage_family_reject_reason", "")
                )
                hybrid_default_trusted_family = bool(resolve_payload.get("trusted_family", False))
                hybrid_default_shortlist_after_reject_count = len(selected_family_keys)
                hybrid_default_shortlist_after_reject_empty = len(selected_family_keys) == 0
                hybrid_default_candidate_doc_count = len(selected_doc_ids)
                hybrid_default_candidate_family_count = len(selected_family_keys)
                hybrid_default_selected_doc_ids = list(selected_doc_ids)
                hybrid_default_selected_family_keys = list(selected_family_keys)
                if not selected_family_keys:
                    hybrid_default_no_confident_family_candidates = True

                if candidate_scores:
                    hybrid_default_doc_score_top1 = float(candidate_scores[0].final_doc_score)
                    hybrid_default_doc_top1_family = str(candidate_scores[0].family_key)
                    hybrid_default_doc_role_prior_top1 = str(candidate_scores[0].doc_role)
                    hybrid_default_family_top1_doc_role = str(candidate_scores[0].doc_role)
                    hybrid_default_family_top1_is_implementation = bool(
                        candidate_scores[0].doc_role in _IMPLEMENTATION_ROLES
                    )
                    hybrid_default_genericity_penalty_top1 = float(candidate_scores[0].genericity_penalty)
                    if len(candidate_scores) >= 2:
                        hybrid_default_doc_score_top2 = float(candidate_scores[1].final_doc_score)
                        hybrid_default_doc_top2_family = str(candidate_scores[1].family_key)
                hybrid_default_doc_score_margin = max(
                    0.0,
                    float(hybrid_default_doc_score_top1 - hybrid_default_doc_score_top2),
                )
                hybrid_default_genericity_penalty_applied_count = int(
                    sum(1 for score in candidate_scores if score.genericity_penalty > 0.0)
                )
                hybrid_default_wrong_doc_role_penalty_applied = bool(
                    any(score.wrong_doc_role_penalty > 0.0 for score in candidate_scores)
                )

                if selected_doc_ids and selected_family_keys:
                    selected_doc_ids, bridge_hits = self._implementation_bridge_expand(
                        selected_doc_ids=selected_doc_ids,
                        selected_family_keys=selected_family_keys,
                        candidates=candidate_scores,
                        trusted_family=bool(hybrid_default_trusted_family),
                        family_score_margin=float(hybrid_default_family_score_margin),
                        max_docs=max(
                            _CONCEPT_MAX_CANDIDATE_DOCS,
                            int(
                                getattr(
                                    cfg,
                                    "HYBRID_DEFAULT_CONCEPT_MAX_CANDIDATE_DOCS",
                                    _CONCEPT_MAX_CANDIDATE_DOCS,
                                )
                            ),
                        ) + 2,
                    )
                    if bridge_hits > 0:
                        hybrid_default_implementation_bridge_used = True
                        hybrid_default_implementation_bridge_hits = int(bridge_hits)
                        hybrid_default_selected_doc_ids = list(selected_doc_ids)

                if selected_doc_ids:
                    selected_doc_id_set = {
                        str(doc_id).strip() for doc_id in selected_doc_ids if str(doc_id).strip()
                    }
                    if allowed_doc_ids:
                        selected_doc_id_set &= {
                            str(doc_id).strip() for doc_id in allowed_doc_ids if str(doc_id).strip()
                        }
                    if selected_doc_id_set:
                        allowed_doc_ids = selected_doc_id_set
                    bm25_results, shortlist_chunk_count = self._retrieve_chunks_within_shortlist(
                        rows=bm25_results,
                        selected_doc_ids=selected_doc_ids,
                        selected_family_keys=selected_family_keys,
                        limit=max(_CONCEPT_MAX_FOCUS_CHUNKS, final_top_k * 2),
                    )
                    hybrid_default_focus_rerank_candidate_chunks = int(shortlist_chunk_count)
                    concept_vector_doc_ids = {
                        str(row.get("doc_id", "")).strip()
                        for row in concept_vector_hits
                        if str(row.get("doc_id", "")).strip()
                    }
                    if hybrid_default_doc_top1_family:
                        vector_family_hits = [
                            row
                            for row in concept_vector_hits
                            if self._canonical_law_family_key(str(row.get("title", ""))) == hybrid_default_doc_top1_family
                        ]
                        hybrid_default_vector_support_same_family = bool(vector_family_hits)
                    else:
                        hybrid_default_vector_support_same_family = bool(
                            concept_vector_doc_ids & set(selected_doc_ids)
                        )
                else:
                    hybrid_default_no_confident_family_candidates = True
                vector_results = []
                vector_result_used = False

            vector_enabled = (
                vector_available
                and not vector_skipped_by_route
                and not vector_skipped_by_quality_gate
            )
            if concept_hybrid_default_enabled:
                pass
            elif vector_enabled and effective_route == "doc_scoped_hybrid_exact_doc":
                bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
                min_hits_to_skip_vector = max(
                    1,
                    int(getattr(cfg, "DOC_SCOPED_EXACT_VECTOR_SKIP_MIN_BM25_HITS", 6)),
                )
                if len(bm25_results) >= min_hits_to_skip_vector:
                    vector_skipped_by_route = True
                    vector_results = []
                    latencies_ms["vector"] = 0
                else:
                    vector_future = self._executor.submit(_run_vector)
                    try:
                        t_wait = time.perf_counter()
                        vector_payload = vector_future.result(
                            timeout=vector_timeout_ms / 1000.0,
                        )
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_results = list(vector_payload.get("rows", []))
                        vector_wall_ms = int(vector_payload.get("wall_ms", 0))
                        latencies_ms["vector"] = int(vector_wall_ms)
                    except TimeoutError:
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_timed_out = True
                        vector_results = []
                        latencies_ms["vector"] = vector_timeout_ms
                        vector_cancelled = bool(vector_future.cancel())
                        vector_wall_ms = max(vector_wall_ms, vector_timeout_ms)
            elif vector_enabled and effective_route == "doc_scoped_hybrid_candidate_docs":
                bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
                should_early_exit, early_exit_debug = self._should_early_exit_scoped_candidate(
                    bm25_results=bm25_results,
                    allowed_doc_ids=allowed_doc_ids,
                    lookup_confidence=document_lookup_top1_confidence or document_lookup_confidence,
                    lookup_margin=document_lookup_margin,
                )
                candidate_route_early_exit_reason = str(early_exit_debug.get("reason", ""))
                candidate_route_early_exit_score_gap = float(early_exit_debug.get("score_gap", 0.0))
                if should_early_exit:
                    candidate_route_early_exit_used = True
                    vector_skipped_by_route = True
                    latencies_ms["vector"] = 0
                    vector_results = []
                else:
                    vector_future = self._executor.submit(_run_vector)
                    try:
                        t_wait = time.perf_counter()
                        vector_payload = vector_future.result(
                            timeout=vector_timeout_ms / 1000.0,
                        )
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_results = list(vector_payload.get("rows", []))
                        vector_wall_ms = int(vector_payload.get("wall_ms", 0))
                        latencies_ms["vector"] = int(vector_wall_ms)
                    except TimeoutError:
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_timed_out = True
                        vector_results = []
                        latencies_ms["vector"] = vector_timeout_ms
                        vector_cancelled = bool(vector_future.cancel())
                        vector_wall_ms = max(vector_wall_ms, vector_timeout_ms)
            elif vector_enabled and effective_route == "narrow_bm25":
                bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
                if bool(getattr(cfg, "VECTOR_SKIP_BY_QUALITY_GATE", True)):
                    vector_skipped_by_quality_gate = self._should_skip_vector_for_narrow(refs, bm25_results)

                if vector_skipped_by_quality_gate:
                    latencies_ms["vector"] = 0
                    vector_results = []
                else:
                    vector_future = self._executor.submit(_run_vector)
                    try:
                        t_wait = time.perf_counter()
                        vector_payload = vector_future.result(
                            timeout=vector_timeout_ms / 1000.0,
                        )
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_results = list(vector_payload.get("rows", []))
                        vector_wall_ms = int(vector_payload.get("wall_ms", 0))
                        latencies_ms["vector"] = int(vector_wall_ms)
                    except TimeoutError:
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_timed_out = True
                        vector_results = []
                        latencies_ms["vector"] = vector_timeout_ms
                        vector_cancelled = bool(vector_future.cancel())
                        vector_wall_ms = max(vector_wall_ms, vector_timeout_ms)
            elif vector_enabled:
                executed_parallel = True
                bm25_future = self._executor.submit(_run_bm25)
                vector_future = self._executor.submit(_run_vector)
                bm25_results, bm25_debug, latencies_ms["bm25"] = bm25_future.result()
                try:
                    t_wait = time.perf_counter()
                    vector_payload = vector_future.result(
                        timeout=vector_timeout_ms / 1000.0,
                    )
                    vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                    vector_results = list(vector_payload.get("rows", []))
                    vector_wall_ms = int(vector_payload.get("wall_ms", 0))
                    latencies_ms["vector"] = int(vector_wall_ms)
                except TimeoutError:
                    vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                    vector_timed_out = True
                    vector_results = []
                    latencies_ms["vector"] = vector_timeout_ms
                    vector_cancelled = bool(vector_future.cancel())
                    vector_wall_ms = max(vector_wall_ms, vector_timeout_ms)
            else:
                bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
                latencies_ms["vector"] = 0

        if effective_route == "hybrid_default":
            bm25_results, bm25_gate_debug = self._apply_hybrid_default_doc_gate(
                bm25_results,
                doc_type_prior=query_rewrite_doc_type_prior,
                exclude_doc_type_hint=query_rewrite_exclude_doc_type_hint,
                hard_filter=hybrid_default_hard_doc_type_filter,
                hard_allow_buckets=set(hybrid_default_hard_doc_type_allowlist),
            )
            vector_results, vector_gate_debug = self._apply_hybrid_default_doc_gate(
                vector_results,
                doc_type_prior=query_rewrite_doc_type_prior,
                exclude_doc_type_hint=query_rewrite_exclude_doc_type_hint,
                hard_filter=hybrid_default_hard_doc_type_filter,
                hard_allow_buckets=set(hybrid_default_hard_doc_type_allowlist),
            )
            hybrid_doc_gate_applied = bool(
                bm25_gate_debug.get("applied", False) or vector_gate_debug.get("applied", False)
            )
            hybrid_doc_gate_bm25_before = int(bm25_gate_debug.get("before", 0))
            hybrid_doc_gate_bm25_after = int(bm25_gate_debug.get("after", 0))
            hybrid_doc_gate_vector_before = int(vector_gate_debug.get("before", 0))
            hybrid_doc_gate_vector_after = int(vector_gate_debug.get("after", 0))
            hybrid_doc_gate_removed_bm25 = int(bm25_gate_debug.get("removed", 0))
            hybrid_doc_gate_removed_vector = int(vector_gate_debug.get("removed", 0))
            hybrid_doc_gate_override_kept_original = bool(
                bm25_gate_debug.get("override_kept_original", False)
                or vector_gate_debug.get("override_kept_original", False)
            )
            hybrid_doc_gate_bm25_filtered_before_override = int(
                bm25_gate_debug.get("filtered_before_override", hybrid_doc_gate_bm25_after)
            )
            hybrid_doc_gate_vector_filtered_before_override = int(
                vector_gate_debug.get("filtered_before_override", hybrid_doc_gate_vector_after)
            )
            has_explicit_doc_anchor = bool(
                refs.get("document_number")
                or refs.get("document_short")
                or refs.get("document_loose")
            )
            scoped_zero_hits = (len(bm25_results) + len(vector_results)) == 0
            bm25_gated_zero = hybrid_doc_gate_bm25_after == 0
            if (
                bool(getattr(cfg, "HYBRID_DEFAULT_SCOPE_FALLBACK_ENABLED", True))
                and allowed_doc_ids
                and not has_explicit_doc_anchor
                and (scoped_zero_hits or bm25_gated_zero)
            ):
                hybrid_default_scope_fallback_used = True
                hybrid_default_scope_fallback_reason = (
                    "scoped_zero_hits"
                    if scoped_zero_hits
                    else "bm25_gated_zero"
                )
                fallback_retry_used = True
                fallback_from_route = "hybrid_default_scoped"
                fallback_to_route = "hybrid_default_unscoped"
                fallback_retry_reason = hybrid_default_scope_fallback_reason

                first_pass_bm25_results = list(bm25_results)
                first_pass_vector_results = list(vector_results)
                hybrid_default_scope_fallback_first_pass_hits = (
                    len(first_pass_bm25_results) + len(first_pass_vector_results)
                )
                if hybrid_default_scope_fallback_first_pass_hits > 0:
                    if first_pass_vector_results:
                        first_pass_fused = self._rrf_fusion(
                            first_pass_bm25_results,
                            first_pass_vector_results,
                        )
                    else:
                        first_pass_fused = list(first_pass_bm25_results)
                    first_pass_deduped, _ = self._dedup_by_legal_unit(first_pass_fused)
                    first_pass_reranked = self._heuristic_rerank(
                        query_for_bm25,
                        first_pass_deduped,
                        legal_refs=refs,
                        route=effective_route,
                        allowed_doc_ids=allowed_doc_ids,
                        document_lookup_confidence=document_lookup_confidence,
                        query_rewrite_doc_type_prior=query_rewrite_doc_type_prior,
                        query_rewrite_exclude_doc_type_hint=query_rewrite_exclude_doc_type_hint,
                        query_rewrite_subclass=query_rewrite_subclass,
                        query_rewrite_legal_concept_tags=query_rewrite_legal_concept_tags,
                        query_rewrite_actor_terms=query_rewrite_actor_terms,
                        query_rewrite_action_terms=query_rewrite_action_terms,
                    )
                    first_pass_hydrated = self._bm25.hydrate_rows(
                        first_pass_reranked[: max(final_top_k, 10)],
                        include_text=True,
                    )
                    hybrid_default_scope_fallback_first_pass_top10 = self._compact_hits_for_eval(
                        first_pass_hydrated,
                        top_k=10,
                    )

                allowed_doc_ids = set()
                if hybrid_anchor_scope_reason:
                    hybrid_anchor_scope_reason = (
                        f"{hybrid_anchor_scope_reason}|fallback_unscoped_zero_hits"
                    )
                else:
                    hybrid_anchor_scope_reason = "fallback_unscoped_zero_hits"

                t_retry = time.perf_counter()
                fallback_query, fallback_query_reason = (
                    self._choose_hybrid_default_fallback_query(
                        raw_query=query,
                        clean_query=query_rewrite_clean,
                        lexical_query=query_rewrite_lexical,
                        intent_templates=query_rewrite_intent_template_hits,
                        focus_terms=query_rewrite_focus_terms,
                    )
                )
                fallback_top_k = max(
                    8,
                    min(
                        12,
                        int(
                            getattr(
                                cfg,
                                "HYBRID_DEFAULT_SCOPE_FALLBACK_BM25_TOP_K",
                                10,
                            )
                        ),
                    ),
                )
                hybrid_default_scope_fallback_bm25_only_query = fallback_query
                hybrid_default_scope_fallback_bm25_only_query_reason = fallback_query_reason
                fallback_payload = self._bm25.search(
                    fallback_query,
                    fallback_top_k,
                    return_debug=True,
                    mode=bm25_mode,
                    legal_refs=refs,
                    allowed_doc_ids=None,
                    include_text=False,
                )
                fallback_elapsed_ms = round((time.perf_counter() - t_retry) * 1000)
                fallback_bm25_results = list(fallback_payload.get("results", []))
                fallback_bm25_debug = dict(fallback_payload.get("debug", {}))
                fallback_bm25_debug["hybrid_default_scope_fallback_bm25_only"] = True
                fallback_bm25_debug["fallback_query"] = fallback_query
                fallback_bm25_debug["fallback_query_reason"] = fallback_query_reason

                fallback_bm25_results, fallback_bm25_gate_debug = self._apply_hybrid_default_doc_gate(
                    fallback_bm25_results,
                    doc_type_prior=query_rewrite_doc_type_prior,
                    exclude_doc_type_hint=query_rewrite_exclude_doc_type_hint,
                    hard_filter=hybrid_default_hard_doc_type_filter,
                    hard_allow_buckets=set(hybrid_default_hard_doc_type_allowlist),
                )
                hybrid_default_scope_fallback_bm25_only_used = True
                hybrid_default_scope_fallback_bm25_only_hits = len(fallback_bm25_results)
                hybrid_default_scope_fallback_bm25_only_latency_ms = int(fallback_elapsed_ms)
                hybrid_default_scope_fallback_second_vector_called = False

                if fallback_bm25_results and hybrid_default_scope_fallback_first_pass_hits == 0:
                    bm25_results = list(fallback_bm25_results)
                    vector_results = []
                elif fallback_bm25_results:
                    bm25_results = self._merge_rows_by_chunk(
                        first_pass_bm25_results,
                        fallback_bm25_results,
                        limit=max(16, fallback_top_k * 2),
                    )
                    vector_results = list(first_pass_vector_results)
                    hybrid_default_scope_fallback_vector_reused = bool(vector_results)
                else:
                    bm25_results = list(first_pass_bm25_results)
                    vector_results = list(first_pass_vector_results)
                    hybrid_default_scope_fallback_vector_reused = bool(vector_results)

                bm25_debug = dict(fallback_bm25_debug)
                latencies_ms["hybrid_default_scope_fallback_bm25_only"] = int(
                    hybrid_default_scope_fallback_bm25_only_latency_ms
                )
                latencies_ms["fallback_retry"] = int(
                    latencies_ms.get("fallback_retry", 0)
                ) + int(hybrid_default_scope_fallback_bm25_only_latency_ms)

                hybrid_doc_gate_applied = bool(
                    bm25_gate_debug.get("applied", False)
                    or vector_gate_debug.get("applied", False)
                    or fallback_bm25_gate_debug.get("applied", False)
                )
                hybrid_doc_gate_bm25_before = int(
                    fallback_bm25_gate_debug.get("before", hybrid_doc_gate_bm25_before)
                )
                hybrid_doc_gate_bm25_after = int(len(bm25_results))
                hybrid_doc_gate_vector_before = int(vector_gate_debug.get("before", hybrid_doc_gate_vector_before))
                hybrid_doc_gate_vector_after = int(len(vector_results))
                hybrid_doc_gate_removed_bm25 = int(
                    max(0, hybrid_doc_gate_bm25_before - hybrid_doc_gate_bm25_after)
                )
                hybrid_doc_gate_removed_vector = int(
                    max(0, hybrid_doc_gate_vector_before - hybrid_doc_gate_vector_after)
                )
                hybrid_doc_gate_override_kept_original = bool(
                    bm25_gate_debug.get("override_kept_original", False)
                    or vector_gate_debug.get("override_kept_original", False)
                    or fallback_bm25_gate_debug.get("override_kept_original", False)
                )
                hybrid_doc_gate_bm25_filtered_before_override = int(
                    fallback_bm25_gate_debug.get(
                        "filtered_before_override",
                        hybrid_doc_gate_bm25_after,
                    )
                )
                hybrid_doc_gate_vector_filtered_before_override = int(
                    vector_gate_debug.get(
                        "filtered_before_override",
                        hybrid_doc_gate_vector_after,
                    )
                )
                hybrid_default_scope_fallback_hits = len(bm25_results) + len(vector_results)

        if (
            bool(getattr(cfg, "DOC_SCOPED_EXACT_FALLBACK_ENABLED", True))
            and effective_route == "doc_scoped_hybrid_exact_doc"
        ):
            total_hits_before_fallback = len(bm25_results) + len(vector_results)
            need_fallback_retry = (
                total_hits_before_fallback == 0
                or vector_timed_out
            )
            if need_fallback_retry:
                fallback_retry_used = True
                fallback_from_route = "doc_scoped_hybrid_exact_doc"
                fallback_retry_reason = (
                    "vector_timeout"
                    if vector_timed_out
                    else "scoped_zero_hits"
                )

                candidate_max_docs = max(1, int(getattr(cfg, "DOCUMENT_LOOKUP_MAX_CANDIDATE_DOCS", 3)))
                fallback_candidate_doc_ids = [
                    doc_id
                    for doc_id in document_lookup_doc_ids
                    if doc_id and doc_id not in allowed_doc_ids
                ][:candidate_max_docs]
                if not fallback_candidate_doc_ids and document_lookup_doc_ids:
                    fallback_candidate_doc_ids = document_lookup_doc_ids[:candidate_max_docs]

                if fallback_candidate_doc_ids:
                    effective_route = "doc_scoped_hybrid_candidate_docs"
                    allowed_doc_ids = set(fallback_candidate_doc_ids)
                    document_lookup_selected_reason = "fallback_from_exact_zero_hits"
                    document_lookup_scope_reason_detail = (
                        f"fallback_retry={fallback_retry_reason},"
                        f"candidate_docs={len(fallback_candidate_doc_ids)}"
                    )
                else:
                    effective_route = "law_anchored_hybrid_loose"
                    soft_top_k = max(1, int(getattr(cfg, "LAW_ANCHORED_LOOSE_SOFT_SCOPE_TOP_K", 5)))
                    if document_lookup_doc_ids:
                        allowed_doc_ids = set(document_lookup_doc_ids[:soft_top_k])
                        document_lookup_soft_scope_applied = True
                    document_lookup_selected_reason = "fallback_to_loose_no_candidate_docs"
                    document_lookup_scope_reason_detail = (
                        f"fallback_retry={fallback_retry_reason},fallback_route=loose"
                    )

                fallback_to_route = effective_route
                bm25_mode = self._bm25_mode_for_route(effective_route)
                if effective_route == "doc_scoped_hybrid_candidate_docs":
                    bm25_exec_top_k = max(
                        6,
                        min(
                            int(bm25_exec_top_k),
                            int(getattr(cfg, "DOC_SCOPED_CANDIDATE_BM25_TOP_K", 10)),
                        ),
                    )
                    vector_top_k_effective = max(
                        6,
                        min(
                            int(vector_top_k_effective),
                            int(getattr(cfg, "DOC_SCOPED_CANDIDATE_VECTOR_TOP_K", 8)),
                        ),
                    )
                    vector_timeout_ms = min(
                        vector_timeout_ms,
                        max(500, int(getattr(cfg, "DOC_SCOPED_CANDIDATE_VECTOR_TIMEOUT_MS", 1100))),
                    )
                    vector_budget_ms = int(vector_timeout_ms)
                document_lookup_selected_route = effective_route
                document_lookup_selected_doc_ids = sorted(allowed_doc_ids)
                lookup_doc_num_map = {
                    str(row.get("doc_id", "")).strip(): str(row.get("document_number", "")).strip()
                    for row in document_lookup_rows
                    if str(row.get("doc_id", "")).strip()
                }
                document_lookup_selected_doc_numbers = [
                    lookup_doc_num_map.get(doc_id, "")
                    for doc_id in document_lookup_selected_doc_ids
                ]

                vector_skipped_by_route = False
                vector_skipped_by_quality_gate = False
                vector_timed_out = False
                vector_cancelled = False
                vector_wall_ms = 0
                t_retry = time.perf_counter()
                if vector_available:
                    executed_parallel = True
                    bm25_future = self._executor.submit(_run_bm25)
                    vector_future = self._executor.submit(_run_vector)
                    bm25_results, bm25_debug, latencies_ms["bm25"] = bm25_future.result()
                    try:
                        t_wait = time.perf_counter()
                        vector_payload = vector_future.result(
                            timeout=vector_timeout_ms / 1000.0,
                        )
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_results = list(vector_payload.get("rows", []))
                        vector_wall_ms = int(vector_payload.get("wall_ms", 0))
                        latencies_ms["vector"] = int(vector_wall_ms)
                    except TimeoutError:
                        vector_join_wait_ms += round((time.perf_counter() - t_wait) * 1000)
                        vector_timed_out = True
                        vector_results = []
                        latencies_ms["vector"] = vector_timeout_ms
                        vector_cancelled = bool(vector_future.cancel())
                        vector_wall_ms = max(vector_wall_ms, vector_timeout_ms)
                else:
                    bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
                    vector_results = []
                    latencies_ms["vector"] = 0
                latencies_ms["fallback_retry"] = round((time.perf_counter() - t_retry) * 1000)

        vector_wall_ms = max(vector_wall_ms, int(latencies_ms.get("vector", 0)))
        vector_deadline_hit = bool(
            vector_available
            and not vector_skipped_by_route
            and not vector_skipped_by_quality_gate
            and vector_wall_ms > vector_budget_ms
        )
        vector_result_used = bool(vector_results)

        scoped_filter_broken = False
        scoped_filter_broken_doc_ids: list[str] = []
        if allowed_doc_ids:
            allowed_doc_ids_norm = {
                str(doc_id).strip()
                for doc_id in allowed_doc_ids
                if str(doc_id).strip()
            }
            for row in bm25_results:
                item_doc_id = str(row.get("doc_id", "")).strip()
                if item_doc_id and item_doc_id not in allowed_doc_ids_norm:
                    scoped_filter_broken = True
                    if item_doc_id not in scoped_filter_broken_doc_ids:
                        scoped_filter_broken_doc_ids.append(item_doc_id)
            if scoped_filter_broken:
                logger.error(
                    "SCOPED_FILTER_BROKEN: route={} allowed={} offending={}",
                    effective_route,
                    sorted(allowed_doc_ids_norm),
                    scoped_filter_broken_doc_ids,
                )

        if vector_timed_out:
            logger.warning(
                "Vector search timed out: timeout={}ms route={} query='{}'",
                vector_timeout_ms,
                effective_route,
                query[:160],
            )
        elif vector_deadline_hit:
            logger.warning(
                "Vector deadline hit without timeout exception: budget={}ms wall={}ms route={} query='{}'",
                vector_budget_ms,
                vector_wall_ms,
                effective_route,
                query[:160],
            )

        latencies_ms["vector_wall"] = int(vector_wall_ms)
        latencies_ms["vector_join_wait"] = int(vector_join_wait_ms)

        latencies_ms["bm25_open_index"] = int(bm25_debug.get("open_index_ms", 0))
        latencies_ms["bm25_open_searcher"] = int(bm25_debug.get("open_searcher_ms", 0))
        latencies_ms["bm25_build_query"] = int(bm25_debug.get("build_query_ms", 0))
        latencies_ms["bm25_execute_search"] = int(bm25_debug.get("execute_search_ms", 0))
        latencies_ms["bm25_hydrate_hits"] = int(bm25_debug.get("hydrate_hits_ms", 0))

        t_rrf = time.perf_counter()
        if vector_results:
            fused = self._rrf_fusion(bm25_results, vector_results)
        else:
            fused = list(bm25_results)
        latencies_ms["rrf"] = round((time.perf_counter() - t_rrf) * 1000)

        t_dedup = time.perf_counter()
        deduped, dedup_filtered = self._dedup_by_legal_unit(fused)
        filtered_out = list(dedup_filtered)
        latencies_ms["dedup"] = round((time.perf_counter() - t_dedup) * 1000)

        t_rerank = time.perf_counter()
        reranked = self._heuristic_rerank(
            query_for_bm25,
            deduped,
            legal_refs=refs,
            route=effective_route,
            allowed_doc_ids=allowed_doc_ids,
            document_lookup_confidence=document_lookup_confidence,
            query_rewrite_doc_type_prior=query_rewrite_doc_type_prior,
            query_rewrite_exclude_doc_type_hint=query_rewrite_exclude_doc_type_hint,
            query_rewrite_subclass=query_rewrite_subclass,
            query_rewrite_legal_concept_tags=query_rewrite_legal_concept_tags,
            query_rewrite_actor_terms=query_rewrite_actor_terms,
            query_rewrite_action_terms=query_rewrite_action_terms,
        )
        if effective_route in {"law_anchored_hybrid", "law_anchored_hybrid_loose"}:
            reranked = self._prioritize_law_anchored_hits(
                reranked,
                law_name=str(refs.get("law_name", "")),
            )
        elif effective_route == "hybrid_default":
            for idx, row in enumerate(reranked, start=1):
                fam = str(row.get("law_family_key", "")).strip() or self._canonical_law_family_key(
                    str(row.get("title", ""))
                )
                if fam and fam in set(hybrid_default_selected_family_keys):
                    rank_first_family_pre_focus = idx
                    break
            if reranked:
                rank_first_doc_role_pre_focus = self._doc_role_of_item(reranked[0])

            can_focus, family_conf = self._should_run_focus_rerank(
                family_score_top1=float(hybrid_default_family_score_top1),
                family_score_margin=float(hybrid_default_family_score_margin),
                candidate_family_count=int(hybrid_default_candidate_family_count),
                candidate_chunks=int(len(reranked)),
                vagueness_level=query_rewrite_vagueness_level,
                trusted_family=bool(hybrid_default_trusted_family),
            )
            hybrid_default_focus_rerank_stage_entered_with_family_confidence = float(family_conf)
            if not can_focus:
                hybrid_default_focus_rerank_blocked_by_low_family_confidence = True
                latencies_ms["intra_doc_focus_rerank"] = 0
            else:
                t_focus = time.perf_counter()
                reranked, focus_debug = self._intra_doc_focus_rerank(
                    query=query_for_bm25,
                    candidates=reranked,
                    query_subclass=query_rewrite_subclass,
                    legal_concept_tags=query_rewrite_legal_concept_tags,
                    actor_terms=query_rewrite_actor_terms,
                    action_terms=query_rewrite_action_terms,
                    object_terms=query_rewrite_object_terms,
                    focus_terms=query_rewrite_focus_terms,
                )
                latencies_ms["intra_doc_focus_rerank"] = round((time.perf_counter() - t_focus) * 1000)
                hybrid_default_intra_doc_focus_rerank_used = bool(focus_debug.get("applied", False))
                hybrid_default_intra_doc_focus_target_doc_ids = [
                    str(v) for v in (focus_debug.get("target_doc_ids") or []) if str(v).strip()
                ]
                hybrid_default_intra_doc_focus_target_family_keys = [
                    str(v) for v in (focus_debug.get("target_family_keys") or []) if str(v).strip()
                ]
                hybrid_default_intra_doc_focus_rank_first_doc = int(focus_debug.get("rank_first_doc", 0))
                hybrid_default_intra_doc_focus_rank_first_family = int(focus_debug.get("rank_first_family", 0))
                hybrid_default_intra_doc_focus_candidates = int(focus_debug.get("candidates", 0))
                hybrid_default_intra_doc_focus_promoted = int(focus_debug.get("promoted", 0))
                hybrid_default_focus_rerank_used = bool(focus_debug.get("applied", False))
                hybrid_default_focus_rerank_candidate_chunks = int(focus_debug.get("candidates", 0))
                hybrid_default_focus_heading_match_top1 = float(focus_debug.get("top1_heading_match", 0.0))
                hybrid_default_focus_actor_action_top1 = float(focus_debug.get("top1_actor_action_match", 0.0))
                hybrid_default_focus_concept_match_top1 = float(focus_debug.get("top1_concept_match", 0.0))

            for idx, row in enumerate(reranked, start=1):
                fam = str(row.get("law_family_key", "")).strip() or self._canonical_law_family_key(
                    str(row.get("title", ""))
                )
                if fam and fam in set(hybrid_default_selected_family_keys):
                    rank_first_family_post_focus = idx
                    break
            if reranked:
                rank_first_doc_role_post_focus = self._doc_role_of_item(reranked[0])
        latencies_ms["rerank"] = round((time.perf_counter() - t_rerank) * 1000)

        final_results = reranked[:final_top_k]
        for cut in reranked[final_top_k:]:
            filtered_out.append({
                "chunk_id": cut.get("chunk_id", ""),
                "reason": "top_k_cutoff",
                "rank_after_rerank": cut.get("rank"),
            })

        t_hydrate_final = time.perf_counter()
        final_results = self._bm25.hydrate_rows(final_results, include_text=True)
        latencies_ms["hydrate_final"] = round((time.perf_counter() - t_hydrate_final) * 1000)

        latencies_ms["total"] = round((time.perf_counter() - t_total) * 1000)

        routing_snapshot = {
            "route": route,
            "initial_route": initial_route,
            "effective_route": effective_route,
            "query_raw": query,
            "query_for_bm25": query_for_bm25,
            "query_for_vector": query_for_vector,
            "legal_refs": refs,
            "structured_lookup_used": structured_lookup_used,
            "structured_lookup_hit_count": structured_lookup_hit_count,
            "structured_lookup_fallback": structured_lookup_fallback,
            "bm25_mode": bm25_mode,
            "bm25_top_k_effective": int(bm25_exec_top_k),
            "vector_top_k_effective": int(vector_top_k_effective),
            "vector_skipped_by_route": vector_skipped_by_route,
            "vector_skipped_by_quality_gate": vector_skipped_by_quality_gate,
            "vector_timeout_ms": vector_timeout_ms,
            "vector_budget_ms": vector_budget_ms,
            "vector_wall_ms": int(vector_wall_ms),
            "vector_join_wait_ms": int(vector_join_wait_ms),
            "vector_deadline_hit": bool(vector_deadline_hit),
            "vector_result_used": bool(vector_result_used),
            "vector_cancelled": bool(vector_cancelled),
            "vector_timed_out": vector_timed_out,
            "structured_lookup_match_type": structured_lookup_match_type,
            "structured_lookup_confidence": structured_lookup_confidence,
            "structured_lookup_candidates": structured_lookup_candidates,
            "document_lookup_used": document_lookup_used,
            "document_lookup_ms": int(document_lookup_ms),
            "document_lookup_matched": document_lookup_matched,
            "document_lookup_match_type": document_lookup_match_type,
            "document_lookup_confidence": float(document_lookup_confidence),
            "document_lookup_doc_ids": list(document_lookup_doc_ids),
            "document_lookup_doc_ids_count": len(document_lookup_doc_ids),
            "document_lookup_rows": list(document_lookup_rows),
            "document_lookup_top_matches": list(document_lookup_top_matches),
            "document_lookup_distinct_top_matches": list(document_lookup_distinct_top_matches),
            "document_lookup_top1_confidence": float(document_lookup_top1_confidence),
            "document_lookup_top2_confidence": float(document_lookup_top2_confidence),
            "document_lookup_margin": float(document_lookup_margin),
            "document_lookup_selected_route": document_lookup_selected_route,
            "document_lookup_selected_doc_ids": list(document_lookup_selected_doc_ids),
            "document_lookup_selected_doc_numbers": list(document_lookup_selected_doc_numbers),
            "document_lookup_selected_reason": document_lookup_selected_reason,
            "document_lookup_scope_reason_detail": document_lookup_scope_reason_detail,
            "document_lookup_parsed_law_name": document_lookup_parsed_law_name,
            "document_lookup_parsed_topic_tail": document_lookup_parsed_topic_tail,
            "document_lookup_law_specificity": float(document_lookup_law_specificity),
            "document_lookup_soft_scope_applied": bool(document_lookup_soft_scope_applied),
            "document_lookup_fallback_to_loose": bool(document_lookup_fallback_to_loose),
            "fallback_retry_used": bool(fallback_retry_used),
            "fallback_from_route": fallback_from_route,
            "fallback_to_route": fallback_to_route,
            "fallback_retry_reason": fallback_retry_reason,
            "query_rewrite_used": bool(query_rewrite_used),
            "query_rewrite_ms": int(query_rewrite_ms),
            "query_rewrite_confidence": float(query_rewrite_confidence),
            "query_too_vague": bool(query_rewrite_too_vague),
            "query_rewrite_clean": query_rewrite_clean,
            "query_rewrite_lexical": query_rewrite_lexical,
            "query_rewrite_semantic": query_rewrite_semantic,
            "query_rewrite_focus_terms": list(query_rewrite_focus_terms),
            "query_rewrite_fillers_removed": list(query_rewrite_fillers_removed),
            "query_rewrite_phrase_repairs": list(query_rewrite_phrase_repairs),
            "query_rewrite_token_classes": dict(query_rewrite_token_classes),
            "query_rewrite_legal_anchor_guess": query_rewrite_legal_anchor_guess,
            "query_rewrite_legal_anchor_guess_list": list(query_rewrite_legal_anchor_guess_list),
            "query_rewrite_doc_type_prior": list(query_rewrite_doc_type_prior),
            "query_rewrite_exclude_doc_type_hint": list(query_rewrite_exclude_doc_type_hint),
            "query_rewrite_topic_class": query_rewrite_topic_class,
            "query_rewrite_subclass": query_rewrite_subclass,
            "query_rewrite_legal_concept_tags": list(query_rewrite_legal_concept_tags),
            "query_rewrite_actor_terms": list(query_rewrite_actor_terms),
            "query_rewrite_action_terms": list(query_rewrite_action_terms),
            "query_rewrite_object_terms": list(query_rewrite_object_terms),
            "query_rewrite_qualifier_terms": list(query_rewrite_qualifier_terms),
            "query_rewrite_vagueness_level": query_rewrite_vagueness_level,
            "query_rewrite_concept_confidence": float(query_rewrite_concept_confidence),
            "query_rewrite_v3_used": bool(query_rewrite_v3_used),
            "query_rewrite_is_concept_query": bool(query_rewrite_is_concept_query),
            "query_rewrite_is_topic_broad": bool(query_rewrite_is_topic_broad),
            "query_rewrite_lexical_core": query_rewrite_lexical_core,
            "query_rewrite_lexical_expanded": query_rewrite_lexical_expanded,
            "query_rewrite_concept_seed_query": query_rewrite_concept_seed_query,
            "query_rewrite_title_anchor_query": query_rewrite_title_anchor_query,
            "query_rewrite_risk": query_rewrite_risk,
            "query_rewrite_lexical_token_count": int(query_rewrite_lexical_token_count),
            "query_rewrite_lexical_is_weak": bool(query_rewrite_lexical_is_weak),
            "query_rewrite_weak_query_abort": bool(query_rewrite_weak_query_abort),
            "query_rewrite_weak_query_abort_reasons": list(query_rewrite_weak_query_abort_reasons),
            "query_rewrite_lexical_quality_flags": list(query_rewrite_lexical_quality_flags),
            "query_rewrite_intent_template_hits": list(query_rewrite_intent_template_hits),
            "query_rewrite_lexical_expansion_used": list(query_rewrite_lexical_expansion_used),
            "hybrid_default_mode": hybrid_default_mode,
            "hybrid_default_candidate_family_count": int(hybrid_default_candidate_family_count),
            "hybrid_default_candidate_doc_count": int(hybrid_default_candidate_doc_count),
            "hybrid_default_title_anchor_used": bool(hybrid_default_title_anchor_used),
            "hybrid_default_title_anchor_hits": int(hybrid_default_title_anchor_hits),
            "hybrid_default_doc_aggregation_used": bool(hybrid_default_doc_aggregation_used),
            "hybrid_default_doc_score_top1": float(hybrid_default_doc_score_top1),
            "hybrid_default_doc_score_top2": float(hybrid_default_doc_score_top2),
            "hybrid_default_doc_score_margin": float(hybrid_default_doc_score_margin),
            "hybrid_default_doc_top1_family": hybrid_default_doc_top1_family,
            "hybrid_default_doc_top2_family": hybrid_default_doc_top2_family,
            "hybrid_default_vector_used_for_doc_recall": bool(
                hybrid_default_vector_used_for_doc_recall
            ),
            "hybrid_default_vector_support_same_family": bool(
                hybrid_default_vector_support_same_family
            ),
            "hybrid_default_genericity_penalty_top1": float(
                hybrid_default_genericity_penalty_top1
            ),
            "hybrid_default_genericity_penalty_applied_count": int(
                hybrid_default_genericity_penalty_applied_count
            ),
            "hybrid_default_focus_rerank_used": bool(hybrid_default_focus_rerank_used),
            "hybrid_default_focus_rerank_candidate_chunks": int(
                hybrid_default_focus_rerank_candidate_chunks
            ),
            "hybrid_default_focus_heading_match_top1": float(
                hybrid_default_focus_heading_match_top1
            ),
            "hybrid_default_focus_actor_action_top1": float(
                hybrid_default_focus_actor_action_top1
            ),
            "hybrid_default_focus_concept_match_top1": float(
                hybrid_default_focus_concept_match_top1
            ),
            "hybrid_default_selected_family_keys": list(hybrid_default_selected_family_keys),
            "hybrid_default_selected_doc_ids": list(hybrid_default_selected_doc_ids),
            "hybrid_default_family_search_used": bool(hybrid_default_family_search_used),
            "hybrid_default_family_search_hits": int(hybrid_default_family_search_hits),
            "hybrid_default_family_candidate_count_pre_filter": int(
                hybrid_default_family_candidate_count_pre_filter
            ),
            "hybrid_default_family_candidate_count_post_filter": int(
                hybrid_default_family_candidate_count_post_filter
            ),
            "hybrid_default_family_score_top1": float(hybrid_default_family_score_top1),
            "hybrid_default_family_score_top2": float(hybrid_default_family_score_top2),
            "hybrid_default_family_score_margin": float(hybrid_default_family_score_margin),
            "hybrid_default_family_identity_score_top1": float(
                hybrid_default_family_identity_score_top1
            ),
            "hybrid_default_family_identity_score_top2": float(
                hybrid_default_family_identity_score_top2
            ),
            "hybrid_default_family_identity_margin": float(
                hybrid_default_family_identity_margin
            ),
            "hybrid_default_family_top1_doc_role": hybrid_default_family_top1_doc_role,
            "hybrid_default_family_top1_is_implementation": bool(
                hybrid_default_family_top1_is_implementation
            ),
            "hybrid_default_family_recovery_used": bool(hybrid_default_family_recovery_used),
            "hybrid_default_family_recovery_promoted": int(hybrid_default_family_recovery_promoted),
            "hybrid_default_weak_query_abort_used": bool(hybrid_default_weak_query_abort_used),
            "hybrid_default_no_confident_family_candidates": bool(
                hybrid_default_no_confident_family_candidates
            ),
            "hybrid_default_garbage_family_rejected": bool(hybrid_default_garbage_family_rejected),
            "hybrid_default_garbage_family_reject_reason": hybrid_default_garbage_family_reject_reason,
            "hybrid_default_trusted_family": bool(hybrid_default_trusted_family),
            "hybrid_default_shortlist_after_reject_count": int(
                hybrid_default_shortlist_after_reject_count
            ),
            "hybrid_default_shortlist_after_reject_empty": bool(
                hybrid_default_shortlist_after_reject_empty
            ),
            "hybrid_default_doc_role_prior_applied": bool(hybrid_default_doc_role_prior_applied),
            "hybrid_default_doc_role_prior_top1": hybrid_default_doc_role_prior_top1,
            "hybrid_default_wrong_doc_role_penalty_applied": bool(
                hybrid_default_wrong_doc_role_penalty_applied
            ),
            "hybrid_default_implementation_bridge_used": bool(
                hybrid_default_implementation_bridge_used
            ),
            "hybrid_default_implementation_bridge_hits": int(
                hybrid_default_implementation_bridge_hits
            ),
            "hybrid_default_focus_rerank_blocked_by_low_family_confidence": bool(
                hybrid_default_focus_rerank_blocked_by_low_family_confidence
            ),
            "hybrid_default_focus_rerank_stage_entered_with_family_confidence": float(
                hybrid_default_focus_rerank_stage_entered_with_family_confidence
            ),
            "rank_first_family_pre_focus": int(rank_first_family_pre_focus),
            "rank_first_family_post_focus": int(rank_first_family_post_focus),
            "rank_first_doc_role_pre_focus": rank_first_doc_role_pre_focus,
            "rank_first_doc_role_post_focus": rank_first_doc_role_post_focus,
            "hybrid_default_bm25_query_fallback_used": bool(hybrid_default_bm25_query_fallback_used),
            "hybrid_default_bm25_query_fallback_reason": hybrid_default_bm25_query_fallback_reason,
            "hybrid_default_hard_doc_type_filter": bool(hybrid_default_hard_doc_type_filter),
            "hybrid_default_hard_doc_type_allowlist": list(hybrid_default_hard_doc_type_allowlist),
            "hybrid_default_concept_policy_applied": bool(hybrid_default_concept_policy_applied),
            "hybrid_default_concept_vector_optional": bool(hybrid_default_concept_vector_optional),
            "hybrid_default_intra_doc_focus_rerank_used": bool(
                hybrid_default_intra_doc_focus_rerank_used
            ),
            "hybrid_default_intra_doc_focus_target_doc_ids": list(
                hybrid_default_intra_doc_focus_target_doc_ids
            ),
            "hybrid_default_intra_doc_focus_target_family_keys": list(
                hybrid_default_intra_doc_focus_target_family_keys
            ),
            "hybrid_default_intra_doc_focus_rank_first_doc": int(
                hybrid_default_intra_doc_focus_rank_first_doc
            ),
            "hybrid_default_intra_doc_focus_rank_first_family": int(
                hybrid_default_intra_doc_focus_rank_first_family
            ),
            "hybrid_default_intra_doc_focus_candidates": int(
                hybrid_default_intra_doc_focus_candidates
            ),
            "hybrid_default_intra_doc_focus_promoted": int(
                hybrid_default_intra_doc_focus_promoted
            ),
            "hybrid_default_scope_fallback_used": bool(hybrid_default_scope_fallback_used),
            "hybrid_default_scope_fallback_reason": hybrid_default_scope_fallback_reason,
            "hybrid_default_scope_fallback_hits": int(hybrid_default_scope_fallback_hits),
            "hybrid_default_scope_fallback_bm25_only_used": bool(
                hybrid_default_scope_fallback_bm25_only_used
            ),
            "hybrid_default_scope_fallback_bm25_only_hits": int(
                hybrid_default_scope_fallback_bm25_only_hits
            ),
            "hybrid_default_scope_fallback_bm25_only_latency_ms": int(
                hybrid_default_scope_fallback_bm25_only_latency_ms
            ),
            "hybrid_default_scope_fallback_bm25_only_query": hybrid_default_scope_fallback_bm25_only_query,
            "hybrid_default_scope_fallback_bm25_only_query_reason": (
                hybrid_default_scope_fallback_bm25_only_query_reason
            ),
            "hybrid_default_scope_fallback_vector_reused": bool(
                hybrid_default_scope_fallback_vector_reused
            ),
            "hybrid_default_scope_fallback_second_vector_called": bool(
                hybrid_default_scope_fallback_second_vector_called
            ),
            "hybrid_default_scope_fallback_first_pass_hits": int(
                hybrid_default_scope_fallback_first_pass_hits
            ),
            "hybrid_default_scope_fallback_first_pass_top10": list(
                hybrid_default_scope_fallback_first_pass_top10
            ),
            "hybrid_anchor_scope_used": bool(hybrid_anchor_scope_used),
            "hybrid_anchor_scope_confidence": float(hybrid_anchor_scope_confidence),
            "hybrid_anchor_scope_doc_ids_count": len(hybrid_anchor_scope_doc_ids),
            "hybrid_anchor_scope_doc_ids": list(hybrid_anchor_scope_doc_ids),
            "hybrid_anchor_scope_reason": hybrid_anchor_scope_reason,
            "document_lookup_candidate_doc_ids_original_count": int(
                document_lookup_candidate_doc_ids_original_count
            ),
            "document_lookup_candidate_doc_ids_trimmed": bool(
                document_lookup_candidate_doc_ids_trimmed
            ),
            "candidate_route_early_exit_used": bool(candidate_route_early_exit_used),
            "candidate_route_early_exit_reason": candidate_route_early_exit_reason,
            "candidate_route_early_exit_score_gap": float(candidate_route_early_exit_score_gap),
            "hybrid_doc_gate_applied": bool(hybrid_doc_gate_applied),
            "hybrid_doc_gate_bm25_before": int(hybrid_doc_gate_bm25_before),
            "hybrid_doc_gate_bm25_after": int(hybrid_doc_gate_bm25_after),
            "hybrid_doc_gate_vector_before": int(hybrid_doc_gate_vector_before),
            "hybrid_doc_gate_vector_after": int(hybrid_doc_gate_vector_after),
            "hybrid_doc_gate_removed_bm25": int(hybrid_doc_gate_removed_bm25),
            "hybrid_doc_gate_removed_vector": int(hybrid_doc_gate_removed_vector),
            "hybrid_doc_gate_override_kept_original": bool(hybrid_doc_gate_override_kept_original),
            "hybrid_doc_gate_bm25_filtered_before_override": int(
                hybrid_doc_gate_bm25_filtered_before_override
            ),
            "hybrid_doc_gate_vector_filtered_before_override": int(
                hybrid_doc_gate_vector_filtered_before_override
            ),
            "allowed_doc_ids_count": len(allowed_doc_ids),
            "allowed_doc_ids": sorted(allowed_doc_ids),
            "bm25_scoped": bool(allowed_doc_ids),
            "vector_scoped": bool(allowed_doc_ids),
            "bm25_hits_count": len(bm25_results),
            "vector_hits_count": len(vector_results),
            "total_candidates_after_scope": len(bm25_results) + len(vector_results),
            "scope_zero_hits": bool((len(bm25_results) + len(vector_results)) == 0),
            "scoped_filter_broken": bool(scoped_filter_broken),
            "scoped_filter_broken_doc_ids": list(scoped_filter_broken_doc_ids),
            "query_token_count": len(tokenize_for_query(query).split()),
        }

        return {
            "bm25_hits": bm25_results,
            "vector_hits": vector_results,
            "rrf_hits": fused,
            "deduped_hits": deduped,
            "reranked_hits": reranked,
            "final_results": final_results,
            "filtered_out": filtered_out,
            "latencies_ms": latencies_ms,
            "bm25_debug": bm25_debug,
            "routing": routing_snapshot,
            "execution": {
                "parallel": executed_parallel,
                "vector_available": vector_available,
                "vector_skipped_by_route": vector_skipped_by_route,
                "vector_skipped_by_quality_gate": vector_skipped_by_quality_gate,
                "vector_timeout_ms": vector_timeout_ms,
                "bm25_top_k_effective": int(bm25_exec_top_k),
                "vector_top_k_effective": int(vector_top_k_effective),
                "vector_budget_ms": vector_budget_ms,
                "vector_wall_ms": int(vector_wall_ms),
                "vector_join_wait_ms": int(vector_join_wait_ms),
                "vector_deadline_hit": bool(vector_deadline_hit),
                "vector_result_used": bool(vector_result_used),
                "vector_cancelled": bool(vector_cancelled),
                "vector_timed_out": vector_timed_out,
                "bm25_scoped": bool(allowed_doc_ids),
                "vector_scoped": bool(allowed_doc_ids),
                "allowed_doc_ids_count": len(allowed_doc_ids),
                "allowed_doc_ids": sorted(allowed_doc_ids),
                "document_lookup_used": document_lookup_used,
                "document_lookup_matched": document_lookup_matched,
                "document_lookup_confidence": float(document_lookup_confidence),
                "document_lookup_match_type": document_lookup_match_type,
                "document_lookup_selected_doc_ids": list(document_lookup_selected_doc_ids),
                "document_lookup_selected_doc_numbers": list(document_lookup_selected_doc_numbers),
                "document_lookup_soft_scope_applied": bool(document_lookup_soft_scope_applied),
                "fallback_retry_used": bool(fallback_retry_used),
                "fallback_from_route": fallback_from_route,
                "fallback_to_route": fallback_to_route,
                "fallback_retry_reason": fallback_retry_reason,
                "query_rewrite_used": bool(query_rewrite_used),
                "query_too_vague": bool(query_rewrite_too_vague),
                "query_rewrite_confidence": float(query_rewrite_confidence),
                "query_rewrite_token_classes": dict(query_rewrite_token_classes),
                "query_rewrite_legal_anchor_guess": query_rewrite_legal_anchor_guess,
                "query_rewrite_legal_anchor_guess_list": list(query_rewrite_legal_anchor_guess_list),
                "query_rewrite_doc_type_prior": list(query_rewrite_doc_type_prior),
                "query_rewrite_exclude_doc_type_hint": list(query_rewrite_exclude_doc_type_hint),
                "query_rewrite_topic_class": query_rewrite_topic_class,
                "query_rewrite_subclass": query_rewrite_subclass,
                "query_rewrite_legal_concept_tags": list(query_rewrite_legal_concept_tags),
                "query_rewrite_actor_terms": list(query_rewrite_actor_terms),
                "query_rewrite_action_terms": list(query_rewrite_action_terms),
                "query_rewrite_object_terms": list(query_rewrite_object_terms),
                "query_rewrite_qualifier_terms": list(query_rewrite_qualifier_terms),
                "query_rewrite_vagueness_level": query_rewrite_vagueness_level,
                "query_rewrite_concept_confidence": float(query_rewrite_concept_confidence),
                "query_rewrite_v3_used": bool(query_rewrite_v3_used),
                "query_rewrite_is_concept_query": bool(query_rewrite_is_concept_query),
                "query_rewrite_is_topic_broad": bool(query_rewrite_is_topic_broad),
                "query_rewrite_lexical_core": query_rewrite_lexical_core,
                "query_rewrite_lexical_expanded": query_rewrite_lexical_expanded,
                "query_rewrite_concept_seed_query": query_rewrite_concept_seed_query,
                "query_rewrite_title_anchor_query": query_rewrite_title_anchor_query,
                "query_rewrite_risk": query_rewrite_risk,
                "query_rewrite_lexical_token_count": int(query_rewrite_lexical_token_count),
                "query_rewrite_lexical_is_weak": bool(query_rewrite_lexical_is_weak),
                "query_rewrite_weak_query_abort": bool(query_rewrite_weak_query_abort),
                "query_rewrite_weak_query_abort_reasons": list(query_rewrite_weak_query_abort_reasons),
                "query_rewrite_lexical_quality_flags": list(query_rewrite_lexical_quality_flags),
                "query_rewrite_intent_template_hits": list(query_rewrite_intent_template_hits),
                "query_rewrite_lexical_expansion_used": list(query_rewrite_lexical_expansion_used),
                "hybrid_default_mode": hybrid_default_mode,
                "hybrid_default_candidate_family_count": int(hybrid_default_candidate_family_count),
                "hybrid_default_candidate_doc_count": int(hybrid_default_candidate_doc_count),
                "hybrid_default_title_anchor_used": bool(hybrid_default_title_anchor_used),
                "hybrid_default_title_anchor_hits": int(hybrid_default_title_anchor_hits),
                "hybrid_default_doc_aggregation_used": bool(hybrid_default_doc_aggregation_used),
                "hybrid_default_doc_score_top1": float(hybrid_default_doc_score_top1),
                "hybrid_default_doc_score_top2": float(hybrid_default_doc_score_top2),
                "hybrid_default_doc_score_margin": float(hybrid_default_doc_score_margin),
                "hybrid_default_doc_top1_family": hybrid_default_doc_top1_family,
                "hybrid_default_doc_top2_family": hybrid_default_doc_top2_family,
                "hybrid_default_vector_used_for_doc_recall": bool(
                    hybrid_default_vector_used_for_doc_recall
                ),
                "hybrid_default_vector_support_same_family": bool(
                    hybrid_default_vector_support_same_family
                ),
                "hybrid_default_genericity_penalty_top1": float(
                    hybrid_default_genericity_penalty_top1
                ),
                "hybrid_default_genericity_penalty_applied_count": int(
                    hybrid_default_genericity_penalty_applied_count
                ),
                "hybrid_default_focus_rerank_used": bool(hybrid_default_focus_rerank_used),
                "hybrid_default_focus_rerank_candidate_chunks": int(
                    hybrid_default_focus_rerank_candidate_chunks
                ),
                "hybrid_default_focus_heading_match_top1": float(
                    hybrid_default_focus_heading_match_top1
                ),
                "hybrid_default_focus_actor_action_top1": float(
                    hybrid_default_focus_actor_action_top1
                ),
                "hybrid_default_focus_concept_match_top1": float(
                    hybrid_default_focus_concept_match_top1
                ),
                "hybrid_default_selected_family_keys": list(hybrid_default_selected_family_keys),
                "hybrid_default_selected_doc_ids": list(hybrid_default_selected_doc_ids),
                "hybrid_default_family_search_used": bool(hybrid_default_family_search_used),
                "hybrid_default_family_search_hits": int(hybrid_default_family_search_hits),
                "hybrid_default_family_candidate_count_pre_filter": int(
                    hybrid_default_family_candidate_count_pre_filter
                ),
                "hybrid_default_family_candidate_count_post_filter": int(
                    hybrid_default_family_candidate_count_post_filter
                ),
                "hybrid_default_family_score_top1": float(hybrid_default_family_score_top1),
                "hybrid_default_family_score_top2": float(hybrid_default_family_score_top2),
                "hybrid_default_family_score_margin": float(hybrid_default_family_score_margin),
                "hybrid_default_family_identity_score_top1": float(
                    hybrid_default_family_identity_score_top1
                ),
                "hybrid_default_family_identity_score_top2": float(
                    hybrid_default_family_identity_score_top2
                ),
                "hybrid_default_family_identity_margin": float(
                    hybrid_default_family_identity_margin
                ),
                "hybrid_default_family_top1_doc_role": hybrid_default_family_top1_doc_role,
                "hybrid_default_family_top1_is_implementation": bool(
                    hybrid_default_family_top1_is_implementation
                ),
                "hybrid_default_family_recovery_used": bool(hybrid_default_family_recovery_used),
                "hybrid_default_family_recovery_promoted": int(
                    hybrid_default_family_recovery_promoted
                ),
                "hybrid_default_weak_query_abort_used": bool(hybrid_default_weak_query_abort_used),
                "hybrid_default_no_confident_family_candidates": bool(
                    hybrid_default_no_confident_family_candidates
                ),
                "hybrid_default_garbage_family_rejected": bool(hybrid_default_garbage_family_rejected),
                "hybrid_default_garbage_family_reject_reason": hybrid_default_garbage_family_reject_reason,
                "hybrid_default_trusted_family": bool(hybrid_default_trusted_family),
                "hybrid_default_shortlist_after_reject_count": int(
                    hybrid_default_shortlist_after_reject_count
                ),
                "hybrid_default_shortlist_after_reject_empty": bool(
                    hybrid_default_shortlist_after_reject_empty
                ),
                "hybrid_default_doc_role_prior_applied": bool(hybrid_default_doc_role_prior_applied),
                "hybrid_default_doc_role_prior_top1": hybrid_default_doc_role_prior_top1,
                "hybrid_default_wrong_doc_role_penalty_applied": bool(
                    hybrid_default_wrong_doc_role_penalty_applied
                ),
                "hybrid_default_implementation_bridge_used": bool(
                    hybrid_default_implementation_bridge_used
                ),
                "hybrid_default_implementation_bridge_hits": int(
                    hybrid_default_implementation_bridge_hits
                ),
                "hybrid_default_focus_rerank_blocked_by_low_family_confidence": bool(
                    hybrid_default_focus_rerank_blocked_by_low_family_confidence
                ),
                "hybrid_default_focus_rerank_stage_entered_with_family_confidence": float(
                    hybrid_default_focus_rerank_stage_entered_with_family_confidence
                ),
                "rank_first_family_pre_focus": int(rank_first_family_pre_focus),
                "rank_first_family_post_focus": int(rank_first_family_post_focus),
                "rank_first_doc_role_pre_focus": rank_first_doc_role_pre_focus,
                "rank_first_doc_role_post_focus": rank_first_doc_role_post_focus,
                "hybrid_default_bm25_query_fallback_used": bool(hybrid_default_bm25_query_fallback_used),
                "hybrid_default_bm25_query_fallback_reason": hybrid_default_bm25_query_fallback_reason,
                "hybrid_default_hard_doc_type_filter": bool(hybrid_default_hard_doc_type_filter),
                "hybrid_default_hard_doc_type_allowlist": list(hybrid_default_hard_doc_type_allowlist),
                "hybrid_default_concept_policy_applied": bool(hybrid_default_concept_policy_applied),
                "hybrid_default_concept_vector_optional": bool(hybrid_default_concept_vector_optional),
                "hybrid_default_intra_doc_focus_rerank_used": bool(
                    hybrid_default_intra_doc_focus_rerank_used
                ),
                "hybrid_default_intra_doc_focus_target_doc_ids": list(
                    hybrid_default_intra_doc_focus_target_doc_ids
                ),
                "hybrid_default_intra_doc_focus_target_family_keys": list(
                    hybrid_default_intra_doc_focus_target_family_keys
                ),
                "hybrid_default_intra_doc_focus_rank_first_doc": int(
                    hybrid_default_intra_doc_focus_rank_first_doc
                ),
                "hybrid_default_intra_doc_focus_rank_first_family": int(
                    hybrid_default_intra_doc_focus_rank_first_family
                ),
                "hybrid_default_intra_doc_focus_candidates": int(
                    hybrid_default_intra_doc_focus_candidates
                ),
                "hybrid_default_intra_doc_focus_promoted": int(
                    hybrid_default_intra_doc_focus_promoted
                ),
                "hybrid_default_scope_fallback_used": bool(hybrid_default_scope_fallback_used),
                "hybrid_default_scope_fallback_reason": hybrid_default_scope_fallback_reason,
                "hybrid_default_scope_fallback_hits": int(hybrid_default_scope_fallback_hits),
                "hybrid_default_scope_fallback_bm25_only_used": bool(
                    hybrid_default_scope_fallback_bm25_only_used
                ),
                "hybrid_default_scope_fallback_bm25_only_hits": int(
                    hybrid_default_scope_fallback_bm25_only_hits
                ),
                "hybrid_default_scope_fallback_bm25_only_latency_ms": int(
                    hybrid_default_scope_fallback_bm25_only_latency_ms
                ),
                "hybrid_default_scope_fallback_bm25_only_query": (
                    hybrid_default_scope_fallback_bm25_only_query
                ),
                "hybrid_default_scope_fallback_bm25_only_query_reason": (
                    hybrid_default_scope_fallback_bm25_only_query_reason
                ),
                "hybrid_default_scope_fallback_vector_reused": bool(
                    hybrid_default_scope_fallback_vector_reused
                ),
                "hybrid_default_scope_fallback_second_vector_called": bool(
                    hybrid_default_scope_fallback_second_vector_called
                ),
                "hybrid_default_scope_fallback_first_pass_hits": int(
                    hybrid_default_scope_fallback_first_pass_hits
                ),
                "hybrid_default_scope_fallback_first_pass_top10": list(
                    hybrid_default_scope_fallback_first_pass_top10
                ),
                "hybrid_anchor_scope_used": bool(hybrid_anchor_scope_used),
                "hybrid_anchor_scope_confidence": float(hybrid_anchor_scope_confidence),
                "hybrid_anchor_scope_doc_ids_count": len(hybrid_anchor_scope_doc_ids),
                "hybrid_anchor_scope_doc_ids": list(hybrid_anchor_scope_doc_ids),
                "hybrid_anchor_scope_reason": hybrid_anchor_scope_reason,
                "document_lookup_candidate_doc_ids_original_count": int(
                    document_lookup_candidate_doc_ids_original_count
                ),
                "document_lookup_candidate_doc_ids_trimmed": bool(
                    document_lookup_candidate_doc_ids_trimmed
                ),
                "candidate_route_early_exit_used": bool(candidate_route_early_exit_used),
                "candidate_route_early_exit_reason": candidate_route_early_exit_reason,
                "candidate_route_early_exit_score_gap": float(candidate_route_early_exit_score_gap),
                "hybrid_doc_gate_applied": bool(hybrid_doc_gate_applied),
                "hybrid_doc_gate_bm25_before": int(hybrid_doc_gate_bm25_before),
                "hybrid_doc_gate_bm25_after": int(hybrid_doc_gate_bm25_after),
                "hybrid_doc_gate_vector_before": int(hybrid_doc_gate_vector_before),
                "hybrid_doc_gate_vector_after": int(hybrid_doc_gate_vector_after),
                "hybrid_doc_gate_removed_bm25": int(hybrid_doc_gate_removed_bm25),
                "hybrid_doc_gate_removed_vector": int(hybrid_doc_gate_removed_vector),
                "hybrid_doc_gate_override_kept_original": bool(hybrid_doc_gate_override_kept_original),
                "hybrid_doc_gate_bm25_filtered_before_override": int(
                    hybrid_doc_gate_bm25_filtered_before_override
                ),
                "hybrid_doc_gate_vector_filtered_before_override": int(
                    hybrid_doc_gate_vector_filtered_before_override
                ),
                "bm25_hits_count": len(bm25_results),
                "vector_hits_count": len(vector_results),
                "scoped_filter_broken": bool(scoped_filter_broken),
                "scoped_filter_broken_doc_ids": list(scoped_filter_broken_doc_ids),
            },
        }

    @staticmethod
    def _bm25_mode_for_route(route: str) -> str:
        if route in {"law_anchored_hybrid", "law_anchored_hybrid_loose"}:
            return "law_anchored_broad"
        if route in {"doc_scoped_hybrid", "doc_scoped_hybrid_exact_doc", "doc_scoped_hybrid_candidate_docs"}:
            return "broad"
        if route == "hybrid_default":
            return "broad"
        return "narrow"

    @staticmethod
    def _is_generic_lexical_query(query_text: str) -> bool:
        tokens = tokenize_for_query(str(query_text or "")).split()
        if not tokens:
            return True
        if len(tokens) <= 2:
            return True
        generic_tokens = {
            "quy",
            "dinh",
            "chung",
            "ve",
            "la",
            "gi",
            "nhu",
            "the",
            "nao",
            "khi",
            "ap",
            "dung",
            "thuc",
            "te",
            "can",
            "luu",
            "y",
            "nhung",
            "trong",
        }
        topical = [tok for tok in tokens if tok not in generic_tokens]
        return len(topical) <= 1

    @classmethod
    def _choose_hybrid_default_fallback_query(
        cls,
        *,
        raw_query: str,
        clean_query: str,
        lexical_query: str,
        intent_templates: list[str],
        focus_terms: list[str],
    ) -> tuple[str, str]:
        lexical_norm = tokenize_for_query(lexical_query)
        clean_norm = tokenize_for_query(clean_query)
        raw_norm = tokenize_for_query(raw_query)
        intent_norm = [
            " ".join(tokenize_for_query(str(v)).split())
            for v in (intent_templates or [])
            if str(v).strip()
        ]
        intent_norm = [v for v in intent_norm if v]

        reason = "rewrite_lexical"
        base_query = lexical_norm or clean_norm or raw_norm
        if not lexical_norm or cls._is_generic_lexical_query(lexical_norm):
            base_query = clean_norm or lexical_norm or raw_norm
            reason = "rewrite_clean"
        action_tokens = {
            "xu",
            "ly",
            "ky",
            "luat",
            "tham",
            "quyen",
            "dieu",
            "kien",
            "trach",
            "nhiem",
            "hieu",
            "luc",
            "tuyen",
            "dung",
            "cham",
            "dut",
            "nghia",
            "vu",
        }
        lexical_token_set = set(tokenize_for_query(lexical_norm).split())
        clean_token_set = set(tokenize_for_query(clean_norm).split())
        if (
            lexical_norm
            and clean_norm
            and not (lexical_token_set & action_tokens)
            and bool(clean_token_set & action_tokens)
        ):
            base_query = clean_norm
            reason = "rewrite_clean_action_repair"
        if intent_norm and (
            cls._is_generic_lexical_query(base_query)
            or len(tokenize_for_query(base_query).split()) < 4
        ):
            base_query = intent_norm[0]
            reason = "intent_template"
        if not base_query:
            base_query = raw_norm
            reason = "raw_query"

        base_tokens = tokenize_for_query(base_query).split()
        focus_tokens = tokenize_for_query(" ".join(focus_terms or [])).split()
        if (len(base_tokens) < 3 or cls._is_generic_lexical_query(base_query)) and focus_tokens:
            merged_tokens: list[str] = []
            seen: set[str] = set()
            for tok in (base_tokens + focus_tokens):
                if not tok or tok in seen:
                    continue
                seen.add(tok)
                merged_tokens.append(tok)
            if merged_tokens:
                base_query = " ".join(merged_tokens[:12])
                reason = f"{reason}+focus_terms"

        final_query = " ".join(tokenize_for_query(base_query).split()) or raw_norm
        return final_query, reason

    @staticmethod
    def _merge_rows_by_chunk(
        primary_rows: list[dict[str, Any]],
        secondary_rows: list[dict[str, Any]],
        *,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order_counter = 0

        def _score_of(row: dict[str, Any]) -> float:
            return float(row.get("score", row.get("rrf_score", 0.0)))

        for rows in (primary_rows or [], secondary_rows or []):
            for row in rows:
                chunk_id = str(row.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                current = merged.get(chunk_id)
                if current is None or _score_of(row) > _score_of(current):
                    updated = dict(row)
                    updated["_order"] = order_counter
                    merged[chunk_id] = updated
                order_counter += 1

        out = sorted(
            merged.values(),
            key=lambda row: (
                float(row.get("score", row.get("rrf_score", 0.0))),
                -int(row.get("_order", 0)),
            ),
            reverse=True,
        )
        for idx, row in enumerate(out):
            row["rank"] = idx
            row.pop("_order", None)
        return out[: max(1, int(limit))]

    @staticmethod
    def _compact_hits_for_eval(hits: list[dict[str, Any]], *, top_k: int = 10) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in (hits or [])[: max(1, int(top_k))]:
            out.append(
                {
                    "chunk_id": str(row.get("chunk_id", "")),
                    "doc_id": str(row.get("doc_id", "")),
                    "title": str(row.get("title", "")),
                    "path": str(row.get("path", "")),
                    "document_number": str(row.get("document_number", "")),
                    "article": str(row.get("article", "")),
                    "clause": str(row.get("clause", "")),
                    "law_family_key": str(row.get("law_family_key", "")),
                    "text": str(row.get("text", ""))[:1200],
                    "rank": int(row.get("rank", 0)),
                }
            )
        return out

    @staticmethod
    def _canonical_law_family_key(text: str) -> str:
        value = tokenize_for_query(str(text or ""))
        if not value:
            return ""

        value = re.sub(r"\bvan\s+ban\s+hop\s+nhat\b", " ", value)
        value = re.sub(r"\b(sua\s+doi|bo\s+sung|hop\s+nhat)\b", " ", value)
        match = re.search(r"\b(bo\s+luat|luat)\b\s+(.+)", value)
        if match:
            value = f"{match.group(1)} {match.group(2)}"
        value = re.sub(r"\b(19|20)\d{2}\b", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _is_local_admin_doc(item: dict[str, Any]) -> bool:
        doc_num = normalize_doc_number(str(item.get("document_number", "")))
        if doc_num:
            if "QD-UBND" in doc_num or "QDHDND" in doc_num:
                return True
            if "NQ-HDND" in doc_num:
                return True
            if "UBND" in doc_num and ("QD" in doc_num or "NQ" in doc_num):
                return True

        hay = tokenize_for_query(" ".join([
            str(item.get("doc_type", "")),
            str(item.get("title", "")),
            str(item.get("path", "")),
            str(item.get("document_number", "")),
        ]))
        if not hay:
            return False

        if re.search(r"\b(luat|bo\s+luat|bo\s+luat|nghi\s+dinh|thong\s+tu)\b", hay):
            return False

        return bool(
            re.search(
                r"\b(quyet\s+dinh|cong\s+van|thong\s+bao|chi\s+thi|ke\s+hoach|quy\s+che)\b",
                hay,
            )
        )

    @staticmethod
    def _doc_type_bucket(item: dict[str, Any]) -> str:
        doc_num = normalize_doc_number(str(item.get("document_number", "")))
        if doc_num:
            if "QD-UBND" in doc_num or "QDHDND" in doc_num:
                return "quyet_dinh_ubnd"
            if "NQ-HDND" in doc_num:
                return "van_ban_dia_phuong"
            if "CONG-VAN" in doc_num or doc_num.startswith("CV"):
                return "cong_van"
            if "VBHN" in doc_num:
                return "vbhn"
            if "ND-CP" in doc_num:
                return "nghi_dinh"
            if "TT" in doc_num and "-" in doc_num:
                return "thong_tu"
            if re.search(r"/QH\d{1,2}$", doc_num):
                return "luat"
            if "L-CTN" in doc_num:
                return "bo_luat"

        hay = tokenize_for_query(" ".join([
            str(item.get("doc_type", "")),
            str(item.get("title", "")),
            str(item.get("path", "")),
            str(item.get("document_number", "")),
        ]))
        if not hay:
            return ""
        if re.search(r"\b(cong\s+van)\b", hay):
            return "cong_van"
        if re.search(r"\b(quyet\s+dinh)\b", hay) and re.search(r"\b(ubnd|hdnd|tinh|thanh\s+pho|quan|huyen)\b", hay):
            return "quyet_dinh_ubnd"
        if re.search(r"\b(nghi\s+quyet)\b", hay) and re.search(r"\b(ubnd|hdnd|tinh|thanh\s+pho|quan|huyen)\b", hay):
            return "van_ban_dia_phuong"
        if re.search(r"\b(van\s+ban\s+hop\s+nhat|vbhn)\b", hay):
            return "vbhn"
        if re.search(r"\b(nghi\s+dinh)\b", hay):
            return "nghi_dinh"
        if re.search(r"\b(bo\s+luat)\b", hay):
            return "bo_luat"
        if re.search(r"\b(luat)\b", hay):
            return "luat"
        if re.search(r"\b(thong\s+tu)\b", hay):
            return "thong_tu"
        return ""

    @classmethod
    def _is_preferred_legal_doc_type(cls, item: dict[str, Any]) -> bool:
        bucket = cls._doc_type_bucket(item)
        if bucket in {"luat", "bo_luat", "nghi_dinh", "vbhn", "thong_tu"}:
            return True
        hay = tokenize_for_query(" ".join([
            str(item.get("title", "")),
            str(item.get("doc_type", "")),
            str(item.get("path", "")),
        ]))
        if not hay:
            return False
        return bool(re.search(r"\b(quoc\s+hoi|chinh\s+phu|bo\s+truong|bo\s+tai\s+chinh|bo\s+noi\s+vu)\b", hay))

    def _apply_hybrid_default_doc_gate(
        self,
        rows: list[dict[str, Any]],
        *,
        doc_type_prior: list[str] | None = None,
        exclude_doc_type_hint: list[str] | None = None,
        hard_filter: bool = False,
        hard_allow_buckets: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        before = len(rows)
        if before == 0:
            return rows, {
                "applied": False,
                "before": 0,
                "after": 0,
                "removed": 0,
                "override_kept_original": False,
                "hard_filter": bool(hard_filter),
                "hard_allow_buckets": sorted(hard_allow_buckets or set()),
            }

        prior_set = {
            str(v).strip().lower()
            for v in (doc_type_prior or [])
            if str(v).strip()
        }
        exclude_set = {
            str(v).strip().lower()
            for v in (exclude_doc_type_hint or [])
            if str(v).strip()
        }
        hard_allow_set = {
            str(v).strip().lower()
            for v in (hard_allow_buckets or set())
            if str(v).strip()
        }
        gate_enabled = bool(prior_set or exclude_set or hard_filter)
        if not gate_enabled:
            return rows, {
                "applied": False,
                "before": before,
                "after": before,
                "removed": 0,
                "override_kept_original": False,
                "hard_filter": bool(hard_filter),
                "hard_allow_buckets": sorted(hard_allow_set),
            }

        filtered: list[dict[str, Any]] = []
        for row in rows:
            bucket = self._doc_type_bucket(row)
            local_admin = self._is_local_admin_doc(row)
            preferred_legal = self._is_preferred_legal_doc_type(row)
            keep = True

            if hard_filter:
                if hard_allow_set and bucket not in hard_allow_set:
                    keep = False
                if bucket in {"quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"}:
                    keep = False
                if local_admin and not preferred_legal:
                    keep = False

            if keep and exclude_set and bucket in exclude_set:
                keep = False
            elif keep and local_admin and not preferred_legal:
                keep = False

            if keep:
                filtered.append(row)

        min_keep = max(1, int(getattr(cfg, "HYBRID_DEFAULT_DOC_GATE_MIN_KEEP", 3)))
        override_kept_original = False
        filtered_before_override = len(filtered)
        if before > 0 and not filtered:
            if not hard_filter:
                # Avoid hard-empty retrieval after gate: keep preferred legal docs first.
                preferred = [row for row in rows if self._is_preferred_legal_doc_type(row)]
                if preferred:
                    filtered = preferred[: min(len(preferred), max(min_keep, 5))]
                else:
                    filtered = rows[: min(before, min_keep)]
                override_kept_original = True

        allow_override = bool(getattr(cfg, "HYBRID_DEFAULT_DOC_GATE_ALLOW_OVERRIDE", False))
        if allow_override and (not hard_filter) and len(filtered) < min_keep and before >= min_keep:
            filtered = rows
            override_kept_original = True

        if prior_set and filtered is not rows:
            prioritized = [
                row for row in filtered
                if self._doc_type_bucket(row) in prior_set or self._is_preferred_legal_doc_type(row)
            ]
            if hard_filter and prioritized:
                filtered = prioritized
            elif len(prioritized) >= min_keep:
                filtered = prioritized

        after = len(filtered)
        return filtered, {
            "applied": True,
            "before": before,
            "after": after,
            "removed": max(0, before - after),
            "override_kept_original": override_kept_original,
            "filtered_before_override": int(filtered_before_override),
            "hard_filter": bool(hard_filter),
            "hard_allow_buckets": sorted(hard_allow_set),
        }

    @staticmethod
    def _is_meta_legal_drafting_doc(item: dict[str, Any]) -> bool:
        hay = tokenize_for_query(" ".join([
            str(item.get("title", "")),
            str(item.get("doc_type", "")),
        ]))
        if not hay:
            return False
        return bool(
            re.search(
                r"\b(ban\s+hanh\s+van\s+ban\s+quy\s+pham|van\s+ban\s+quy\s+pham\s+phap\s+luat)\b",
                hay,
            )
        )

    @staticmethod
    def _hit_doc_anchor(item: dict[str, Any], refs: dict[str, str]) -> bool:
        target_doc_number = normalize_doc_number(str(refs.get("document_number", "")))
        target_doc_loose = normalize_doc_number(str(refs.get("document_loose", "")))
        target_doc_short = str(refs.get("document_short", "")).strip().upper()

        item_doc_number = normalize_doc_number(str(item.get("document_number", "")))
        if target_doc_number and item_doc_number == target_doc_number:
            return True
        if target_doc_loose and item_doc_number == target_doc_loose:
            return True

        hay = " ".join([
            str(item.get("doc_id", "")),
            str(item.get("title", "")),
            str(item.get("path", "")),
            str(item.get("document_number", "")),
        ]).upper()

        if target_doc_number and target_doc_number in hay:
            return True
        if target_doc_loose and target_doc_loose in hay:
            return True
        if target_doc_short and re.search(rf"(?<!\d){re.escape(target_doc_short)}(?!\d)", hay):
            return True
        return False

    @classmethod
    def _hit_article_clause_anchor(cls, item: dict[str, Any], refs: dict[str, str]) -> bool:
        target_article = extract_numeric_ref(str(refs.get("article_number", "")))
        target_clause = extract_numeric_ref(str(refs.get("clause_number", "")))

        if not target_article and not target_clause:
            return True

        item_article = extract_numeric_ref(str(item.get("article", "")))
        item_clause = extract_numeric_ref(str(item.get("clause", "")))
        path = str(item.get("path", ""))

        if not item_article:
            m = re.search(r"(?:điều|dieu)\s+(\d+)", path, flags=re.IGNORECASE)
            if m:
                item_article = m.group(1)

        if not item_clause:
            m = re.search(r"(?:khoản|khoan)\s+(\d+)", path, flags=re.IGNORECASE)
            if m:
                item_clause = m.group(1)

        if target_article and item_article != target_article:
            return False
        if target_clause and item_clause != target_clause:
            return False
        return True

    def _should_skip_vector_for_narrow(self, refs: dict[str, str], bm25_results: list[dict[str, Any]]) -> bool:
        if not bm25_results:
            return False

        has_doc_anchor = bool(
            refs.get("document_number")
            or refs.get("document_short")
            or refs.get("document_loose")
        )
        if not has_doc_anchor:
            return False

        top1 = bm25_results[0]
        if not self._hit_doc_anchor(top1, refs):
            return False

        top_k = bm25_results[:3]
        for row in top_k:
            if self._hit_doc_anchor(row, refs) and self._hit_article_clause_anchor(row, refs):
                return True
        return False

    def _should_early_exit_scoped_candidate(
        self,
        *,
        bm25_results: list[dict[str, Any]],
        allowed_doc_ids: set[str],
        lookup_confidence: float,
        lookup_margin: float,
    ) -> tuple[bool, dict[str, Any]]:
        if not bm25_results:
            return False, {"reason": "no_bm25_hits", "score_gap": 0.0}

        allowed_ids = {
            str(doc_id).strip()
            for doc_id in (allowed_doc_ids or set())
            if str(doc_id).strip()
        }
        if not allowed_ids:
            return False, {"reason": "no_allowed_doc_ids", "score_gap": 0.0}

        min_hits = max(2, int(getattr(cfg, "DOC_SCOPED_CANDIDATE_EARLY_EXIT_MIN_HITS", 2)))
        if len(bm25_results) < min_hits:
            return False, {"reason": "insufficient_bm25_hits", "score_gap": 0.0}

        top_rows = bm25_results[: min(4, len(bm25_results))]
        top_doc_ids = [str(row.get("doc_id", "")).strip() for row in top_rows if str(row.get("doc_id", "")).strip()]
        if not top_doc_ids:
            return False, {"reason": "missing_top_doc_ids", "score_gap": 0.0}
        if any(doc_id not in allowed_ids for doc_id in top_doc_ids):
            return False, {"reason": "top_rows_outside_scope", "score_gap": 0.0}

        top1_score = float(top_rows[0].get("score", 0.0))
        top2_score = float(top_rows[1].get("score", 0.0)) if len(top_rows) >= 2 else 0.0
        score_gap = float(top1_score - top2_score)
        distinct_top_docs = {doc_id for doc_id in top_doc_ids}

        conf_th = float(getattr(cfg, "DOC_SCOPED_CANDIDATE_EARLY_EXIT_CONFIDENCE", 0.86))
        margin_th = float(getattr(cfg, "DOC_SCOPED_CANDIDATE_EARLY_EXIT_MARGIN", 0.02))
        gap_th = float(getattr(cfg, "DOC_SCOPED_CANDIDATE_EARLY_EXIT_SCORE_GAP", 0.015))

        if len(allowed_ids) == 1 and score_gap >= -0.01:
            return True, {"reason": "single_doc_scope", "score_gap": score_gap}
        if len(allowed_ids) <= 2 and len(distinct_top_docs) <= 1 and score_gap >= -0.02:
            return True, {"reason": "small_scope_consistent_doc", "score_gap": score_gap}

        if (
            float(lookup_confidence) >= conf_th
            and (float(lookup_margin) >= margin_th or len(distinct_top_docs) <= 1)
            and score_gap >= gap_th
        ):
            return True, {
                "reason": "scoped_bm25_strong_lead",
                "score_gap": score_gap,
            }
        return False, {"reason": "no_strong_lead", "score_gap": score_gap}

    @staticmethod
    def _normalize_key(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _extract_chapter_ref(cls, text: str) -> str:
        m = re.search(r"(?:chương|chuong)\s+([ivxlcdm]+|\d+)", str(text or ""), flags=re.IGNORECASE)
        if not m:
            return ""
        return normalize_chapter_ref(m.group(1))

    def _legal_unit_key(self, item: dict[str, Any]) -> str:
        doc_key = self._normalize_key(
            str(item.get("doc_id", ""))
            or str(item.get("document_number", ""))
            or str(item.get("path", ""))
        )
        article = extract_numeric_ref(item.get("article", ""))
        clause = extract_numeric_ref(item.get("clause", ""))

        if doc_key and (article or clause):
            return f"{doc_key}|a:{article}|c:{clause}"

        text = self._normalize_key(str(item.get("text", "")))
        if not text:
            text = self._normalize_key(" ".join([
                str(item.get("title", "")),
                str(item.get("path", "")),
                str(item.get("article", "")),
                str(item.get("clause", "")),
            ]))
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

        if doc_key:
            return f"{doc_key}|h:{digest}"
        return f"h:{digest}"

    def _dedup_by_legal_unit(self, rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        seen: dict[str, str] = {}
        deduped: list[dict] = []
        filtered: list[dict] = []

        for row in rows:
            dedup_key = self._legal_unit_key(row)
            chunk_id = str(row.get("chunk_id", ""))
            if dedup_key in seen:
                filtered.append({
                    "chunk_id": chunk_id,
                    "reason": "duplicate_legal_unit",
                    "dedup_key": dedup_key,
                    "kept_chunk_id": seen[dedup_key],
                })
                continue

            seen[dedup_key] = chunk_id
            deduped.append(row)

        return deduped, filtered

    @staticmethod
    def _extract_query_phrases(tokens: list[str]) -> set[str]:
        phrases = set(tokens)
        for idx in range(len(tokens) - 1):
            phrases.add(f"{tokens[idx]} {tokens[idx + 1]}")
        return phrases

    @staticmethod
    def _rrf_rank_score(rank: int | None, *, k: int = 60) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return 1.0 / (float(k) + float(rank))

    @classmethod
    def score_concept_coverage(cls, item: dict[str, Any], rewrite: dict[str, Any]) -> float:
        tags = [
            str(v).strip() for v in (rewrite.get("concept_tags") or [])
            if str(v).strip()
        ]
        if not tags:
            return 0.0
        title_text = tokenize_for_query(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("path", "")),
                    str(item.get("article", "")),
                ]
            )
        )
        body_text = tokenize_for_query(str(item.get("text", ""))[:1200])
        score = 0.0
        for tag in tags:
            for phrase in _INTRA_DOC_CONCEPT_PHRASES.get(tag, ()):
                if phrase and phrase in title_text:
                    score += 1.0
                elif phrase and phrase in body_text:
                    score += 0.5
        return min(score, 2.0)

    @staticmethod
    def score_actor_action_alignment(item: dict[str, Any], rewrite: dict[str, Any]) -> float:
        actor_terms = [
            str(v).strip() for v in (rewrite.get("actor_terms") or [])
            if str(v).strip()
        ]
        action_terms = [
            str(v).strip() for v in (rewrite.get("action_terms") or [])
            if str(v).strip()
        ]
        object_terms = [
            str(v).strip() for v in (rewrite.get("object_terms") or [])
            if str(v).strip()
        ]
        hay = tokenize_for_query(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("path", "")),
                    str(item.get("article", "")),
                    str(item.get("text", ""))[:1200],
                ]
            )
        )
        actor_hit = any(term in hay for term in actor_terms)
        action_hit = any(term in hay for term in action_terms)
        object_hit = any(term in hay for term in object_terms)
        score = 0.0
        if actor_hit:
            score += 0.5
        if action_hit:
            score += 0.5
        if object_hit:
            score += 0.3
        if actor_hit and action_hit:
            score += 0.4
        return min(score, 1.5)

    @classmethod
    def score_title_anchor_alignment(cls, item: dict[str, Any], rewrite: dict[str, Any]) -> float:
        anchors = [
            str(v).strip() for v in (rewrite.get("legal_anchor_guess") or [])
            if str(v).strip()
        ]
        tags = [
            str(v).strip() for v in (rewrite.get("concept_tags") or [])
            if str(v).strip()
        ]
        title_text = tokenize_for_query(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("path", "")),
                    str(item.get("document_number", "")),
                ]
            )
        )
        score = 0.0
        for anchor in anchors:
            if anchor and anchor in title_text:
                score += 0.8
        for tag in tags:
            for phrase in _INTRA_DOC_CONCEPT_PHRASES.get(tag, ()):
                if phrase and phrase in title_text:
                    score += 0.7
        return min(score, 2.0)

    @staticmethod
    def score_doc_type_prior(item: dict[str, Any], rewrite: dict[str, Any]) -> float:
        doc_type_prior = [
            str(v).strip().lower() for v in (rewrite.get("doc_type_prior") or [])
            if str(v).strip()
        ]
        if not doc_type_prior:
            return 0.0
        doc_bucket = HybridRetriever._doc_type_bucket(item)
        try:
            idx = doc_type_prior.index(doc_bucket)
        except ValueError:
            return 0.0
        return max(0.0, 1.0 - idx * 0.2)

    @classmethod
    def _doc_role_of_item(cls, item: dict[str, Any]) -> str:
        bucket = cls._doc_type_bucket(item)
        title = tokenize_for_query(" ".join([str(item.get("title", "")), str(item.get("path", ""))]))
        if bucket in {"luat", "bo_luat"}:
            return "law_core"
        if bucket == "vbhn":
            return "consolidated"
        if bucket in {"nghi_dinh", "thong_tu"}:
            if re.search(r"\b(xu phat|vi pham hanh chinh|ky luat)\b", title):
                return "implementation_sanction"
            return "implementation"
        if bucket in {"quyet_dinh_ubnd", "van_ban_dia_phuong"}:
            return "local_admin"
        if bucket == "cong_van":
            return "guidance"
        return "unknown"

    @classmethod
    def _build_doc_role_prior(cls, rewrite: dict[str, Any]) -> list[str]:
        concept_tags = [
            str(v).strip().lower() for v in (rewrite.get("concept_tags") or [])
            if str(v).strip()
        ]
        topic_class = str(rewrite.get("topic_class", "")).strip().lower()
        role_prior: list[str] = []
        for tag in concept_tags:
            role_prior.extend(_CONCEPT_DOC_ROLE_PRIOR.get(tag, []))
        if not role_prior:
            role_prior.extend(_TOPIC_DOC_ROLE_PRIOR.get(topic_class, _TOPIC_DOC_ROLE_PRIOR["general_legal"]))
        return [r for r in dict.fromkeys(role_prior) if r]

    @classmethod
    def _score_doc_role_prior(
        cls,
        *,
        item: dict[str, Any],
        rewrite: dict[str, Any],
    ) -> tuple[float, float, str]:
        role_prior = cls._build_doc_role_prior(rewrite)
        role = cls._doc_role_of_item(item)
        if not role_prior:
            return 0.0, 0.0, role
        try:
            idx = role_prior.index(role)
            score = max(0.0, 1.0 - idx * 0.22)
            return score, 0.0, role
        except ValueError:
            penalty = 0.0
            if role in {"local_admin", "guidance"}:
                penalty = 0.7
            elif role == "unknown":
                penalty = 0.25
            else:
                penalty = 0.35
            return 0.0, penalty, role

    @classmethod
    def _family_is_garbage(cls, family_key: str) -> bool:
        key = tokenize_for_query(str(family_key or ""))
        if not key:
            return True
        for pattern in _GARBAGE_FAMILY_PATTERNS:
            if re.search(pattern, key):
                return True
        return False

    @classmethod
    def _family_has_legal_morphology(cls, family_key: str) -> bool:
        key = tokenize_for_query(str(family_key or ""))
        if not key:
            return False
        if cls._family_is_garbage(key):
            return False
        return bool(_FAMILY_LEGAL_MORPH_PATTERN.search(key))

    @classmethod
    def score_family_identity(
        cls,
        *,
        item: dict[str, Any],
        family_key: str,
    ) -> float:
        title = tokenize_for_query(
            " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("path", "")),
                    str(item.get("document_number", "")),
                    str(item.get("doc_type", "")),
                ]
            )
        )
        score = 0.0
        if cls._family_has_legal_morphology(family_key):
            score += 0.95
        elif re.search(r"\b(nghi\s+dinh|thong\s+tu|vbhn|van\s+ban\s+hop\s+nhat)\b", title):
            score += 0.65
        else:
            score -= 0.45
        if cls._is_local_admin_doc(item):
            score -= 0.75
        if cls._doc_type_bucket(item) in {"cong_van", "quyet_dinh_ubnd", "van_ban_dia_phuong"}:
            score -= 0.60
        if cls._family_is_garbage(family_key) or cls._family_is_garbage(title):
            score -= 1.20
        if len(title.split()) <= 3:
            score -= 0.15
        return max(-2.0, min(1.8, score))

    @staticmethod
    def _source_support_count(cand: CandidateDocScore) -> int:
        count = 0
        if cand.bm25_core_rank is not None:
            count += 1
        if cand.bm25_expanded_rank is not None:
            count += 1
        if cand.bm25_title_rank is not None:
            count += 1
        if cand.vector_rank is not None:
            count += 1
        return count

    @classmethod
    def _can_use_vague_concept_recovery(
        cls,
        *,
        lexical_core: str,
        concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        title_anchor_query: str,
        weak_query_abort: bool,
    ) -> bool:
        if weak_query_abort:
            return False
        lexical_tokens = [
            tok
            for tok in tokenize_for_query(lexical_core).split()
            if tok and len(tok) > 1 and tok not in {"la", "gi", "ve", "can", "chuc", "quan"}
        ]
        useful_token_count = len(dict.fromkeys(lexical_tokens))
        if useful_token_count >= 2:
            return True
        if concept_tags and (
            (actor_terms and action_terms)
            or (action_terms and object_terms)
            or (actor_terms and object_terms)
        ):
            return True
        title_tokens = tokenize_for_query(title_anchor_query).split()
        if len(title_tokens) >= 4 and not any(tok in {"la", "gi", "ve"} for tok in title_tokens):
            return True
        return False

    @staticmethod
    def score_genericity_penalty(
        item: dict[str, Any],
        rewrite: dict[str, Any],
        *,
        title_anchor_score: float,
        concept_coverage_score: float,
        actor_action_score: float,
    ) -> float:
        _ = rewrite
        title = tokenize_for_query(str(item.get("title", "")))
        if not title:
            return 0.0
        has_generic_marker = any(marker in title for marker in _GENERIC_DOC_TITLE_MARKERS)
        if not has_generic_marker:
            return 0.0
        penalty = 0.0
        if title_anchor_score < 0.6:
            penalty += 0.5
        if concept_coverage_score < 0.7:
            penalty += 0.4
        if actor_action_score < 0.6:
            penalty += 0.3
        return min(1.0, penalty)

    @classmethod
    def _should_use_vector_for_concept_doc_recall(
        cls,
        *,
        concept_confidence: float,
        lexical_quality_flags: list[str],
        topic_class: str,
        bm25_title_hits: list[dict[str, Any]],
        bm25_core_hits: list[dict[str, Any]],
        bm25_expanded_hits: list[dict[str, Any]],
        candidate_family_count: int,
        family_score_margin: float,
        weak_query_abort: bool,
    ) -> bool:
        weak_lexical = any(
            flag in {"weak_structure", "low_token_variety", "too_short", "missing_actor", "missing_action"}
            for flag in (lexical_quality_flags or [])
        )
        title_thin = len(bm25_title_hits) < 3
        core_thin = len(bm25_core_hits) < 4
        expanded_thin = len(bm25_expanded_hits) < 6
        broad_topic = topic_class in {"general_legal", "civil_service", "labor", "administrative_sanction"}
        if candidate_family_count == 0:
            return True
        if family_score_margin < float(getattr(cfg, "HYBRID_DEFAULT_VECTOR_FAMILY_MARGIN_THRESHOLD", 0.10)):
            return True
        if weak_query_abort and (title_thin or core_thin):
            return True
        if concept_confidence < 0.45 and not weak_lexical:
            return False
        if weak_lexical or title_thin or core_thin or expanded_thin:
            return True
        if broad_topic and len(bm25_title_hits) < 5:
            return True
        return False

    @classmethod
    def _aggregate_concept_doc_candidates(
        cls,
        *,
        rewrite: dict[str, Any],
        bm25_core_hits: list[dict[str, Any]],
        bm25_expanded_hits: list[dict[str, Any]],
        bm25_title_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        use_family_identity: bool = True,
    ) -> list[CandidateDocScore]:
        registry: dict[str, dict[str, Any]] = {}

        def _key_of(row: dict[str, Any]) -> str:
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                return doc_id
            chunk_id = str(row.get("chunk_id", "")).strip()
            if chunk_id:
                return f"chunk::{chunk_id}"
            title = str(row.get("title", "")).strip()
            if title:
                return f"title::{title[:120]}"
            return ""

        def _merge_pass(hits: list[dict[str, Any]], pass_name: str) -> None:
            for idx, row in enumerate(hits, start=1):
                key = _key_of(row)
                if not key:
                    continue
                state = registry.setdefault(
                    key,
                    {
                        "doc_id": str(row.get("doc_id", "")).strip(),
                        "family_key": str(row.get("law_family_key", "")).strip()
                        or cls._canonical_law_family_key(str(row.get("title", "")))
                        or str(row.get("doc_id", "")).strip(),
                        "sample_row": dict(row),
                        "best_chunk_id": str(row.get("chunk_id", "")).strip() or None,
                        "bm25_core_rank": None,
                        "bm25_expanded_rank": None,
                        "bm25_title_rank": None,
                        "vector_rank": None,
                    },
                )
                rank_field = f"{pass_name}_rank"
                prev_rank = state.get(rank_field)
                if prev_rank is None or idx < int(prev_rank):
                    state[rank_field] = int(idx)
                    state["sample_row"] = dict(row)
                    state["best_chunk_id"] = str(row.get("chunk_id", "")).strip() or state.get("best_chunk_id")

        _merge_pass(bm25_core_hits or [], "bm25_core")
        _merge_pass(bm25_expanded_hits or [], "bm25_expanded")
        _merge_pass(bm25_title_hits or [], "bm25_title")
        _merge_pass(vector_hits or [], "vector")

        out: list[CandidateDocScore] = []
        for state in registry.values():
            sample = dict(state.get("sample_row") or {})
            bm25_core_rank = state.get("bm25_core_rank")
            bm25_expanded_rank = state.get("bm25_expanded_rank")
            bm25_title_rank = state.get("bm25_title_rank")
            vector_rank = state.get("vector_rank")

            concept_coverage_score = cls.score_concept_coverage(sample, rewrite)
            actor_action_score = cls.score_actor_action_alignment(sample, rewrite)
            title_anchor_score = cls.score_title_anchor_alignment(sample, rewrite)
            doc_type_prior_score = cls.score_doc_type_prior(sample, rewrite)
            doc_role_prior_score, wrong_doc_role_penalty, doc_role = cls._score_doc_role_prior(
                item=sample,
                rewrite=rewrite,
            )
            family_key = str(state.get("family_key", "")).strip()
            family_identity_score = cls.score_family_identity(
                item=sample,
                family_key=family_key,
            ) if use_family_identity else 0.0
            genericity_penalty = cls.score_genericity_penalty(
                sample,
                rewrite,
                title_anchor_score=title_anchor_score,
                concept_coverage_score=concept_coverage_score,
                actor_action_score=actor_action_score,
            )

            bm25_doc_score = (
                1.00 * cls._rrf_rank_score(bm25_core_rank)
                + 0.85 * cls._rrf_rank_score(bm25_expanded_rank)
            )
            title_support_score = 1.10 * cls._rrf_rank_score(bm25_title_rank)
            vector_doc_score = 0.55 * cls._rrf_rank_score(vector_rank)
            concept_alignment_score = (
                0.62 * concept_coverage_score
                + 0.38 * actor_action_score
            )
            support_source_count = int(
                int(bm25_core_rank is not None)
                + int(bm25_expanded_rank is not None)
                + int(bm25_title_rank is not None)
                + int(vector_rank is not None)
            )
            final_doc_score = (
                bm25_doc_score
                + title_support_score
                + vector_doc_score
                + 0.80 * concept_alignment_score
                + 0.18 * doc_type_prior_score
                + 0.20 * doc_role_prior_score
                + 0.28 * title_anchor_score
                + (0.55 * family_identity_score if use_family_identity else 0.0)
                - 0.55 * wrong_doc_role_penalty
                - 0.50 * genericity_penalty
            )
            out.append(
                CandidateDocScore(
                    doc_id=str(state.get("doc_id", "")).strip(),
                    family_key=str(state.get("family_key", "")).strip(),
                    doc_role=str(doc_role),
                    best_chunk_id=state.get("best_chunk_id"),
                    bm25_core_rank=int(bm25_core_rank) if bm25_core_rank else None,
                    bm25_expanded_rank=int(bm25_expanded_rank) if bm25_expanded_rank else None,
                    bm25_title_rank=int(bm25_title_rank) if bm25_title_rank else None,
                    vector_rank=int(vector_rank) if vector_rank else None,
                    bm25_doc_score=float(bm25_doc_score),
                    vector_doc_score=float(vector_doc_score),
                    title_anchor_score=float(title_anchor_score),
                    concept_coverage_score=float(concept_coverage_score),
                    actor_action_score=float(actor_action_score),
                    doc_type_prior_score=float(doc_type_prior_score),
                    doc_role_prior_score=float(doc_role_prior_score),
                    wrong_doc_role_penalty=float(wrong_doc_role_penalty),
                    genericity_penalty=float(genericity_penalty),
                    family_identity_score=float(family_identity_score),
                    concept_alignment_score=float(concept_alignment_score),
                    bm25_support_score=float(bm25_doc_score),
                    title_support_score=float(title_support_score),
                    vector_support_score=float(vector_doc_score),
                    family_support_source_count=int(support_source_count),
                    final_doc_score=float(final_doc_score),
                )
            )
        out.sort(key=lambda it: (it.final_doc_score, it.title_anchor_score), reverse=True)
        return out

    @staticmethod
    def _select_concept_candidate_docs(
        candidates: list[CandidateDocScore],
        *,
        max_families: int = _CONCEPT_MAX_CANDIDATE_FAMILIES,
        max_docs: int = _CONCEPT_MAX_CANDIDATE_DOCS,
    ) -> tuple[list[str], list[str]]:
        if not candidates:
            return [], []

        family_scores: dict[str, float] = {}
        family_best_doc: dict[str, CandidateDocScore] = {}
        for cand in candidates:
            family = str(cand.family_key or cand.doc_id)
            if not family:
                continue
            family_scores[family] = family_scores.get(family, 0.0) + float(cand.final_doc_score)
            best = family_best_doc.get(family)
            if best is None or cand.final_doc_score > best.final_doc_score:
                family_best_doc[family] = cand

        ranked_families = sorted(
            family_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if not ranked_families:
            return [], []

        keep_families = max(1, int(max_families))
        if len(ranked_families) >= 2:
            top1 = float(ranked_families[0][1])
            top2 = float(ranked_families[1][1])
            if top2 > 0 and top1 >= top2 * 1.25:
                keep_families = min(2, keep_families)
            else:
                keep_families = min(keep_families, 3)
        selected_family_keys = [fam for fam, _ in ranked_families[:keep_families]]
        selected_doc_ids: list[str] = []
        for cand in candidates:
            if cand.family_key not in selected_family_keys:
                continue
            if cand.doc_id and cand.doc_id not in selected_doc_ids:
                selected_doc_ids.append(cand.doc_id)
            if len(selected_doc_ids) >= max_docs:
                break
        return selected_doc_ids, selected_family_keys

    @classmethod
    def _resolve_family_candidates(
        cls,
        *,
        candidates: list[CandidateDocScore],
        max_families: int,
        max_docs: int,
        enable_garbage_rejection: bool = True,
    ) -> dict[str, Any]:
        if not candidates:
            return {
                "ranked_families": [],
                "selected_doc_ids": [],
                "selected_family_keys": [],
                "family_doc_count": {},
                "family_top_role": {},
                "family_score_top1": 0.0,
                "family_score_top2": 0.0,
                "family_score_margin": 0.0,
                "family_identity_score_top1": 0.0,
                "family_identity_score_top2": 0.0,
                "family_identity_margin": 0.0,
                "garbage_family_rejected": False,
                "garbage_family_reject_reason": "",
                "trusted_family": False,
                "trusted_family_confidence": 0.0,
                "source_agreement_top1": 0,
            }

        family_stats: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            family = str(cand.family_key or cand.doc_id).strip()
            if not family:
                continue
            stat = family_stats.setdefault(
                family,
                {
                    "doc_count": 0,
                    "identity_sum": 0.0,
                    "concept_sum": 0.0,
                    "bm25_support_sum": 0.0,
                    "title_support_sum": 0.0,
                    "vector_support_sum": 0.0,
                    "prior_sum": 0.0,
                    "wrong_role_penalty_sum": 0.0,
                    "genericity_penalty_sum": 0.0,
                    "top_role": "",
                    "top_doc_score": -1e9,
                    "source_flags": set(),
                },
            )
            stat["doc_count"] = int(stat["doc_count"]) + 1
            stat["identity_sum"] = float(stat["identity_sum"]) + float(cand.family_identity_score)
            stat["concept_sum"] = float(stat["concept_sum"]) + float(cand.concept_alignment_score)
            stat["bm25_support_sum"] = float(stat["bm25_support_sum"]) + float(cand.bm25_support_score)
            stat["title_support_sum"] = float(stat["title_support_sum"]) + float(cand.title_support_score)
            stat["vector_support_sum"] = float(stat["vector_support_sum"]) + float(cand.vector_support_score)
            stat["prior_sum"] = float(stat["prior_sum"]) + float(cand.doc_role_prior_score)
            stat["wrong_role_penalty_sum"] = float(stat["wrong_role_penalty_sum"]) + float(cand.wrong_doc_role_penalty)
            stat["genericity_penalty_sum"] = float(stat["genericity_penalty_sum"]) + float(cand.genericity_penalty)
            if cand.bm25_core_rank is not None:
                stat["source_flags"].add("concept_seed")
            if cand.bm25_expanded_rank is not None:
                stat["source_flags"].add("bm25")
            if cand.bm25_title_rank is not None:
                stat["source_flags"].add("title")
            if cand.vector_rank is not None:
                stat["source_flags"].add("vector")
            if cand.final_doc_score > float(stat["top_doc_score"]):
                stat["top_doc_score"] = float(cand.final_doc_score)
                stat["top_role"] = str(cand.doc_role or "")

        family_doc_count: dict[str, int] = {
            family: int(stat.get("doc_count", 0))
            for family, stat in family_stats.items()
        }
        family_top_role: dict[str, str] = {
            family: str(stat.get("top_role", ""))
            for family, stat in family_stats.items()
        }

        pre_rank_rows: list[dict[str, Any]] = []
        for family, stat in family_stats.items():
            doc_count = max(1, int(stat.get("doc_count", 1)))
            identity = float(stat.get("identity_sum", 0.0)) / doc_count
            concept = float(stat.get("concept_sum", 0.0)) / doc_count
            bm25_support = float(stat.get("bm25_support_sum", 0.0)) / doc_count
            title_support = float(stat.get("title_support_sum", 0.0)) / doc_count
            vector_support = float(stat.get("vector_support_sum", 0.0)) / doc_count
            prior = float(stat.get("prior_sum", 0.0)) / doc_count
            wrong_role_penalty = float(stat.get("wrong_role_penalty_sum", 0.0)) / doc_count
            genericity_penalty = float(stat.get("genericity_penalty_sum", 0.0)) / doc_count
            source_agreement = len(set(stat.get("source_flags", set())))
            base_score = (
                0.95 * identity
                + 0.62 * concept
                + 0.52 * bm25_support
                + 0.34 * title_support
                + 0.22 * vector_support
                - 0.35 * genericity_penalty
                - 0.30 * wrong_role_penalty
            )
            pre_rank_rows.append(
                {
                    "family": family,
                    "base_score": float(base_score),
                    "identity": float(identity),
                    "concept": float(concept),
                    "bm25_support": float(bm25_support),
                    "title_support": float(title_support),
                    "vector_support": float(vector_support),
                    "prior": float(prior),
                    "source_agreement": int(source_agreement),
                    "top_role": str(stat.get("top_role", "")),
                }
            )

        pre_rank_rows.sort(key=lambda row: row["base_score"], reverse=True)
        pre_top1 = float(pre_rank_rows[0]["base_score"]) if pre_rank_rows else 0.0
        pre_top2 = float(pre_rank_rows[1]["base_score"]) if len(pre_rank_rows) >= 2 else 0.0
        pre_margin = max(0.0, pre_top1 - pre_top2)
        low_confidence = pre_top1 < 0.85 or pre_margin < 0.10
        prior_weight = 0.30 if low_confidence else 1.0

        ranked_rows: list[dict[str, Any]] = []
        for row in pre_rank_rows:
            row["score"] = float(row["base_score"] + prior_weight * 0.40 * float(row["prior"]))
            ranked_rows.append(row)
        ranked_rows.sort(key=lambda row: row["score"], reverse=True)
        ranked_families = sorted(
            ((str(row["family"]), float(row["score"])) for row in ranked_rows),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )

        keep_families = max(1, int(max_families))
        if len(ranked_families) >= 2:
            top1 = float(ranked_families[0][1])
            top2 = float(ranked_families[1][1])
            if top2 > 0 and top1 >= top2 * 1.30:
                keep_families = min(2, keep_families)
            else:
                keep_families = min(keep_families, 3)
        selected_family_keys = [family for family, _ in ranked_families[:keep_families]]

        top_row = ranked_rows[0] if ranked_rows else {}
        top1_family = str(top_row.get("family", ""))
        top1_score = float(top_row.get("score", 0.0))
        top2_score = float(ranked_rows[1].get("score", 0.0)) if len(ranked_rows) >= 2 else 0.0
        top_margin = max(0.0, top1_score - top2_score)
        top1_source_agreement = int(top_row.get("source_agreement", 0))
        top1_role = str(top_row.get("top_role", ""))
        top1_legal_morph = cls._family_has_legal_morphology(top1_family)
        garbage_rejected = False
        garbage_reject_reason = ""
        if enable_garbage_rejection and top1_family:
            reject_reasons: list[str] = []
            margin_eps = float(getattr(cfg, "HYBRID_DEFAULT_GARBAGE_FAMILY_MARGIN_EPSILON", 0.08))
            if cls._family_is_garbage(top1_family):
                reject_reasons.append("garbage_family_key")
            if not top1_legal_morph and top_margin < margin_eps:
                reject_reasons.append("non_legal_title_morph")
            if top1_role == "unknown" and top_margin < margin_eps:
                reject_reasons.append("unknown_doc_role_low_margin")
            if top_margin < margin_eps:
                reject_reasons.append("low_margin")
            if top1_source_agreement < 2:
                reject_reasons.append("low_source_agreement")
            if reject_reasons:
                garbage_rejected = True
                garbage_reject_reason = "+".join(dict.fromkeys(reject_reasons))
                selected_family_keys = [
                    family
                    for family, _ in ranked_families
                    if family != top1_family and cls._family_has_legal_morphology(family)
                ][: max(1, min(2, keep_families))]

        selected_doc_ids: list[str] = []
        for cand in candidates:
            if cand.family_key not in selected_family_keys:
                continue
            if cand.doc_id and cand.doc_id not in selected_doc_ids:
                selected_doc_ids.append(cand.doc_id)
            if len(selected_doc_ids) >= max_docs:
                break

        trusted_family = bool(
            selected_family_keys
            and not garbage_rejected
            and top1_legal_morph
            and top1_source_agreement >= 2
            and (top_margin >= 0.10 or top1_score >= 1.0)
            and not (top1_role == "unknown" and top_margin < 0.14)
        )
        trusted_family_conf = float(top1_score + min(0.40, top_margin) + 0.10 * max(0, top1_source_agreement - 1))

        top1_family_score = float(ranked_families[0][1]) if ranked_families else 0.0
        top2_family_score = float(ranked_families[1][1]) if len(ranked_families) >= 2 else 0.0
        top1_identity = float(ranked_rows[0].get("identity", 0.0)) if ranked_rows else 0.0
        top2_identity = float(ranked_rows[1].get("identity", 0.0)) if len(ranked_rows) >= 2 else 0.0
        return {
            "ranked_families": ranked_families,
            "selected_doc_ids": selected_doc_ids,
            "selected_family_keys": selected_family_keys,
            "family_doc_count": family_doc_count,
            "family_top_role": family_top_role,
            "family_score_top1": top1_family_score,
            "family_score_top2": top2_family_score,
            "family_score_margin": max(0.0, top1_family_score - top2_family_score),
            "family_identity_score_top1": top1_identity,
            "family_identity_score_top2": top2_identity,
            "family_identity_margin": max(0.0, top1_identity - top2_identity),
            "garbage_family_rejected": bool(garbage_rejected),
            "garbage_family_reject_reason": str(garbage_reject_reason),
            "trusted_family": bool(trusted_family),
            "trusted_family_confidence": float(trusted_family_conf),
            "source_agreement_top1": int(top1_source_agreement),
            "doc_role_prior_weight": float(prior_weight),
        }

    @classmethod
    def _implementation_bridge_expand(
        cls,
        *,
        selected_doc_ids: list[str],
        selected_family_keys: list[str],
        candidates: list[CandidateDocScore],
        max_docs: int,
        trusted_family: bool,
        family_score_margin: float,
    ) -> tuple[list[str], int]:
        if not trusted_family or family_score_margin < float(
            getattr(cfg, "HYBRID_DEFAULT_IMPLEMENTATION_BRIDGE_MIN_MARGIN", 0.10)
        ):
            return list(selected_doc_ids or []), 0
        selected_doc_set = {
            str(v).strip()
            for v in (selected_doc_ids or [])
            if str(v).strip()
        }
        promoted = 0
        for cand in candidates:
            if cand.family_key not in selected_family_keys:
                continue
            if cand.doc_id in selected_doc_set:
                continue
            if cand.doc_role not in _IMPLEMENTATION_ROLES:
                continue
            selected_doc_set.add(cand.doc_id)
            promoted += 1
            if len(selected_doc_set) >= max_docs:
                break
        return list(selected_doc_set), promoted

    @classmethod
    def _retrieve_chunks_within_shortlist(
        cls,
        *,
        rows: list[dict[str, Any]],
        selected_doc_ids: list[str],
        selected_family_keys: list[str],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not rows:
            return [], 0
        doc_set = {
            str(v).strip()
            for v in (selected_doc_ids or [])
            if str(v).strip()
        }
        fam_set = {
            str(v).strip()
            for v in (selected_family_keys or [])
            if str(v).strip()
        }
        filtered: list[dict[str, Any]] = []
        for row in rows:
            doc_id = str(row.get("doc_id", "")).strip()
            family = str(row.get("law_family_key", "")).strip() or cls._canonical_law_family_key(str(row.get("title", "")))
            if doc_set and doc_id and doc_id in doc_set:
                filtered.append(row)
                continue
            if fam_set and family and family in fam_set:
                filtered.append(row)
                continue
        if not filtered:
            filtered = list(rows)
        return filtered[: max(1, int(limit))], len(filtered)

    @staticmethod
    def _should_run_focus_rerank(
        *,
        family_score_top1: float,
        family_score_margin: float,
        candidate_family_count: int,
        candidate_chunks: int,
        vagueness_level: str,
        trusted_family: bool,
    ) -> tuple[bool, float]:
        confidence = float(family_score_top1 + min(0.4, family_score_margin))
        if not trusted_family:
            return False, confidence
        if vagueness_level == "hard":
            return False, confidence
        if candidate_chunks < 4:
            return False, confidence
        if candidate_family_count <= 1:
            return True, confidence
        if family_score_top1 >= 0.55:
            return True, confidence
        if family_score_margin >= 0.15:
            return True, confidence
        return False, confidence

    @classmethod
    def _intra_doc_focus_rerank(
        cls,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        query_subclass: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        focus_terms: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates:
            return candidates, {
                "applied": False,
                "reason": "no_candidates",
                "target_doc_ids": [],
                "target_family_keys": [],
                "rank_first_doc": 0,
                "rank_first_family": 0,
                "candidates": 0,
                "promoted": 0,
            }

        top_window = list(candidates[:12])
        doc_counts: dict[str, int] = {}
        doc_first_rank: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        family_first_rank: dict[str, int] = {}

        for idx, row in enumerate(top_window, start=1):
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                if doc_id not in doc_first_rank:
                    doc_first_rank[doc_id] = idx
            family_key = str(row.get("law_family_key", "")).strip()
            if not family_key:
                family_key = cls._canonical_law_family_key(str(row.get("title", "")))
            if family_key:
                family_counts[family_key] = family_counts.get(family_key, 0) + 1
                if family_key not in family_first_rank:
                    family_first_rank[family_key] = idx

        target_doc_ids = [
            doc_id
            for doc_id, count in doc_counts.items()
            if count >= 2 and doc_first_rank.get(doc_id, 999) <= 3
        ][:2]
        target_family_keys = [
            family_key
            for family_key, count in family_counts.items()
            if count >= 2 and family_first_rank.get(family_key, 999) <= 3
        ][:2]
        rank_first_doc = min((doc_first_rank.get(doc_id, 999) for doc_id in target_doc_ids), default=0)
        rank_first_family = min((family_first_rank.get(k, 999) for k in target_family_keys), default=0)

        if not target_doc_ids and not target_family_keys:
            return candidates, {
                "applied": False,
                "reason": "no_target_doc_or_family",
                "target_doc_ids": [],
                "target_family_keys": [],
                "rank_first_doc": 0,
                "rank_first_family": 0,
                "candidates": len(candidates),
                "promoted": 0,
            }

        concept_phrases = {
            phrase
            for tag in (legal_concept_tags or [])
            for phrase in _INTRA_DOC_CONCEPT_PHRASES.get(str(tag), ())
            if phrase
        }
        actor_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(actor_terms or [])).split()
            if tok
        }
        action_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(action_terms or [])).split()
            if tok
        }
        object_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(object_terms or [])).split()
            if tok
        }
        focus_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(focus_terms or [])).split()
            if tok
        }
        query_tokens = set(tokenize_for_query(query).split())

        promoted = 0
        reranked: list[dict[str, Any]] = []
        for row in candidates:
            updated = dict(row)
            base_score = float(updated.get("rerank_score", updated.get("rrf_score", 0.0)))
            doc_id = str(updated.get("doc_id", "")).strip()
            family_key = str(updated.get("law_family_key", "")).strip()
            if not family_key:
                family_key = cls._canonical_law_family_key(str(updated.get("title", "")))
            in_target_doc = bool(doc_id and doc_id in target_doc_ids)
            in_target_family = bool(family_key and family_key in target_family_keys)

            focus_bonus = 0.0
            if in_target_doc or in_target_family:
                haystack = " ".join(
                    [
                        str(updated.get("title", "")),
                        str(updated.get("path", "")),
                        str(updated.get("article", "")),
                        str(updated.get("clause", "")),
                        str(updated.get("text", ""))[:800],
                    ]
                )
                hay_tokenized = tokenize_for_query(haystack)
                hay_tokens = set(hay_tokenized.split())
                heading_tokenized = tokenize_for_query(
                    " ".join(
                        [
                            str(updated.get("title", "")),
                            str(updated.get("path", "")),
                            str(updated.get("article", "")),
                        ]
                    )
                )
                heading_hits = sum(1 for phrase in concept_phrases if phrase and phrase in heading_tokenized)
                concept_hits = sum(1 for phrase in concept_phrases if phrase and phrase in hay_tokenized)
                actor_hits = len(actor_tokens & hay_tokens)
                action_hits = len(action_tokens & hay_tokens)
                object_hits = len(object_tokens & hay_tokens)
                focus_hits = len(focus_tokens & hay_tokens)
                query_hits = len(query_tokens & hay_tokens)

                if in_target_doc:
                    focus_bonus += 0.006
                elif in_target_family:
                    focus_bonus += 0.003
                focus_bonus += min(0.010, heading_hits * 0.0025)
                focus_bonus += min(0.010, concept_hits * 0.0020)
                if actor_hits > 0 and action_hits > 0:
                    focus_bonus += min(0.010, (actor_hits + action_hits) * 0.0018)
                elif action_hits > 0:
                    focus_bonus += min(0.006, action_hits * 0.0015)
                if object_hits > 0:
                    focus_bonus += min(0.005, object_hits * 0.0015)
                focus_bonus += min(0.004, focus_hits * 0.0010)
                focus_bonus += min(0.003, query_hits * 0.0007)

                if query_subclass == "concept_generic":
                    article_num = extract_numeric_ref(str(updated.get("article", "")))
                    if not article_num:
                        m = re.search(r"(?:dieu)\s+(\d+)", str(updated.get("path", "")), flags=re.IGNORECASE)
                        if m:
                            article_num = m.group(1)
                    try:
                        article_idx = int(article_num or "0")
                    except Exception:
                        article_idx = 0
                    if 1 <= article_idx <= 3:
                        focus_bonus += 0.004
                    elif article_idx > 0:
                        focus_bonus -= 0.001
                    if concept_hits == 0 and action_hits == 0:
                        focus_bonus -= 0.001

            updated["rerank_score"] = base_score + focus_bonus
            updated["_intra_doc_focus_bonus"] = round(float(focus_bonus), 6)
            updated["_focus_heading_match"] = float(min(1.0, heading_hits * 0.5)) if (in_target_doc or in_target_family) else 0.0
            updated["_focus_actor_action_match"] = float(1.0 if (actor_hits > 0 and action_hits > 0) else 0.0) if (in_target_doc or in_target_family) else 0.0
            updated["_focus_concept_match"] = float(min(1.0, concept_hits * 0.4)) if (in_target_doc or in_target_family) else 0.0
            if focus_bonus > 0.0005:
                promoted += 1
            reranked.append(updated)

        reranked.sort(
            key=lambda x: (float(x.get("rerank_score", 0.0)), -int(x.get("original_rank", 0))),
            reverse=True,
        )
        for new_rank, row in enumerate(reranked):
            row["rank"] = new_rank

        top1 = reranked[0] if reranked else {}

        return reranked, {
            "applied": bool(promoted > 0),
            "reason": "reranked" if promoted > 0 else "no_positive_bonus",
            "target_doc_ids": target_doc_ids,
            "target_family_keys": target_family_keys,
            "rank_first_doc": int(rank_first_doc),
            "rank_first_family": int(rank_first_family),
            "candidates": len(candidates),
            "promoted": int(promoted),
            "top1_heading_match": float(top1.get("_focus_heading_match", 0.0)),
            "top1_actor_action_match": float(top1.get("_focus_actor_action_match", 0.0)),
            "top1_concept_match": float(top1.get("_focus_concept_match", 0.0)),
        }

    def _prioritize_law_anchored_hits(
        self,
        candidates: list[dict[str, Any]],
        *,
        law_name: str,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return candidates

        law_phrase = " ".join(tokenize_for_query(law_name).split())
        law_tokens = set(law_phrase.split())
        if not law_tokens:
            return candidates

        ranked_rows: list[dict[str, Any]] = []
        anchored_count = 0

        for row in candidates:
            hay = " ".join([
                str(row.get("title", "")),
                str(row.get("path", "")),
                str(row.get("document_number", "")),
            ])
            hay_tokenized = " ".join(tokenize_for_query(hay).split())
            hay_tokens = set(hay_tokenized.split())

            phrase_match = bool(law_phrase and law_phrase in hay_tokenized)
            token_overlap = len(law_tokens & hay_tokens)
            token_threshold = min(3, max(2, len(law_tokens)))

            if phrase_match:
                law_priority = 2
            elif token_overlap >= token_threshold:
                law_priority = 1
            else:
                law_priority = 0

            if law_priority > 0:
                anchored_count += 1

            updated = dict(row)
            updated["_law_priority"] = law_priority
            ranked_rows.append(updated)

        if anchored_count < 1:
            return candidates

        ranked_rows.sort(
            key=lambda row: (
                int(row.get("_law_priority", 0)),
                float(row.get("rerank_score", 0.0)),
                -int(row.get("original_rank", 0)),
            ),
            reverse=True,
        )
        for idx, row in enumerate(ranked_rows):
            row["rank"] = idx
            row.pop("_law_priority", None)
        return ranked_rows

    def _heuristic_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        legal_refs: dict[str, Any] | None = None,
        route: str = "hybrid_default",
        allowed_doc_ids: set[str] | None = None,
        document_lookup_confidence: float = 0.0,
        query_rewrite_doc_type_prior: list[str] | None = None,
        query_rewrite_exclude_doc_type_hint: list[str] | None = None,
        query_rewrite_subclass: str = "unknown",
        query_rewrite_legal_concept_tags: list[str] | None = None,
        query_rewrite_actor_terms: list[str] | None = None,
        query_rewrite_action_terms: list[str] | None = None,
    ) -> list[dict]:
        """
        Secondary rerank layer.
        Keep it lightweight and generic; routing controls the main behavior.
        """
        if not candidates:
            return candidates

        refs = legal_refs or {}
        allowed_doc_ids_norm = {
            str(doc_id).strip()
            for doc_id in (allowed_doc_ids or set())
            if str(doc_id).strip()
        }
        target_doc_number = normalize_doc_number(str(refs.get("document_number", "")))
        target_doc_short = str(refs.get("document_short", "")).strip()
        if not target_doc_short and target_doc_number:
            target_doc_short = extract_doc_short(target_doc_number)
        target_article = extract_numeric_ref(str(refs.get("article_number", "")))
        target_clause = extract_numeric_ref(str(refs.get("clause_number", "")))
        target_chapter = normalize_chapter_ref(str(refs.get("chapter", "")))
        target_law_tokens = set(tokenize_for_query(str(refs.get("law_name", ""))).split())
        target_law_phrase = " ".join(tokenize_for_query(str(refs.get("law_name", ""))).split())
        target_law_family = self._canonical_law_family_key(str(refs.get("law_name", "")))
        prior_doc_type_set = {
            str(v).strip().lower()
            for v in (query_rewrite_doc_type_prior or [])
            if str(v).strip()
        }
        exclude_doc_type_set = {
            str(v).strip().lower()
            for v in (query_rewrite_exclude_doc_type_hint or [])
            if str(v).strip()
        }
        query_subclass = str(query_rewrite_subclass or "unknown")
        concept_tags = [
            str(v).strip().lower()
            for v in (query_rewrite_legal_concept_tags or [])
            if str(v).strip()
        ]
        concept_hint_phrases = {
            phrase
            for tag in concept_tags
            for phrase in _INTRA_DOC_CONCEPT_PHRASES.get(tag, ())
        }
        actor_hint_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(query_rewrite_actor_terms or [])).split()
            if tok
        }
        action_hint_tokens = {
            tok
            for tok in tokenize_for_query(" ".join(query_rewrite_action_terms or [])).split()
            if tok
        }

        query_tokens = tokenize_for_query(query).split()
        query_token_set = set(query_tokens)
        query_phrases = {p for p in self._extract_query_phrases(query_tokens) if len(p) >= 6}
        meta_law_query = bool(
            re.search(
                r"\b(ban hanh|van ban quy pham|hieu luc|nghi dinh|thong tu)\b",
                " ".join(query_tokens),
            )
        )
        intent_token_set = {
            token
            for token in query_token_set
            if token
            and token not in target_law_tokens
            and token not in {"luật", "luat", "theo", "văn", "van", "bản", "ban", "điều", "dieu", "khoản", "khoan", "chương", "chuong"}
        }

        reranked = []
        for original_rank, item in enumerate(candidates):
            text_preview = str(item.get("text", ""))[:800]
            haystack = " ".join([
                str(item.get("title", "")),
                str(item.get("path", "")),
                str(item.get("article", "")),
                str(item.get("clause", "")),
                str(item.get("document_number", "")),
                text_preview,
            ]).lower()

            hay_tokens_list = tokenize_for_query(haystack).split()
            hay_tokens = set(hay_tokens_list)
            haystack_tokenized = " ".join(hay_tokens_list)
            token_hits = len(query_token_set & hay_tokens)
            phrase_hits = sum(1 for phrase in query_phrases if phrase in haystack)
            law_overlap = len(target_law_tokens & hay_tokens)
            intent_overlap = len(intent_token_set & hay_tokens)
            actor_overlap = len(actor_hint_tokens & hay_tokens)
            action_overlap = len(action_hint_tokens & hay_tokens)
            concept_phrase_hits = sum(1 for phrase in concept_hint_phrases if phrase and phrase in haystack_tokenized)

            base_score = float(item.get("rrf_score", 0.0))
            if base_score <= 0:
                base_score = float(item.get("score", 0.0)) * 0.001

            item_doc_id = str(item.get("doc_id", "")).strip()
            item_doc_number = normalize_doc_number(str(item.get("document_number", "")))
            item_doc_short = extract_doc_short(item_doc_number)
            item_article = extract_numeric_ref(str(item.get("article", "")))
            item_clause = extract_numeric_ref(str(item.get("clause", "")))
            item_chapter = self._extract_chapter_ref(str(item.get("path", "")))
            item_law_family = self._canonical_law_family_key(str(item.get("title", "")))
            law_family_match = bool(
                target_law_family
                and item_law_family
                and (
                    target_law_family == item_law_family
                    or target_law_family in item_law_family
                    or item_law_family in target_law_family
                )
            )
            local_admin_doc = self._is_local_admin_doc(item)
            meta_drafting_doc = self._is_meta_legal_drafting_doc(item)
            item_doc_type_bucket = self._doc_type_bucket(item)

            doc_match = False
            if target_doc_number:
                doc_match = item_doc_number == target_doc_number
            elif target_doc_short:
                doc_match = item_doc_short == target_doc_short

            bonus = 0.0
            if target_doc_number or target_doc_short:
                if doc_match:
                    bonus += 0.004
                elif item_doc_number:
                    bonus -= 0.0015

            if target_article:
                if item_article == target_article:
                    bonus += 0.006
                elif doc_match and item_article:
                    bonus -= 0.002

            if target_clause:
                if item_clause == target_clause:
                    bonus += 0.004
                elif doc_match and item_clause:
                    bonus -= 0.0015

            if target_chapter:
                if item_chapter == target_chapter:
                    bonus += 0.003
                elif doc_match and item_chapter:
                    bonus -= 0.001

            if route == "doc_scoped_hybrid_exact_doc":
                if allowed_doc_ids_norm:
                    if item_doc_id in allowed_doc_ids_norm:
                        bonus += 0.010
                    else:
                        bonus -= 0.020
                if intent_token_set:
                    if intent_overlap > 0:
                        bonus += min(0.012, intent_overlap * 0.0030)
                    else:
                        bonus -= 0.004

            if route == "doc_scoped_hybrid_candidate_docs":
                if allowed_doc_ids_norm:
                    if item_doc_id in allowed_doc_ids_norm:
                        bonus += 0.004 + min(0.004, float(document_lookup_confidence) * 0.0030)
                    else:
                        bonus -= 0.010
                if intent_token_set:
                    if intent_overlap > 0:
                        bonus += min(0.008, intent_overlap * 0.0020)
                    else:
                        bonus -= 0.002

            if route in {"law_anchored_hybrid", "law_anchored_hybrid_loose"}:
                if route == "law_anchored_hybrid_loose" and allowed_doc_ids_norm:
                    if item_doc_id in allowed_doc_ids_norm:
                        bonus += 0.006
                    else:
                        bonus -= 0.006
                if target_law_family and item_law_family:
                    if law_family_match:
                        bonus += 0.012
                    else:
                        bonus -= 0.016
                if target_law_phrase:
                    if target_law_phrase in haystack_tokenized:
                        bonus += 0.030
                    else:
                        bonus -= 0.015
                if target_law_tokens:
                    if law_overlap > 0:
                        bonus += min(0.008, law_overlap * 0.0020)
                    else:
                        bonus -= 0.020
                if intent_token_set:
                    if intent_overlap > 0:
                        bonus += min(0.010, intent_overlap * 0.0025)
                    else:
                        bonus -= 0.003
                if meta_drafting_doc and not meta_law_query:
                    bonus -= 0.020
                if local_admin_doc and (law_overlap <= 1 or intent_overlap == 0):
                    bonus -= 0.010

            if route == "hybrid_default":
                if intent_token_set and intent_overlap > 0:
                    bonus += min(0.006, intent_overlap * 0.0018)
                if phrase_hits > 0:
                    bonus += min(0.004, phrase_hits * 0.0013)
                if target_law_family and item_law_family and not law_family_match:
                    bonus -= 0.004
                if local_admin_doc and phrase_hits == 0 and intent_overlap == 0:
                    bonus -= 0.003
                if prior_doc_type_set and item_doc_type_bucket in prior_doc_type_set:
                    bonus += 0.004
                if exclude_doc_type_set and item_doc_type_bucket in exclude_doc_type_set:
                    bonus -= 0.012
                elif (
                    item_doc_type_bucket in {"quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"}
                    and intent_overlap <= 1
                ):
                    bonus -= 0.006
                if query_subclass == "concept_generic":
                    if item_doc_type_bucket in {"luat", "bo_luat", "nghi_dinh", "vbhn"}:
                        bonus += 0.006
                    elif item_doc_type_bucket in {"quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"}:
                        bonus -= 0.010
                    if concept_phrase_hits > 0:
                        bonus += min(0.010, concept_phrase_hits * 0.0025)
                if actor_overlap > 0 and action_overlap > 0:
                    bonus += min(0.008, (actor_overlap + action_overlap) * 0.0015)
                elif query_subclass in {"subject_action", "sanction_power"} and action_overlap > 0:
                    bonus += min(0.006, action_overlap * 0.0018)
                elif query_subclass in {"subject_action", "sanction_power"}:
                    bonus -= 0.003
                if query_subclass == "procedural":
                    procedural_hits = sum(
                        1
                        for tok in ("thu", "tuc", "trinh", "tu", "thoi", "han")
                        if tok in hay_tokens
                    )
                    if procedural_hits > 0:
                        bonus += min(0.006, procedural_hits * 0.0015)
                    else:
                        bonus -= 0.002

            rerank_score = base_score + (0.0005 * token_hits) + (0.0015 * phrase_hits) + bonus

            updated = dict(item)
            updated["rerank_score"] = rerank_score
            updated["original_rank"] = original_rank
            if item_law_family:
                updated["law_family_key"] = item_law_family
            reranked.append(updated)

        reranked.sort(
            key=lambda x: (x["rerank_score"], -x["original_rank"]),
            reverse=True,
        )
        for new_rank, row in enumerate(reranked):
            row["rank"] = new_rank
        return reranked

    @staticmethod
    def _decide_route(query: str, refs: dict[str, str]) -> str:
        token_count = len(tokenize_for_query(query).split())

        has_doc_number = bool(refs.get("document_number"))
        has_doc_short = bool(refs.get("document_short"))
        has_doc_loose = bool(refs.get("document_loose"))
        has_article = bool(refs.get("article_number"))
        has_clause = bool(refs.get("clause_number"))
        has_chapter = bool(refs.get("chapter"))
        has_law_name = bool(refs.get("law_name"))

        has_doc_anchor = has_doc_number or has_doc_short or has_doc_loose

        if has_doc_anchor and (has_article or has_clause):
            return "structured_exact"

        if has_doc_anchor and has_chapter and token_count <= 12:
            return "structured_exact"

        if has_doc_anchor or has_article or has_clause or has_chapter:
            return "narrow_bm25"

        if has_law_name:
            if token_count <= 5:
                return "narrow_bm25"
            return "law_anchored_hybrid"

        return "hybrid_default"

    def _rrf_fusion(self, bm25_results: list[dict], vector_results: list[dict]) -> list[dict]:
        """
        Reciprocal Rank Fusion:
        RRF_score(d) = sum(1 / (k + rank(d)))
        """
        k = self._rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for row in bm25_results:
            chunk_id = row["chunk_id"]
            rank = int(row["rank"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            docs[chunk_id] = row

        for row in vector_results:
            chunk_id = row["chunk_id"]
            rank = int(row["rank"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in docs:
                docs[chunk_id] = row

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        out: list[dict] = []
        for rank, (chunk_id, rrf_score) in enumerate(ranked):
            item = dict(docs[chunk_id])
            item["rrf_score"] = float(rrf_score)
            item["rank"] = rank
            out.append(item)
        return out

    @property
    def mode(self) -> str:
        if self._vector and self._vector.available:
            return "hybrid"
        return "bm25_only"
