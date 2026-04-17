"""
Hybrid Retrieval Engine — BM25 + Vector + RRF Fusion.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from loguru import logger

from src.retrieval.text_tokenizer import tokenize_for_query


class HybridRetriever:
    """
    Hybrid search kết hợp BM25 (keyword) và FAISS (semantic).
    Dùng Reciprocal Rank Fusion (RRF) để merge kết quả.
    Fallback: nếu vector index chưa build, chỉ dùng BM25.
    """

    def __init__(self, bm25_index_dir: Path, vector_index_dir: Path,
                 chunks_path: Path, ollama_url: str, embedding_model: str,
                 rrf_k: int = 60):
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
                logger.warning(
                    "HybridRetriever: Vector index chưa build → BM25 ONLY mode"
                )
        except Exception as e:
            logger.warning(f"Vector searcher init failed: {e} → BM25 ONLY mode")

    def search(self, query: str, bm25_top_k: int = 20,
               vector_top_k: int = 20, final_top_k: int = 10) -> list[dict]:
        """
        Hybrid search:
        1. BM25 → top_k results
        2. Vector → top_k results (nếu available)
        3. RRF fusion → final_top_k results
        """
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
        """
        Search + tr? v? ??y ?? snapshot c?c stage ?? debug/observability.
        Kh?ng l?m thay ??i logic x?p h?ng hi?n t?i.
        """
        latencies_ms: dict[str, int] = {}
        t_total = time.perf_counter()

        vector_available = bool(self._vector and self._vector.available)
        executed_parallel = False

        bm25_debug: dict = {}
        bm25_results: list[dict] = []
        vector_results: list[dict] = []

        def _run_bm25() -> tuple[list[dict], dict, int]:
            t0 = time.perf_counter()
            payload = self._bm25.search(query, bm25_top_k, return_debug=True)
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            if isinstance(payload, dict):
                return (
                    list(payload.get("results", [])),
                    dict(payload.get("debug", {})),
                    elapsed_ms,
                )
            return list(payload or []), {}, elapsed_ms

        def _run_vector() -> tuple[list[dict], int]:
            t0 = time.perf_counter()
            rows: list[dict] = []
            if vector_available:
                try:
                    rows = self._vector.search(query, vector_top_k)
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return rows, elapsed_ms

        if vector_available:
            executed_parallel = True
            bm25_future = self._executor.submit(_run_bm25)
            vector_future = self._executor.submit(_run_vector)
            bm25_results, bm25_debug, latencies_ms["bm25"] = bm25_future.result()
            vector_results, latencies_ms["vector"] = vector_future.result()
        else:
            bm25_results, bm25_debug, latencies_ms["bm25"] = _run_bm25()
            latencies_ms["vector"] = 0

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
        seen = set()
        deduped = []
        filtered_out = []
        for r in fused:
            cid = r["chunk_id"]
            if cid in seen:
                filtered_out.append({
                    "chunk_id": cid,
                    "reason": "duplicate_chunk_id",
                })
                continue
            seen.add(cid)
            deduped.append(r)
        latencies_ms["dedup"] = round((time.perf_counter() - t_dedup) * 1000)

        t_rerank = time.perf_counter()
        reranked = self._heuristic_rerank(query, deduped)
        latencies_ms["rerank"] = round((time.perf_counter() - t_rerank) * 1000)

        final_results = reranked[:final_top_k]
        for cut in reranked[final_top_k:]:
            filtered_out.append({
                "chunk_id": cut.get("chunk_id", ""),
                "reason": "top_k_cutoff",
                "rank_after_rerank": cut.get("rank"),
            })

        latencies_ms["total"] = round((time.perf_counter() - t_total) * 1000)

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
            "execution": {
                "parallel": executed_parallel,
                "vector_available": vector_available,
            },
        }

    def _rrf_fusion(self, bm25_results: list[dict],
                    vector_results: list[dict]) -> list[dict]:
        """
        Reciprocal Rank Fusion:
        RRF_score(d) = Σ 1/(k + rank(d))
        """
        k = self._rrf_k
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        # BM25 contributions
        for r in bm25_results:
            cid = r["chunk_id"]
            rank = r["rank"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            docs[cid] = r

        # Vector contributions
        for r in vector_results:
            cid = r["chunk_id"]
            rank = r["rank"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = r

        # Sort by RRF score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (cid, rrf_score) in enumerate(ranked):
            r = docs[cid].copy()
            r["rrf_score"] = rrf_score
            r["rank"] = rank
            results.append(r)

        return results

    def _heuristic_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Rerank nhẹ sau RRF để ưu tiên chunk khớp cụm pháp lý trong query.
        Mục tiêu: tăng recall cho điều/khoản đúng khi câu hỏi có cụm đặc thù
        như "điều kiện tuyển dụng công chức", "mức lương cơ sở", ...
        """
        if not candidates:
            return candidates

        query_lower = query.lower()
        query_tokens = tokenize_for_query(query).split()
        query_token_set = set(query_tokens)

        # Phrase quan trọng: unigram dài + bigram.
        phrases = {
            p for p in self._extract_query_phrases(query_tokens)
            if len(p) >= 6
        }

        has_cbcc_phrase = ("cán bộ" in query_lower and "công chức" in query_lower)
        asks_conditions = "điều kiện" in query_lower
        asks_recruitment = "tuyển dụng" in query_lower
        asks_base_salary = ("lương cơ sở" in query_lower)
        asks_exact_quote = ("trích dẫn" in query_lower) and (
            "chính xác" in query_lower or "nguyên văn" in query_lower
        )

        refs = self._extract_legal_refs(query)
        target_doc_number = refs["document_number"]
        target_doc_short = refs["document_short"]
        target_article = refs["article_number"]
        target_chapter = refs["chapter"]

        reranked = []
        for original_rank, item in enumerate(candidates):
            haystack = " ".join([
                str(item.get("title", "")),
                str(item.get("path", "")),
                str(item.get("text", "")),
                str(item.get("article", "")),
                str(item.get("clause", "")),
                str(item.get("doc_type", "")),
                str(item.get("document_number", "")),
            ]).lower()

            hay_tokens = set(tokenize_for_query(haystack).split())
            token_hits = len(query_token_set & hay_tokens)
            phrase_hits = sum(1 for p in phrases if p in haystack)

            base_score = float(item.get("rrf_score", 0.0))
            if base_score <= 0:
                # bm25-only mode fallback
                base_score = float(item.get("score", 0.0)) * 0.001

            bonus = 0.0
            doc_id = str(item.get("doc_id", ""))
            clause = str(item.get("clause", "")).strip().lower()
            item_doc_number = self._normalize_doc_number(item.get("document_number", ""))
            item_doc_short = self._extract_doc_short(item_doc_number)
            item_article = self._extract_numeric_ref(item.get("article", ""))
            item_chapter = self._extract_chapter_ref(item.get("path", ""))

            doc_match = False
            if target_doc_number:
                doc_match = item_doc_number == target_doc_number
            elif target_doc_short:
                doc_match = item_doc_short == target_doc_short

            article_match = bool(target_article and item_article == target_article)
            chapter_match = bool(target_chapter and item_chapter == target_chapter)

            # Ưu tiên văn bản đúng luật được gọi tên trong query.
            if has_cbcc_phrase and "can-bo-cong-chuc" in doc_id:
                bonus += 0.018

            # Query chứa số hiệu văn bản/Điều/Chương: ưu tiên đúng target pháp lý.
            if target_doc_number or target_doc_short:
                if doc_match:
                    bonus += 0.042
                elif item_doc_number:
                    bonus -= 0.015

            if target_article:
                if article_match:
                    bonus += 0.032
                elif doc_match and item_article:
                    bonus -= 0.010

            if target_chapter:
                if chapter_match:
                    bonus += 0.016
                elif doc_match and item_chapter:
                    bonus -= 0.006

            if doc_match and article_match:
                bonus += 0.055
                if asks_exact_quote:
                    bonus += 0.012

            if doc_match and article_match and chapter_match:
                bonus += 0.030

            # Câu hỏi điều kiện tuyển dụng công chức thường cần khoản 1 điều kiện.
            if asks_conditions and asks_recruitment:
                if "đăng ký dự tuyển công chức" in haystack:
                    bonus += 0.020
                    if clause == "khoản 1":
                        bonus += 0.028
                elif "tuyển dụng công chức" in haystack:
                    bonus += 0.010

            # Câu hỏi về lương cơ sở thường cần nghị định quy định mức lương.
            if asks_base_salary:
                if "mức lương cơ sở" in haystack:
                    bonus += 0.012
                if "nghị định" in str(item.get("doc_type", "")).lower():
                    bonus += 0.006

            # Tăng trọng số lexical match để giảm ảnh hưởng vector nhiễu.
            rerank_score = (
                base_score
                + 0.0014 * token_hits
                + 0.0045 * phrase_hits
                + bonus
            )

            updated = item.copy()
            updated["rerank_score"] = rerank_score
            updated["original_rank"] = original_rank
            reranked.append(updated)

        reranked.sort(
            key=lambda x: (x["rerank_score"], -x["original_rank"]),
            reverse=True,
        )
        for new_rank, row in enumerate(reranked):
            row["rank"] = new_rank

        return reranked

    @staticmethod
    def _extract_query_phrases(tokens: list[str]) -> set[str]:
        phrases = set(tokens)
        for i in range(len(tokens) - 1):
            phrases.add(f"{tokens[i]} {tokens[i + 1]}")
        return phrases

    @staticmethod
    def _normalize_doc_number(text: str) -> str:
        normalized = str(text or "").upper().strip()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    @staticmethod
    def _extract_doc_short(doc_number: str) -> str:
        m = re.match(r"^(\d{1,4}/\d{4})", doc_number or "")
        return m.group(1) if m else ""

    @staticmethod
    def _extract_numeric_ref(text: str) -> str:
        m = re.search(r"\b(\d+)\b", str(text or ""))
        return m.group(1) if m else ""

    @staticmethod
    def _normalize_chapter_ref(raw: str) -> str:
        value = str(raw or "").strip().upper()
        if not value:
            return ""
        if value.isdigit():
            return value
        if re.fullmatch(r"[IVXLCDM]+", value):
            return value
        return ""

    @classmethod
    def _extract_chapter_ref(cls, text: str) -> str:
        m = re.search(r"chương\s+([ivxlcdm]+|\d+)", str(text or ""), flags=re.IGNORECASE)
        if not m:
            return ""
        return cls._normalize_chapter_ref(m.group(1))

    @classmethod
    def _extract_legal_refs(cls, query: str) -> dict:
        q = str(query or "")
        q_no_space = re.sub(r"\s+", "", q)
        q_lower = q.lower()

        doc_number = ""
        m_doc = re.search(
            r"(\d{1,4}\s*/\s*\d{4}\s*/\s*[0-9a-zđA-ZĐ-]{2,20})",
            q,
            flags=re.IGNORECASE,
        )
        if m_doc:
            doc_number = cls._normalize_doc_number(m_doc.group(1))

        doc_short = ""
        m_short = re.search(r"(\d{1,4}\s*/\s*\d{4})", q_no_space)
        if m_short:
            doc_short = m_short.group(1)

        m_article = re.search(r"\bđiều\s+(\d+)\b", q_lower)
        article = m_article.group(1) if m_article else ""

        m_chapter = re.search(r"\bchương\s+([ivxlcdm]+|\d+)\b", q_lower)
        chapter = cls._normalize_chapter_ref(m_chapter.group(1)) if m_chapter else ""

        return {
            "document_number": doc_number,
            "document_short": doc_short,
            "article_number": article,
            "chapter": chapter,
        }

    @property
    def mode(self) -> str:
        """Return search mode: 'hybrid' or 'bm25_only'."""
        if self._vector and self._vector.available:
            return "hybrid"
        return "bm25_only"
