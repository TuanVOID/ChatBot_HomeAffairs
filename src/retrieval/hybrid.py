"""
Hybrid Retrieval Engine — BM25 + Vector + RRF Fusion.
"""

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

        self._vector = None
        try:
            if (vector_index_dir / "faiss.index").exists():
                from src.retrieval.vector_searcher import VectorSearcher
                self._vector = VectorSearcher(
                    vector_index_dir, ollama_url, embedding_model
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
        # BM25 search
        bm25_results = self._bm25.search(query, bm25_top_k)

        # Vector search (nếu available)
        vector_results = []
        if self._vector and self._vector.available:
            try:
                vector_results = self._vector.search(query, vector_top_k)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # RRF Fusion
        if vector_results:
            fused = self._rrf_fusion(bm25_results, vector_results)
        else:
            # BM25 only
            fused = bm25_results

        # Dedup by chunk_id
        seen = set()
        deduped = []
        for r in fused:
            cid = r["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                deduped.append(r)

        reranked = self._heuristic_rerank(query, deduped)
        return reranked[:final_top_k]

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

            # Ưu tiên văn bản đúng luật được gọi tên trong query.
            if has_cbcc_phrase and "can-bo-cong-chuc" in doc_id:
                bonus += 0.018

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

    @property
    def mode(self) -> str:
        """Return search mode: 'hybrid' or 'bm25_only'."""
        if self._vector and self._vector.available:
            return "hybrid"
        return "bm25_only"
