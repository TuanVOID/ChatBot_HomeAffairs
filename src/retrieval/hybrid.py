"""
Hybrid Retrieval Engine — BM25 + Vector + RRF Fusion.
"""

from pathlib import Path
from loguru import logger


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

        return deduped[:final_top_k]

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

    @property
    def mode(self) -> str:
        """Return search mode: 'hybrid' or 'bm25_only'."""
        if self._vector and self._vector.available:
            return "hybrid"
        return "bm25_only"
