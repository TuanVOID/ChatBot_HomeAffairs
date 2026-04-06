"""
Vector Searcher — Semantic search với FAISS + Ollama embedding.
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger


class VectorSearcher:
    """Wrapper cho FAISS vector index + Ollama embedding."""

    def __init__(self, index_dir: Path, ollama_url: str, model: str):
        import faiss

        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.jsonl"
        config_path = index_dir / "config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index không tồn tại: {index_path}")

        self._index = faiss.read_index(str(index_path))
        self._ollama_url = ollama_url
        self._model = model

        # Load metadata
        self._metadata: list[dict] = []
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self._metadata.append(json.loads(line.strip()))

        # Load config
        self._dim = self._index.d
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self._dim = config.get("dim", self._dim)

        logger.info(
            f"VectorSearcher loaded: {self._index.ntotal} vectors, "
            f"dim={self._dim}, model={self._model}"
        )

        # Load full chunks for text return
        self._chunks_map: dict[str, dict] = {}
        chunks_path = index_dir.parent.parent / "processed" / "chunks.jsonl"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line.strip())
                    self._chunks_map[c["chunk_id"]] = c

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts via Ollama API."""
        import httpx
        import faiss

        r = httpx.post(
            f"{self._ollama_url}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=60,
        )
        r.raise_for_status()
        embeddings = np.array(r.json()["embeddings"], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Semantic search.
        Returns list of {'chunk_id', 'score', 'rank', ...metadata}
        """
        try:
            q_emb = self._embed([query])
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]
            chunk_id = meta["chunk_id"]

            result = {
                "chunk_id": chunk_id,
                "score": float(score),
                "rank": rank,
                "title": meta.get("title", ""),
                "article": meta.get("article", ""),
                "clause": meta.get("clause", ""),
                "path": meta.get("path", ""),
                "doc_id": meta.get("doc_id", ""),
            }

            if chunk_id in self._chunks_map:
                c = self._chunks_map[chunk_id]
                result["text"] = c.get("text", "")
                result["doc_type"] = c.get("doc_type", "")
                result["issuer"] = c.get("issuer", "")
                result["issue_date"] = c.get("issue_date", "")
                result["document_number"] = c.get("document_number", "")

            results.append(result)

        return results

    @property
    def available(self) -> bool:
        """Check if vector search is available (index loaded)."""
        return self._index is not None and self._index.ntotal > 0
