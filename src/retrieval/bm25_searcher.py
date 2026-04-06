"""
BM25 Searcher — Tìm kiếm keyword với Whoosh.
"""

import json
from pathlib import Path

from loguru import logger


class BM25Searcher:
    """Wrapper cho Whoosh BM25 index."""

    def __init__(self, index_dir: Path, chunks_path: Path = None):
        import whoosh.index as windex
        from whoosh.qparser import MultifieldParser, OrGroup

        if not index_dir.exists():
            raise FileNotFoundError(f"BM25 index không tồn tại: {index_dir}")

        self._ix = windex.open_dir(str(index_dir))
        self._parser = MultifieldParser(
            ["content", "title"], self._ix.schema, group=OrGroup
        )

        # Load chunks text for returning
        self._chunks_map: dict[str, dict] = {}
        if chunks_path and chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line.strip())
                    self._chunks_map[c["chunk_id"]] = c

        logger.info(f"BM25Searcher loaded: {self._ix.doc_count()} docs, "
                     f"{len(self._chunks_map)} chunks cached")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Tìm kiếm BM25.
        Returns list of {'chunk_id', 'score', 'rank', ...metadata}
        """
        try:
            parsed_query = self._parser.parse(query)
        except Exception as e:
            logger.warning(f"BM25 parse error: {e}, fallback to term query")
            from whoosh.qparser import QueryParser
            qp = QueryParser("content", self._ix.schema)
            parsed_query = qp.parse(query)

        results = []
        with self._ix.searcher() as searcher:
            hits = searcher.search(parsed_query, limit=top_k)
            for rank, hit in enumerate(hits):
                chunk_id = hit["chunk_id"]
                result = {
                    "chunk_id": chunk_id,
                    "score": float(hit.score),
                    "rank": rank,
                    "title": hit.get("title", ""),
                    "doc_type": hit.get("doc_type", ""),
                    "article": hit.get("article", ""),
                    "clause": hit.get("clause", ""),
                    "path": hit.get("path", ""),
                }
                # Add full text from chunks cache
                if chunk_id in self._chunks_map:
                    result["text"] = self._chunks_map[chunk_id].get("text", "")
                    result["doc_id"] = self._chunks_map[chunk_id].get("doc_id", "")
                    result["issuer"] = self._chunks_map[chunk_id].get("issuer", "")
                    result["issue_date"] = self._chunks_map[chunk_id].get("issue_date", "")
                    result["document_number"] = self._chunks_map[chunk_id].get("document_number", "")
                results.append(result)

        return results
