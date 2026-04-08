"""
BM25 Searcher — Tìm kiếm keyword với Whoosh + Vietnamese tokenizer.
Query được pre-tokenize bằng underthesea/pyvi trước khi search
để match với pre-tokenized index.
"""

import json
from pathlib import Path

from loguru import logger
from src.retrieval.text_tokenizer import tokenize_for_query


class BM25Searcher:
    """Wrapper cho Whoosh BM25 index với Vietnamese tokenization."""

    def __init__(self, index_dir: Path, chunks_path: Path = None):
        import whoosh.index as windex
        from whoosh.qparser import MultifieldParser, OrGroup

        if not index_dir.exists():
            raise FileNotFoundError(f"BM25 index không tồn tại: {index_dir}")

        self._ix = windex.open_dir(str(index_dir))
        self._parser = MultifieldParser(
            ["content", "title"], self._ix.schema, group=OrGroup
        )

        logger.info("BM25Searcher: using internal Unicode query tokenizer")

        # Load chunks text for returning full text in results
        self._chunks_map: dict[str, dict] = {}
        if chunks_path and chunks_path.exists():
            logger.info(f"Loading chunks cache từ {chunks_path.name}...")
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        c = json.loads(line.strip())
                        self._chunks_map[c["chunk_id"]] = c
                    except (json.JSONDecodeError, KeyError):
                        continue

        logger.info(f"BM25Searcher loaded: {self._ix.doc_count()} docs, "
                     f"{len(self._chunks_map)} chunks cached")

    def _tokenize_query(self, query: str) -> str:
        """Tokenize query bằng Vietnamese tokenizer (match pre-tokenized index)."""
        return tokenize_for_query(query)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Tìm kiếm BM25.
        Query được tokenize trước → match với pre-tokenized index.
        Returns list of {'chunk_id', 'score', 'rank', ...metadata}
        """
        # Tokenize query
        tokenized_query = self._tokenize_query(query)
        if not tokenized_query:
            return []

        try:
            parsed_query = self._parser.parse(tokenized_query)
        except Exception as e:
            logger.warning(f"BM25 parse error: {e}, fallback to term query")
            from whoosh.qparser import QueryParser
            qp = QueryParser("content", self._ix.schema)
            parsed_query = qp.parse(tokenized_query)

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
                    cached = self._chunks_map[chunk_id]
                    result["text"] = cached.get("text", "")
                    result["doc_id"] = cached.get("doc_id", "")
                    result["issuer"] = cached.get("issuer", "")
                    result["issue_date"] = cached.get("issue_date", "")
                    result["document_number"] = cached.get("document_number", "")
                results.append(result)

        return results
