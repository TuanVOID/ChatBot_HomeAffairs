"""
BM25 searcher using Whoosh with pre-tokenized Vietnamese text.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.retrieval.text_tokenizer import tokenize_for_query


class BM25Searcher:
    """Whoosh BM25 wrapper with detailed timing breakdown."""

    def __init__(
        self,
        index_dir: Path,
        chunks_path: Path | None = None,
        chunks_map: dict[str, dict[str, Any]] | None = None,
    ):
        import whoosh.index as windex

        if not index_dir.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_dir}")

        self._windex = windex
        self._index_dir = Path(index_dir)
        self._ix = None

        self._index_open_count = 0
        self._searcher_open_count = 0
        self._search_call_count = 0
        self._last_open_index_ms = 0

        self._open_index()
        self._startup_open_index_ms = self._last_open_index_ms

        # Load chunks cache for returning full text/metadata.
        self._chunks_map: dict[str, dict[str, Any]] = {}
        if chunks_map is not None:
            self._chunks_map = chunks_map
            logger.info(
                "BM25Searcher using shared chunks cache: {} chunks",
                len(self._chunks_map),
            )
        elif chunks_path and chunks_path.exists():
            logger.info(f"Loading chunks cache from {chunks_path.name}...")
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line.strip())
                        cid = str(row.get("chunk_id", ""))
                        if cid:
                            self._chunks_map[cid] = row
                    except (json.JSONDecodeError, TypeError):
                        continue

        logger.info(
            "BM25Searcher loaded: {} docs, {} chunks cached",
            self._ix.doc_count(),
            len(self._chunks_map),
        )

    @property
    def chunks_map(self) -> dict[str, dict[str, Any]]:
        return self._chunks_map

    def _open_index(self) -> None:
        t0 = time.perf_counter()
        self._ix = self._windex.open_dir(str(self._index_dir))
        self._last_open_index_ms = round((time.perf_counter() - t0) * 1000)
        self._index_open_count += 1

    @staticmethod
    def _extract_query_fields(parsed_query: str) -> list[str]:
        fields = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:", parsed_query or ""))
        return sorted(fields)

    def _tokenize_query(self, query: str) -> str:
        return tokenize_for_query(query)

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        return_debug: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        BM25 search.

        When return_debug=True, returns:
        {
          "results": [...],
          "debug": {
              open_index_ms, open_searcher_ms, build_query_ms,
              execute_search_ms, hydrate_hits_ms, ...
          }
        }
        """
        from whoosh.qparser import MultifieldParser, OrGroup, QueryParser

        self._search_call_count += 1

        open_index_ms = 0
        if self._ix is None:
            self._open_index()
            open_index_ms = self._last_open_index_ms

        tokenized_query = self._tokenize_query(query)
        if not tokenized_query:
            empty_debug = {
                "open_index_ms": open_index_ms,
                "open_searcher_ms": 0,
                "build_query_ms": 0,
                "execute_search_ms": 0,
                "hydrate_hits_ms": 0,
                "top_k": int(top_k),
                "tokenized_query": "",
                "parsed_query": "",
                "query_fields": [],
                "is_multifield": False,
                "has_wildcard": False,
                "has_fuzzy": False,
                "fallback_parser": False,
                "open_dir_reused": open_index_ms == 0,
                "index_open_count": self._index_open_count,
                "searcher_open_count": self._searcher_open_count,
                "search_call_count": self._search_call_count,
                "startup_open_index_ms": self._startup_open_index_ms,
                "results_count": 0,
            }
            if return_debug:
                return {"results": [], "debug": empty_debug}
            return []

        t_build_query = time.perf_counter()
        fallback_parser = False
        try:
            parser = MultifieldParser(["content", "title"], self._ix.schema, group=OrGroup)
            parsed_query = parser.parse(tokenized_query)
        except Exception as exc:
            logger.warning(f"BM25 parse error: {exc}, fallback to QueryParser(content)")
            fallback_parser = True
            qp = QueryParser("content", self._ix.schema)
            parsed_query = qp.parse(tokenized_query)
        build_query_ms = round((time.perf_counter() - t_build_query) * 1000)

        parsed_query_str = str(parsed_query)
        query_fields = self._extract_query_fields(parsed_query_str)

        results: list[dict[str, Any]] = []

        t_open_searcher = time.perf_counter()
        searcher = self._ix.searcher()
        open_searcher_ms = round((time.perf_counter() - t_open_searcher) * 1000)
        self._searcher_open_count += 1

        execute_search_ms = 0
        hydrate_hits_ms = 0
        try:
            t_execute = time.perf_counter()
            hits = searcher.search(parsed_query, limit=top_k)
            execute_search_ms = round((time.perf_counter() - t_execute) * 1000)

            t_hydrate = time.perf_counter()
            for rank, hit in enumerate(hits):
                chunk_id = str(hit.get("chunk_id", ""))
                result: dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "score": float(hit.score),
                    "rank": rank,
                    "title": hit.get("title", ""),
                    "doc_type": hit.get("doc_type", ""),
                    "article": hit.get("article", ""),
                    "clause": hit.get("clause", ""),
                    "path": hit.get("path", ""),
                }

                cached = self._chunks_map.get(chunk_id)
                if cached:
                    result["text"] = cached.get("text", "")
                    result["doc_id"] = cached.get("doc_id", "")
                    result["issuer"] = cached.get("issuer", "")
                    result["issue_date"] = cached.get("issue_date", "")
                    result["document_number"] = cached.get("document_number", "")

                results.append(result)

            hydrate_hits_ms = round((time.perf_counter() - t_hydrate) * 1000)
        finally:
            searcher.close()

        debug_payload = {
            "open_index_ms": open_index_ms,
            "open_searcher_ms": open_searcher_ms,
            "build_query_ms": build_query_ms,
            "execute_search_ms": execute_search_ms,
            "hydrate_hits_ms": hydrate_hits_ms,
            "top_k": int(top_k),
            "tokenized_query": tokenized_query,
            "parsed_query": parsed_query_str,
            "query_fields": query_fields,
            "is_multifield": len(query_fields) > 1,
            "has_wildcard": "*" in parsed_query_str,
            "has_fuzzy": "~" in parsed_query_str,
            "fallback_parser": fallback_parser,
            "open_dir_reused": open_index_ms == 0,
            "index_open_count": self._index_open_count,
            "searcher_open_count": self._searcher_open_count,
            "search_call_count": self._search_call_count,
            "startup_open_index_ms": self._startup_open_index_ms,
            "results_count": len(results),
        }

        if return_debug:
            return {"results": results, "debug": debug_payload}
        return results
