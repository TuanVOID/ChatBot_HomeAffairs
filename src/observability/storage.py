from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SQLiteObservabilityStore:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS trace_runs (
            trace_id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            user_query TEXT NOT NULL,
            query_tokenized TEXT,
            query_type TEXT,
            expected_sources TEXT,
            reference_note TEXT,
            prompt_version TEXT,
            retrieval_config_version TEXT,
            index_version TEXT,
            model_name TEXT,
            model_version TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            retrieval_latency_ms INTEGER,
            total_latency_ms INTEGER,
            llm_latency_ms INTEGER,
            answer_text TEXT,
            used_context_json TEXT,
            citations_json TEXT,
            grounded_flag INTEGER,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            doc_id TEXT,
            title TEXT,
            article_ref TEXT,
            breadcrumb TEXT,
            score REAL,
            rank INTEGER,
            kept INTEGER NOT NULL DEFAULT 1,
            filter_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_cases (
            case_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            expected_doc_id TEXT,
            expected_article_ref TEXT,
            expected_keywords TEXT,
            gold_answer TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            total_cases INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            trace_id TEXT,
            hit_top5 INTEGER,
            hit_top10 INTEGER,
            rank_of_first_correct INTEGER,
            hit_expected_source INTEGER,
            hit_expected_article INTEGER,
            citation_correct INTEGER,
            grounded INTEGER,
            completeness_score INTEGER,
            hallucination INTEGER,
            retrieval_latency_ms INTEGER,
            answer_latency_ms INTEGER,
            total_latency_ms INTEGER,
            notes TEXT
        );
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        if not params:
            return
        with self._lock:
            self._conn.executemany(sql, params)
            self._conn.commit()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def upsert_trace_start(
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
        status: str = "running",
        started_at: str | None = None,
    ) -> None:
        started = started_at or utc_now_iso()
        sql = """
        INSERT INTO trace_runs (
            trace_id, endpoint, user_query, query_tokenized,
            prompt_version, retrieval_config_version, index_version,
            model_name, model_version, started_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id) DO UPDATE SET
            endpoint=excluded.endpoint,
            user_query=excluded.user_query,
            query_tokenized=COALESCE(excluded.query_tokenized, trace_runs.query_tokenized),
            prompt_version=COALESCE(excluded.prompt_version, trace_runs.prompt_version),
            retrieval_config_version=COALESCE(excluded.retrieval_config_version, trace_runs.retrieval_config_version),
            index_version=COALESCE(excluded.index_version, trace_runs.index_version),
            model_name=COALESCE(excluded.model_name, trace_runs.model_name),
            model_version=COALESCE(excluded.model_version, trace_runs.model_version),
            started_at=excluded.started_at,
            status=excluded.status
        """
        self._execute(sql, (
            trace_id, endpoint, user_query, query_tokenized,
            prompt_version, retrieval_config_version, index_version,
            model_name, model_version, started, status
        ))

    def update_trace(self, trace_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        vals: list[Any] = []
        for key, value in fields.items():
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(trace_id)
        sql = f"UPDATE trace_runs SET {', '.join(cols)} WHERE trace_id = ?"
        self._execute(sql, tuple(vals))

    def insert_event(self, trace_id: str, stage: str, payload: dict[str, Any], created_at: str | None = None) -> None:
        sql = """
        INSERT INTO trace_events (trace_id, stage, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """
        self._execute(sql, (
            trace_id,
            stage,
            json.dumps(payload, ensure_ascii=False),
            created_at or utc_now_iso(),
        ))

    def insert_hits(
        self,
        trace_id: str,
        stage: str,
        hits: list[dict[str, Any]],
        *,
        kept: int = 1,
        default_filter_reason: str | None = None,
    ) -> None:
        sql = """
        INSERT INTO retrieval_hits (
            trace_id, stage, chunk_id, doc_id, title,
            article_ref, breadcrumb, score, rank, kept, filter_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows: list[tuple[Any, ...]] = []
        for hit in hits:
            rows.append((
                trace_id,
                stage,
                hit.get("chunk_id", ""),
                hit.get("doc_id", ""),
                hit.get("title", ""),
                hit.get("article_ref"),
                hit.get("breadcrumb"),
                float(hit.get("score", 0.0)),
                int(hit.get("rank", 0)),
                kept,
                hit.get("filter_reason", default_filter_reason),
            ))
        self._executemany(sql, rows)

    def get_trace(self, trace_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM trace_runs WHERE trace_id = ?", (trace_id,))

    def get_events(self, trace_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT id, trace_id, stage, payload_json, created_at "
            "FROM trace_events WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        )

    def get_hits(self, trace_id: str, stage: str | None = None) -> list[dict]:
        if stage:
            return self._fetchall(
                "SELECT * FROM retrieval_hits WHERE trace_id = ? AND stage = ? ORDER BY rank ASC, id ASC",
                (trace_id, stage),
            )
        return self._fetchall(
            "SELECT * FROM retrieval_hits WHERE trace_id = ? ORDER BY stage ASC, rank ASC, id ASC",
            (trace_id,),
        )

    def list_recent_traces(self, limit: int = 50) -> list[dict]:
        lim = max(1, min(int(limit), 500))
        return self._fetchall(
            "SELECT trace_id, endpoint, user_query, status, total_latency_ms, started_at, finished_at "
            "FROM trace_runs ORDER BY started_at DESC LIMIT ?",
            (lim,),
        )

    def upsert_eval_cases(self, cases: list[dict[str, Any]]) -> int:
        sql = """
        INSERT INTO eval_cases (
            case_id, question, expected_doc_id,
            expected_article_ref, expected_keywords, gold_answer
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            question=excluded.question,
            expected_doc_id=excluded.expected_doc_id,
            expected_article_ref=excluded.expected_article_ref,
            expected_keywords=excluded.expected_keywords,
            gold_answer=excluded.gold_answer
        """
        rows: list[tuple[Any, ...]] = []
        for case in cases:
            rows.append((
                case.get("case_id"),
                case.get("question", ""),
                case.get("expected_doc_id"),
                case.get("expected_article_ref"),
                json.dumps(case.get("expected_keywords", []), ensure_ascii=False),
                case.get("gold_answer"),
            ))
        self._executemany(sql, rows)
        return len(rows)

    def list_eval_cases(self, limit: int | None = None) -> list[dict]:
        if limit is None:
            rows = self._fetchall("SELECT * FROM eval_cases ORDER BY case_id ASC")
        else:
            lim = max(1, min(int(limit), 10_000))
            rows = self._fetchall("SELECT * FROM eval_cases ORDER BY case_id ASC LIMIT ?", (lim,))
        for row in rows:
            raw = row.get("expected_keywords")
            try:
                row["expected_keywords"] = json.loads(raw) if raw else []
            except Exception:
                row["expected_keywords"] = []
        return rows

    def create_eval_run(self, run_id: str, total_cases: int) -> None:
        sql = """
        INSERT INTO eval_runs (run_id, started_at, status, total_cases)
        VALUES (?, ?, 'running', ?)
        ON CONFLICT(run_id) DO UPDATE SET
            started_at=excluded.started_at,
            status='running',
            total_cases=excluded.total_cases,
            finished_at=NULL,
            summary_json=NULL,
            error_message=NULL
        """
        self._execute(sql, (run_id, utc_now_iso(), int(total_cases)))

    def finish_eval_run(
        self,
        run_id: str,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        self._execute(
            "UPDATE eval_runs SET status = ?, finished_at = ?, summary_json = ?, error_message = ? WHERE run_id = ?",
            (
                status,
                utc_now_iso(),
                json.dumps(summary or {}, ensure_ascii=False) if summary is not None else None,
                error_message,
                run_id,
            ),
        )

    def insert_eval_result(self, record: dict[str, Any]) -> None:
        sql = """
        INSERT INTO eval_results (
            run_id, case_id, trace_id, hit_top5, hit_top10, rank_of_first_correct,
            hit_expected_source, hit_expected_article, citation_correct, grounded,
            completeness_score, hallucination, retrieval_latency_ms, answer_latency_ms,
            total_latency_ms, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute(sql, (
            record.get("run_id"),
            record.get("case_id"),
            record.get("trace_id"),
            record.get("hit_top5"),
            record.get("hit_top10"),
            record.get("rank_of_first_correct"),
            record.get("hit_expected_source"),
            record.get("hit_expected_article"),
            record.get("citation_correct"),
            record.get("grounded"),
            record.get("completeness_score"),
            record.get("hallucination"),
            record.get("retrieval_latency_ms"),
            record.get("answer_latency_ms"),
            record.get("total_latency_ms"),
            json.dumps(record.get("notes", {}), ensure_ascii=False),
        ))

    def get_eval_run(self, run_id: str) -> dict | None:
        row = self._fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        if row and row.get("summary_json"):
            try:
                row["summary_json"] = json.loads(row["summary_json"])
            except Exception:
                row["summary_json"] = {}
        return row

    def list_eval_results(self, run_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        for row in rows:
            raw = row.get("notes")
            try:
                row["notes"] = json.loads(raw) if raw else {}
            except Exception:
                row["notes"] = {}
        return rows

