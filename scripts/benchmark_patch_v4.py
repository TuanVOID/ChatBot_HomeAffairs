"""
Expanded benchmark runner for patch v6.

What this script does:
1. Build expanded eval set (65+ cases) across retrieval routes:
   - structured_exact
   - doc_scoped_hybrid_exact_doc
   - doc_scoped_hybrid_candidate_docs
   - law_anchored_hybrid_loose
   - narrow_bm25
   - hybrid_default
2. Add richer eval fields:
   - expected_route
   - expected_focus_terms
   - expected_doc_number / expected_article_num / expected_clause_num
3. Run warm + cold-ish benchmarks.
4. Output:
   - benchmark_expanded_summary.json
   - benchmark_expanded_results.json
   - benchmark_expanded_compare_exact.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import cfg
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.legal_query_parser import normalize_doc_number, parse_legal_refs


EXPECTED_ROUTES = (
    "structured_exact",
    "doc_scoped_hybrid_exact_doc",
    "doc_scoped_hybrid_candidate_docs",
    "law_anchored_hybrid_loose",
    "narrow_bm25",
    "hybrid_default",
)

CASE_MIN_COUNTS = {
    "structured_exact": 20,
    "doc_scoped_hybrid_exact_doc": 6,
    "doc_scoped_hybrid_candidate_docs": 4,
    "narrow_bm25": 15,
    "law_anchored_hybrid_loose": 5,
    "hybrid_default": 15,
}

STOP_TOKENS = {
    "la",
    "gi",
    "nao",
    "nhu",
    "the",
    "ra",
    "sao",
    "theo",
    "duoc",
    "quy",
    "dinh",
    "noi",
    "dung",
    "trong",
    "tam",
    "cua",
    "van",
    "ban",
    "nay",
    "ve",
    "mot",
    "so",
    "nhung",
    "khi",
    "thi",
    "can",
    "cho",
    "toi",
    "biet",
    "tom",
    "tat",
    "nhanh",
    "hien",
    "hanh",
    "thuc",
    "te",
    "nguoi",
    "co",
    "va",
    "tai",
    "day",
    "cac",
    "chu",
    "yeu",
    "sua",
    "doi",
    "bo",
    "luat",
    "dieu",
    "khoan",
    "chuong",
    "nam",
}

SOFT_FAMILY_EQUIV_ROUTES = {
    "doc_scoped_hybrid_candidate_docs",
    "law_anchored_hybrid_loose",
    "law_anchored_hybrid",
    "hybrid_default",
}

DOC_META_BY_ID: dict[str, dict[str, Any]] = {}

ACCEPTABLE_ROUTE_PROMOTION_LABELS = {
    "ROUTE_ACCEPTABLE_ALIAS_TIE",
    "ROUTE_ACCEPTABLE_SCOPE_PROMOTION",
}

FOCUS_FAILURE_LABELS = {
    "BM25_NOT_FOCUSED",
    "QUERY_TOO_VAGUE",
    "RERANK_BAD_TOP1",
}


def _to_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or ""))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _canonical_law_family_key(title: str) -> str:
    text = _to_ascii(title).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\bvan\s+ban\s+hop\s+nhat\b", " ", text)
    text = re.sub(r"\b(sua\s+doi|bo\s+sung|hop\s+nhat)\b", " ", text)
    m = re.search(r"\b(bo\s+luat|luat)\b\s+(.+)", text)
    if m:
        text = f"{m.group(1)} {m.group(2)}"
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_doc_key(doc_number: str, doc_id: str = "") -> str:
    number = normalize_doc_number(str(doc_number or ""))
    if number:
        return f"doc::{number}"
    doc = _norm_text(doc_id)
    if doc:
        return f"id::{doc}"
    return ""


def _expected_family_key(case: dict[str, Any]) -> str:
    expected_doc_id = _norm_text(case.get("expected_doc_id", ""))
    if expected_doc_id:
        meta = DOC_META_BY_ID.get(expected_doc_id, {})
        key = str(meta.get("canonical_law_key", "")).strip() or str(meta.get("canonical_doc_key", "")).strip()
        if key:
            return key

    expected_doc_number = str(case.get("expected_doc_number", "")).strip()
    if expected_doc_number:
        return _canonical_doc_key(expected_doc_number, expected_doc_id)
    return ""


def _hit_family_key(hit: dict[str, Any]) -> str:
    hit_doc_id = _norm_text(hit.get("doc_id", ""))
    if hit_doc_id:
        meta = DOC_META_BY_ID.get(hit_doc_id, {})
        key = str(meta.get("canonical_law_key", "")).strip() or str(meta.get("canonical_doc_key", "")).strip()
        if key:
            return key

    title = str(hit.get("title", "")).strip()
    law_key = _canonical_law_family_key(title)
    if law_key:
        return law_key
    return _canonical_doc_key(hit.get("document_number", ""), hit_doc_id)


def _family_match(expected_key: str, hit_key: str) -> bool:
    if not expected_key or not hit_key:
        return False
    return (
        expected_key == hit_key
        or expected_key in hit_key
        or hit_key in expected_key
    )


def _tokenize_ascii(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _to_ascii(text).lower())


def _extract_num_token(text: str) -> str:
    m = re.search(r"\b(\d+[a-z]?)\b", _to_ascii(text).lower())
    return m.group(1) if m else ""


def _parse_expected_article_clause(ref: str) -> tuple[str, str]:
    tokens = re.findall(r"\b(\d+[a-z]?)\b", _to_ascii(ref).lower())
    article = tokens[0] if len(tokens) >= 1 else ""
    clause = tokens[1] if len(tokens) >= 2 else ""
    return article, clause


def _extract_focus_terms(*texts: str, limit: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for tok in _tokenize_ascii(text):
            if not tok or tok in STOP_TOKENS:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 3:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                return out
    return out


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    ordered = sorted(int(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return int(round(ordered[lo] * (1 - frac) + ordered[hi] * frac))


def _summary_stats(values: list[int]) -> dict[str, int]:
    vals = [int(v) for v in values]
    return {
        "avg": int(round(statistics.mean(vals))) if vals else 0,
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
        "max": max(vals) if vals else 0,
    }


def _load_documents(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    docs_map: dict[str, dict[str, Any]] = {}
    law_candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", "")).strip()
            if not doc_id:
                continue
            title = str(obj.get("title", "")).strip()
            doc_number = str(obj.get("document_number", "")).strip()
            doc_type = str(obj.get("doc_type", "")).strip()
            canonical_law_key = _canonical_law_family_key(title)
            canonical_doc_key = _canonical_doc_key(doc_number, doc_id)
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "document_number": doc_number,
                "doc_type": doc_type,
                "canonical_law_key": canonical_law_key,
                "canonical_doc_key": canonical_doc_key,
            }

            title_ascii = _to_ascii(title).lower()
            doc_type_ascii = _to_ascii(doc_type).lower()
            if (
                doc_type_ascii == "luat"
                or title_ascii.startswith("luat ")
                or title_ascii.startswith("bo luat ")
            ):
                law_candidates.append(docs_map[doc_id])
    return docs_map, law_candidates


def _load_base_exact_cases(db_path: Path, limit: int = 20) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT case_id, question, expected_doc_id, expected_article_ref "
        "FROM eval_cases WHERE case_id LIKE 'debug_bm25_%' "
        "ORDER BY case_id LIMIT ?",
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "case_id": row[0],
            "question": row[1] or "",
            "expected_doc_id": row[2] or "",
            "expected_article_ref": row[3] or "",
        }
        for row in rows
    ]


def _collect_chunk_samples(
    chunks_path: Path,
    *,
    exact_targets: set[tuple[str, str, str]],
    law_doc_ids: set[str],
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, dict[str, str]]]:
    found_exact: dict[tuple[str, str, str], dict[str, str]] = {}
    found_law: dict[str, dict[str, str]] = {}
    watched_doc_ids = {doc_id for doc_id, _, _ in exact_targets} | set(law_doc_ids)

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", ""))
            if doc_id not in watched_doc_ids:
                continue

            article_num = _extract_num_token(obj.get("article", ""))
            clause_num = _extract_num_token(obj.get("clause", ""))
            sample = {
                "chunk_id": str(obj.get("chunk_id", "")),
                "article_num": article_num,
                "clause_num": clause_num,
                "title": str(obj.get("title", "")),
                "path": str(obj.get("path", "")),
                "text": str(obj.get("text", "")),
            }

            if doc_id in law_doc_ids and doc_id not in found_law and article_num:
                found_law[doc_id] = sample

            key = (doc_id, article_num, clause_num)
            if key in exact_targets and key not in found_exact:
                found_exact[key] = sample

            fallback_key = (doc_id, article_num, "")
            if fallback_key in exact_targets and fallback_key not in found_exact:
                found_exact[fallback_key] = sample

            if len(found_exact) >= len(exact_targets) and len(found_law) >= len(law_doc_ids):
                break

    return found_exact, found_law


def _extract_law_name_for_query(title: str) -> str:
    raw = _to_ascii(title)
    clean = re.sub(r"[^A-Za-z0-9,\\-\\s/]", " ", raw)
    clean = re.sub(r"\s+", " ", clean).strip()
    m = re.search(r"\b(bo luat|luat)\b\s+(.+)", clean, flags=re.IGNORECASE)
    if not m:
        return ""

    prefix = m.group(1).title()
    suffix = m.group(2)
    suffix = re.split(r"\bnam\s+\d{4}\b", suffix, maxsplit=1, flags=re.IGNORECASE)[0]
    suffix = re.split(r"\bsua doi\b", suffix, maxsplit=1, flags=re.IGNORECASE)[0]
    suffix = re.split(r"\bhuong dan\b", suffix, maxsplit=1, flags=re.IGNORECASE)[0]
    suffix = re.split(r"\bve\b", suffix, maxsplit=1, flags=re.IGNORECASE)[0]
    suffix = re.sub(r"\s+", " ", suffix).strip(" ,.-")
    if not suffix:
        return ""

    keep_words = suffix.split()[:8]
    phrase = f"{prefix} {' '.join(keep_words)}".strip()
    phrase = re.sub(r"\s+,", ",", phrase)
    return phrase


def _query_doc_number(doc_number: str) -> str:
    return _to_ascii(doc_number).upper().replace(" ", "")


def _expected_route_of_query(query: str) -> str:
    refs = parse_legal_refs(query)
    return HybridRetriever._decide_route(query, refs)


def _natural_focus_phrase(
    focus_terms: list[str],
    sample_text: str = "",
    title: str = "",
) -> str:
    bag = set(_tokenize_ascii(" ".join([sample_text, title, " ".join(focus_terms)])))
    patterns = [
        ({"pham", "vi", "dieu", "chinh"}, "pham vi dieu chinh"),
        ({"quyen", "nghia", "vu"}, "quyen va nghia vu"),
        ({"dieu", "kien", "tuyen", "dung"}, "dieu kien tuyen dung"),
        ({"tham", "quyen"}, "tham quyen"),
        ({"trach", "nhiem"}, "trach nhiem"),
        ({"xu", "ly", "vi", "pham"}, "xu ly vi pham"),
        ({"hop", "dong", "lao", "dong"}, "hop dong lao dong"),
        ({"bao", "hiem", "xa", "hoi"}, "bao hiem xa hoi"),
        ({"boi", "thuong"}, "boi thuong thiet hai"),
    ]
    for must, phrase in patterns:
        if len(must & bag) >= max(2, len(must) - 1):
            return phrase

    tokens = [tok for tok in focus_terms if tok and len(tok) >= 3]
    if len(tokens) >= 3:
        return " ".join(tokens[:3])
    if len(tokens) >= 2:
        return " ".join(tokens[:2])
    if len(tokens) == 1:
        return tokens[0]
    return "quy dinh lien quan"


def _law_hint_phrase(law_name: str, max_tokens: int = 3) -> str:
    tokens = [tok for tok in _tokenize_ascii(law_name) if tok not in {"luat", "bo"}]
    if not tokens:
        return ""
    return " ".join(tokens[: max(1, int(max_tokens))])


def _build_expanded_cases(
    *,
    base_exact: list[dict[str, Any]],
    docs_map: dict[str, dict[str, Any]],
    law_candidates: list[dict[str, Any]],
    exact_samples: dict[tuple[str, str, str], dict[str, str]],
    law_samples: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    cases_structured: list[dict[str, Any]] = []
    cases_narrow: list[dict[str, Any]] = []
    cases_doc_scoped_exact: list[dict[str, Any]] = []
    cases_doc_scoped_candidate: list[dict[str, Any]] = []
    cases_law_loose: list[dict[str, Any]] = []
    cases_hybrid_default: list[dict[str, Any]] = []

    structured_source: list[dict[str, Any]] = []
    for base in base_exact:
        doc_id = str(base.get("expected_doc_id", ""))
        if not doc_id:
            continue
        meta = docs_map.get(doc_id, {})
        doc_number = str(meta.get("document_number", "")).strip()
        article_num, clause_num = _parse_expected_article_clause(base.get("expected_article_ref", ""))

        sample = exact_samples.get((doc_id, article_num, clause_num))
        if sample is None:
            sample = exact_samples.get((doc_id, article_num, ""))

        focus_terms = _extract_focus_terms(
            sample.get("text", "") if sample else "",
            sample.get("path", "") if sample else "",
            meta.get("title", ""),
            base.get("question", ""),
        )
        if not focus_terms:
            focus_terms = ["quy", "dinh"]

        item = {
            "source_case_id": base.get("case_id", ""),
            "question": base.get("question", ""),
            "expected_doc_id": doc_id,
            "expected_article_ref": base.get("expected_article_ref", ""),
            "expected_doc_number": doc_number,
            "expected_article_num": article_num,
            "expected_clause_num": clause_num,
            "expected_focus_terms": focus_terms[:3],
            "title": meta.get("title", ""),
        }
        structured_source.append(item)

    structured_source = structured_source[:20]
    for idx, item in enumerate(structured_source, 1):
        case = {
            "case_id": f"v4_structured_{idx:02d}",
            "question": item["question"],
            "expected_doc_id": item["expected_doc_id"],
            "expected_article_ref": item["expected_article_ref"],
            "expected_keywords": item["expected_focus_terms"],
            "expected_route": "structured_exact",
            "expected_focus_terms": item["expected_focus_terms"],
            "expected_topical_terms": item["expected_focus_terms"],
            "expected_doc_number": item["expected_doc_number"],
            "expected_article_num": item["expected_article_num"],
            "expected_clause_num": item["expected_clause_num"],
            "notes": f"seed_from_{item['source_case_id']}",
            "gold_answer": None,
        }
        cases_structured.append(case)

    for idx, item in enumerate(structured_source[:15], 1):
        doc_num_query = _query_doc_number(item["expected_doc_number"])
        focus_phrase = _natural_focus_phrase(
            item["expected_focus_terms"],
            title=item.get("title", ""),
        )
        if idx % 3 == 1:
            question = f"Theo van ban {doc_num_query}, quy dinh ve {focus_phrase} duoc ap dung nhu the nao?"
        elif idx % 3 == 2:
            question = f"Trong van ban {doc_num_query}, noi dung ve {focus_phrase} duoc neu ra sao?"
        else:
            question = f"Van ban {doc_num_query} quy dinh gi lien quan den {focus_phrase}?"

        case = {
            "case_id": f"v4_narrow_{idx:02d}",
            "question": question,
            "expected_doc_id": item["expected_doc_id"],
            "expected_article_ref": "",
            "expected_keywords": item["expected_focus_terms"],
            "expected_route": "narrow_bm25",
            "expected_focus_terms": item["expected_focus_terms"],
            "expected_topical_terms": item["expected_focus_terms"],
            "expected_doc_number": item["expected_doc_number"],
            "expected_article_num": "",
            "expected_clause_num": "",
            "notes": f"doc_only_from_{item['source_case_id']}",
            "gold_answer": None,
        }
        if _expected_route_of_query(question) == "narrow_bm25":
            cases_narrow.append(case)

    used_doc_ids = {item["expected_doc_id"] for item in structured_source}
    for meta in law_candidates:
        doc_id = str(meta.get("doc_id", ""))
        if not doc_id or doc_id in used_doc_ids:
            continue
        if doc_id not in law_samples:
            continue
        law_name = _extract_law_name_for_query(meta.get("title", ""))
        if not law_name:
            continue
        sample = law_samples.get(doc_id, {})
        focus_terms = _extract_focus_terms(
            sample.get("text", ""),
            sample.get("path", ""),
            meta.get("title", ""),
        )
        if len(focus_terms) < 2:
            continue

        focus_phrase = _natural_focus_phrase(
            focus_terms[:3],
            sample_text=sample.get("text", ""),
            title=str(meta.get("title", "")),
        )
        law_hint = _law_hint_phrase(law_name, max_tokens=3)
        if not law_hint:
            continue
        law_hint_loose = _law_hint_phrase(law_name, max_tokens=2)
        if not law_hint_loose:
            law_hint_loose = law_hint

        if len(cases_doc_scoped_exact) < 6:
            idx = len(cases_doc_scoped_exact) + 1
            if idx % 2 == 1:
                question = f"Theo {law_name}, quy dinh ve {focus_phrase} duoc neu nhu the nao?"
            else:
                question = f"Theo {law_name}, noi dung lien quan den {focus_phrase} la gi?"

            case = {
                "case_id": f"v4_doc_scoped_exact_{idx:02d}",
                "question": question,
                "expected_doc_id": doc_id,
                "expected_article_ref": "",
                "expected_keywords": focus_terms[:3],
                "expected_route": "doc_scoped_hybrid_exact_doc",
                "expected_focus_terms": focus_terms[:3],
                "expected_topical_terms": focus_terms[:3],
                "expected_doc_number": str(meta.get("document_number", "")),
                "expected_article_num": "",
                "expected_clause_num": "",
                "notes": "law_name_exact_doc",
                "gold_answer": None,
            }
            if _expected_route_of_query(question) == "law_anchored_hybrid":
                cases_doc_scoped_exact.append(case)

        if len(cases_doc_scoped_candidate) < 4:
            idx = len(cases_doc_scoped_candidate) + 1
            if idx % 2 == 1:
                question = f"Theo luat {law_hint}, quy dinh lien quan den {focus_phrase} nhu the nao?"
            else:
                question = f"Trong luat {law_hint}, noi dung ve {focus_phrase} duoc quy dinh ra sao?"

            case = {
                "case_id": f"v4_doc_scoped_candidate_{idx:02d}",
                "question": question,
                "expected_doc_id": doc_id,
                "expected_article_ref": "",
                "expected_keywords": focus_terms[:3],
                "expected_route": "doc_scoped_hybrid_candidate_docs",
                "expected_focus_terms": focus_terms[:3],
                "expected_topical_terms": focus_terms[:3],
                "expected_doc_number": str(meta.get("document_number", "")),
                "expected_article_num": "",
                "expected_clause_num": "",
                "notes": "law_name_candidate_docs",
                "gold_answer": None,
            }
            if _expected_route_of_query(question) == "law_anchored_hybrid":
                cases_doc_scoped_candidate.append(case)

        if len(cases_law_loose) < 5:
            idx = len(cases_law_loose) + 1
            if idx % 2 == 1:
                question = f"Trong cac quy dinh cua luat ve {law_hint_loose}, {focus_phrase} duoc de cap ra sao?"
            else:
                question = f"Luat ve {law_hint_loose} hien nay quy dinh nhu the nao ve {focus_phrase}?"

            case = {
                "case_id": f"v4_law_loose_{idx:02d}",
                "question": question,
                "expected_doc_id": doc_id,
                "expected_article_ref": "",
                "expected_keywords": focus_terms[:3],
                "expected_route": "law_anchored_hybrid_loose",
                "expected_focus_terms": focus_terms[:3],
                "expected_topical_terms": focus_terms[:3],
                "expected_doc_number": str(meta.get("document_number", "")),
                "expected_article_num": "",
                "expected_clause_num": "",
                "notes": "law_name_loose_query",
                "gold_answer": None,
            }
            if _expected_route_of_query(question) == "law_anchored_hybrid":
                cases_law_loose.append(case)

        if (
            len(cases_doc_scoped_exact) >= 6
            and len(cases_doc_scoped_candidate) >= 4
            and len(cases_law_loose) >= 5
        ):
            break

    for idx, case in enumerate(cases_doc_scoped_exact, 1):
        case["case_id"] = f"v4_doc_scoped_exact_{idx:02d}"
    for idx, case in enumerate(cases_doc_scoped_candidate, 1):
        case["case_id"] = f"v4_doc_scoped_candidate_{idx:02d}"
    for idx, case in enumerate(cases_law_loose, 1):
        case["case_id"] = f"v4_law_loose_{idx:02d}"

    for idx, item in enumerate(structured_source[:15], 1):
        fts = item["expected_focus_terms"]
        focus_phrase = _natural_focus_phrase(fts, title=item.get("title", ""))
        if idx % 3 == 1:
            question = f"Trong phap luat hien hanh, {focus_phrase} duoc quy dinh nhu the nao?"
        elif idx % 3 == 2:
            question = f"Quy dinh chung ve {focus_phrase} la gi?"
        else:
            question = f"Khi ap dung thuc te, {focus_phrase} can luu y nhung gi?"

        case = {
            "case_id": f"v4_hybrid_default_{idx:02d}",
            "question": question,
            "expected_doc_id": item["expected_doc_id"],
            "expected_article_ref": item["expected_article_ref"],
            "expected_keywords": fts[:3],
            "expected_route": "hybrid_default",
            "expected_focus_terms": fts[:3],
            "expected_topical_terms": fts[:3],
            "expected_doc_number": item["expected_doc_number"],
            "expected_article_num": item["expected_article_num"],
            "expected_clause_num": item["expected_clause_num"],
            "notes": f"naturalized_from_{item['source_case_id']}",
            "gold_answer": None,
        }
        if _expected_route_of_query(question) == "hybrid_default":
            cases_hybrid_default.append(case)

    fallback_hybrid_questions = [
        "Dieu kien tuyen dung cong chuc hien nay la gi?",
        "Nguoi lao dong duoc don phuong cham dut hop dong khi nao?",
        "Truong hop nao cong chuc bi xu ly ky luat?",
        "Tham quyen xu phat vi pham hanh chinh duoc quy dinh ra sao?",
        "Quyen va nghia vu co ban cua cong chuc la gi?",
        "Nguyen tac quan ly nha nuoc trong linh vuc lao dong la gi?",
        "Thu tuc giai quyet khieu nai hanh chinh gom nhung buoc nao?",
        "Cac truong hop duoc huong tro cap that nghiep la gi?",
        "Dieu kien bo nhiem cong chuc lanh dao duoc quy dinh the nao?",
        "Nguoi su dung lao dong co trach nhiem gi ve an toan lao dong?",
    ]
    for idx, question in enumerate(fallback_hybrid_questions, 1):
        if len(cases_hybrid_default) >= 15:
            break
        if _expected_route_of_query(question) != "hybrid_default":
            continue

        src = structured_source[(idx - 1) % len(structured_source)]
        focus_terms = _extract_focus_terms(question, src.get("title", ""))
        if not focus_terms:
            focus_terms = list(src.get("expected_focus_terms", []))[:3]

        case = {
            "case_id": "v4_hybrid_default_tmp",
            "question": question,
            "expected_doc_id": src["expected_doc_id"],
            "expected_article_ref": src["expected_article_ref"],
            "expected_keywords": focus_terms[:3],
            "expected_route": "hybrid_default",
            "expected_focus_terms": focus_terms[:3],
            "expected_topical_terms": focus_terms[:3],
            "expected_doc_number": src["expected_doc_number"],
            "expected_article_num": src["expected_article_num"],
            "expected_clause_num": src["expected_clause_num"],
            "notes": "fallback_hybrid_natural_query",
            "gold_answer": None,
        }
        cases_hybrid_default.append(case)

    for idx, case in enumerate(cases_hybrid_default, 1):
        case["case_id"] = f"v4_hybrid_default_{idx:02d}"
    cases_hybrid_default = cases_hybrid_default[:15]

    expanded = (
        cases_structured[:20]
        + cases_narrow[:15]
        + cases_doc_scoped_exact[:6]
        + cases_doc_scoped_candidate[:4]
        + cases_law_loose[:5]
        + cases_hybrid_default[:15]
    )
    return expanded


def _sync_eval_cases_to_db(db_path: Path, cases: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_cases (
            case_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            expected_doc_id TEXT,
            expected_article_ref TEXT,
            expected_keywords TEXT,
            expected_route TEXT,
            expected_focus_terms TEXT,
            expected_topical_terms TEXT,
            expected_doc_number TEXT,
            expected_article_num TEXT,
            expected_clause_num TEXT,
            notes TEXT,
            gold_answer TEXT
        )
        """
    )

    existing_cols = {
        row[1]
        for row in cur.execute("PRAGMA table_info(eval_cases)").fetchall()
    }
    for col_name in (
        "expected_route",
        "expected_focus_terms",
        "expected_topical_terms",
        "expected_doc_number",
        "expected_article_num",
        "expected_clause_num",
        "notes",
    ):
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE eval_cases ADD COLUMN {col_name} TEXT")

    rows: list[tuple[Any, ...]] = []
    for case in cases:
        rows.append(
            (
                case.get("case_id"),
                case.get("question", ""),
                case.get("expected_doc_id"),
                case.get("expected_article_ref"),
                json.dumps(case.get("expected_keywords", []), ensure_ascii=False),
                case.get("expected_route"),
                json.dumps(case.get("expected_focus_terms", []), ensure_ascii=False),
                json.dumps(case.get("expected_topical_terms", []), ensure_ascii=False),
                case.get("expected_doc_number"),
                case.get("expected_article_num"),
                case.get("expected_clause_num"),
                case.get("notes"),
                case.get("gold_answer"),
            )
        )

    cur.executemany(
        """
        INSERT INTO eval_cases (
            case_id, question, expected_doc_id, expected_article_ref, expected_keywords,
            expected_route, expected_focus_terms, expected_topical_terms, expected_doc_number, expected_article_num,
            expected_clause_num, notes, gold_answer
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            question=excluded.question,
            expected_doc_id=excluded.expected_doc_id,
            expected_article_ref=excluded.expected_article_ref,
            expected_keywords=excluded.expected_keywords,
            expected_route=excluded.expected_route,
            expected_focus_terms=excluded.expected_focus_terms,
            expected_topical_terms=excluded.expected_topical_terms,
            expected_doc_number=excluded.expected_doc_number,
            expected_article_num=excluded.expected_article_num,
            expected_clause_num=excluded.expected_clause_num,
            notes=excluded.notes,
            gold_answer=excluded.gold_answer
        """,
        rows,
    )

    conn.commit()
    conn.close()


def _normalize_doc_number_for_match(text: str) -> str:
    return normalize_doc_number(str(text or ""))


def _hit_expected_doc(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    expected_doc_id = _norm_text(case.get("expected_doc_id", ""))
    expected_doc_number = _normalize_doc_number_for_match(case.get("expected_doc_number", ""))

    hit_doc_id = _norm_text(hit.get("doc_id", ""))
    hit_doc_number = _normalize_doc_number_for_match(hit.get("document_number", ""))
    hay = _norm_text(" ".join([
        str(hit.get("doc_id", "")),
        str(hit.get("document_number", "")),
        str(hit.get("title", "")),
        str(hit.get("path", "")),
    ]))
    hay_doc_norm = _normalize_doc_number_for_match(hay)

    if expected_doc_id and expected_doc_id == hit_doc_id:
        return True
    if expected_doc_number:
        if hit_doc_number == expected_doc_number:
            return True
        if expected_doc_number in hay_doc_norm:
            return True
    return False


def _hit_expected_family(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    expected_key = _expected_family_key(case)
    if not expected_key:
        return False
    hit_key = _hit_family_key(hit)
    return _family_match(expected_key, hit_key)


def _parse_hit_article_clause(hit: dict[str, Any]) -> tuple[str, str]:
    article = _extract_num_token(hit.get("article", ""))
    clause = _extract_num_token(hit.get("clause", ""))
    path = str(hit.get("path", ""))

    if not article:
        m = re.search(r"\b(?:dieu)\s+(\d+[a-z]?)\b", _to_ascii(path).lower())
        if m:
            article = m.group(1)
    if not clause:
        m = re.search(r"\b(?:khoan)\s+(\d+[a-z]?)\b", _to_ascii(path).lower())
        if m:
            clause = m.group(1)
    return article, clause


def _hit_expected_article(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    exp_article = _norm_text(case.get("expected_article_num", ""))
    exp_clause = _norm_text(case.get("expected_clause_num", ""))
    if not exp_article:
        return False
    hit_article, hit_clause = _parse_hit_article_clause(hit)
    if _norm_text(hit_article) != exp_article:
        return False
    if exp_clause and _norm_text(hit_clause) != exp_clause:
        return False
    return True


def _case_focus_terms(case: dict[str, Any]) -> list[str]:
    raw = case.get("expected_focus_terms") or case.get("expected_topical_terms") or []
    return [str(t).lower() for t in raw if str(t).strip()]


def _hit_focus_terms(hit: dict[str, Any], case: dict[str, Any], *, min_match: int = 2) -> bool:
    focus_terms = _case_focus_terms(case)
    if not focus_terms:
        return False

    hay = " ".join([
        str(hit.get("title", "")),
        str(hit.get("path", "")),
        str(hit.get("article", "")),
        str(hit.get("clause", "")),
        str(hit.get("text", ""))[:1200],
    ])
    tokens = set(_tokenize_ascii(hay))
    hit_count = sum(1 for term in focus_terms if term in tokens)
    need = min_match if len(focus_terms) >= min_match else 1
    return hit_count >= need


def _evaluate_hit_quality(case: dict[str, Any], reranked_top10: list[dict[str, Any]]) -> dict[str, Any]:
    expected_route = str(case.get("expected_route", ""))
    rank_doc: int | None = None
    rank_family: int | None = None
    rank_article: int | None = None
    rank_focus: int | None = None
    rank_family_focus: int | None = None
    rank_topical: int | None = None

    for rank, hit in enumerate(reranked_top10, 1):
        doc_match = _hit_expected_doc(hit, case)
        family_match = _hit_expected_family(hit, case)
        if rank_doc is None and doc_match:
            rank_doc = rank
        if rank_family is None and family_match:
            rank_family = rank

        if rank_article is None and doc_match and _hit_expected_article(hit, case):
            rank_article = rank

        if rank_topical is None and _hit_focus_terms(hit, case, min_match=1):
            rank_topical = rank

        focus_match = _hit_focus_terms(hit, case, min_match=2)
        if rank_focus is None and focus_match:
            if expected_route == "hybrid_default":
                rank_focus = rank
            elif case.get("expected_doc_id") or case.get("expected_doc_number"):
                if doc_match:
                    rank_focus = rank
            else:
                rank_focus = rank
        if rank_family_focus is None:
            if _hit_expected_article(hit, case):
                if family_match:
                    rank_family_focus = rank
            elif focus_match and family_match:
                rank_family_focus = rank

    has_article_expectation = bool(case.get("expected_article_num"))
    if rank_doc is not None and rank_family is None:
        rank_family = rank_doc

    if expected_route == "hybrid_default":
        correct_article = int(rank_topical is not None)
        focus_top1 = int(rank_focus == 1)
    elif has_article_expectation:
        correct_article = int(rank_article is not None)
        focus_top1 = int(rank_article == 1)
    elif _case_focus_terms(case):
        correct_article = int(rank_focus is not None)
        focus_top1 = int(rank_focus == 1)
    else:
        correct_article = int(rank_doc is not None)
        focus_top1 = int(rank_doc == 1)

    focus_top3 = int(rank_focus is not None and rank_focus <= 3)
    focus_top1_within_family = int(rank_family_focus == 1)
    topical_doc_top10 = int(rank_topical is not None)
    law_found_top10 = int(rank_doc is not None)
    correct_article_top10_given_law_found = int(correct_article) if law_found_top10 else 0
    focus_top1_given_law_found = int(focus_top1) if law_found_top10 else 0
    top1 = reranked_top10[0] if reranked_top10 else {}
    answer_grounded = int(
        bool(
            top1
            and (
                str(top1.get("doc_id", "")).strip()
                or str(top1.get("document_number", "")).strip()
            )
            and (
                str(top1.get("article", "")).strip()
                or str(top1.get("path", "")).strip()
            )
        )
    )

    return {
        "rank_first_law": rank_doc,
        "rank_first_family": rank_family,
        "rank_first_article": rank_article,
        "rank_first_focus": rank_focus,
        "rank_first_focus_within_family": rank_family_focus,
        "rank_first_topical": rank_topical,
        "exact_doc_top10": int(rank_doc is not None),
        "correct_law_top10": int(rank_doc is not None),
        "correct_family_top10": int((rank_family is not None) or (rank_doc is not None)),
        "law_found_top10": int(law_found_top10),
        "correct_article_top10": int(correct_article),
        "correct_article_top10_given_law_found": int(correct_article_top10_given_law_found),
        "focus_top1": int(focus_top1),
        "focus_top1_given_law_found": int(focus_top1_given_law_found),
        "focus_top3": int(focus_top3),
        "focus_top1_within_family": int(focus_top1_within_family),
        "topical_doc_top10": int(topical_doc_top10),
        "answer_grounded": int(answer_grounded),
    }


def _classify_failure(case: dict[str, Any], row: dict[str, Any], routing: dict[str, Any]) -> tuple[list[str], str]:
    labels: list[str] = []
    expected_route = str(case.get("expected_route", ""))
    actual_route = str(row.get("actual_route", ""))
    legal_refs = dict(routing.get("legal_refs", {}))
    query_too_vague = bool(row.get("query_too_vague", False))
    exact_law_ok = int(row.get("correct_law_top10", 0)) == 1
    family_ok = int(row.get("correct_family_top10", 0)) == 1

    route_alias_tie_acceptable = (
        expected_route == "doc_scoped_hybrid_exact_doc"
        and actual_route == "doc_scoped_hybrid_candidate_docs"
        and exact_law_ok
        and str(row.get("document_lookup_selected_reason", "")) == "candidate_low_margin_or_tie"
    )
    route_scope_promotion_acceptable = (
        expected_route == "law_anchored_hybrid_loose"
        and actual_route == "doc_scoped_hybrid_candidate_docs"
        and (exact_law_ok or family_ok)
        and int(row.get("allowed_doc_ids_count", 0)) >= 1
    )
    if expected_route and actual_route != expected_route:
        if route_alias_tie_acceptable:
            labels.append("ROUTE_ACCEPTABLE_ALIAS_TIE")
        elif route_scope_promotion_acceptable:
            labels.append("ROUTE_ACCEPTABLE_SCOPE_PROMOTION")
        else:
            labels.append("ROUTE_WRONG")

    if bool(row.get("scoped_filter_broken", False)):
        labels.append("SCOPED_FILTER_BROKEN")

    if expected_route in {"structured_exact", "narrow_bm25"}:
        if case.get("expected_doc_number") and not (
            legal_refs.get("document_number")
            or legal_refs.get("document_loose")
            or legal_refs.get("document_short")
        ):
            labels.append("PARSER_DOC_NUMBER_FAIL")

    if expected_route in {
        "doc_scoped_hybrid_exact_doc",
        "doc_scoped_hybrid_candidate_docs",
        "law_anchored_hybrid_loose",
        "law_anchored_hybrid",
    } and not legal_refs.get("law_name"):
        labels.append("PARSER_LAW_NAME_FAIL")

    if expected_route in {"doc_scoped_hybrid_exact_doc", "doc_scoped_hybrid_candidate_docs"}:
        if actual_route == "law_anchored_hybrid_loose":
            labels.append("DOC_LOOKUP_CONFIDENCE_TOO_LOW")
        if actual_route.startswith("doc_scoped_hybrid") and not exact_law_ok and not family_ok:
            labels.append("DOC_LOOKUP_SCOPE_WRONG")
        elif actual_route.startswith("doc_scoped_hybrid") and not exact_law_ok and family_ok:
            labels.append("DOC_FAMILY_EQUIVALENT")

    if expected_route == "hybrid_default":
        if query_too_vague:
            labels.append("QUERY_TOO_VAGUE")
        if int(row.get("topical_doc_top10", 0)) == 1 and not exact_law_ok:
            if family_ok:
                labels.append("TOPICAL_FAMILY_EQUIVALENT")
            else:
                labels.append("TOPICAL_RETRIEVAL_OK_BUT_NOT_EXACT")
        elif int(row.get("topical_doc_top10", 0)) == 0 and int(row.get("focus_top3", 0)) == 0:
            labels.append("BM25_NOT_FOCUSED")
        if int(row.get("focus_top3", 0)) == 1 and int(row.get("focus_top1", 0)) == 0:
            labels.append("RERANK_BAD_TOP1")
    else:
        if not exact_law_ok:
            if expected_route in SOFT_FAMILY_EQUIV_ROUTES and family_ok:
                labels.append("DOC_FAMILY_EQUIVALENT")
            elif expected_route == "structured_exact":
                labels.append("LOOKUP_DOC_MISS")
            else:
                labels.append("BM25_NOT_FOCUSED")
        if exact_law_ok:
            if case.get("expected_article_num") and int(row.get("correct_article_top10", 0)) == 0:
                if expected_route == "structured_exact":
                    labels.append("LOOKUP_ARTICLE_MISS")
                else:
                    labels.append("BM25_NOT_FOCUSED")

            if int(row.get("correct_article_top10", 0)) == 1 and int(row.get("focus_top1", 0)) == 0:
                labels.append("RERANK_BAD_TOP1")

    if (
        int(row.get("focus_top1", 0)) == 0
        and not bool(row.get("vector_skipped_by_route", False))
        and not bool(row.get("vector_skipped_by_quality_gate", False))
        and not bool(row.get("vector_timed_out", False))
        and not bool(row.get("vector_deadline_hit", False))
        and exact_law_ok
        and expected_route != "hybrid_default"
    ):
        labels.append("VECTOR_NOISE")

    unique_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique_labels.append(label)

    if not unique_labels:
        return ["PASS"], "PASS"
    return unique_labels, unique_labels[0]


def _build_route_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_out: dict[str, Any] = {}
    for route in EXPECTED_ROUTES:
        route_rows = [r for r in rows if str(r.get("actual_route", "")) == route]
        if not route_rows:
            route_out[route] = {
                "count": 0,
                "latency_ms": {
                    "bm25": _summary_stats([]),
                    "vector": _summary_stats([]),
                    "vector_wall": _summary_stats([]),
                    "vector_join_wait": _summary_stats([]),
                    "retrieval": _summary_stats([]),
                },
                "quality": {
                    "exact_doc_top10_rate": 0.0,
                    "correct_law_top10_rate": 0.0,
                    "correct_family_top10_rate": 0.0,
                    "correct_article_top10_rate": 0.0,
                    "law_found_top10_rate": 0.0,
                    "correct_article_top10_given_law_found_rate": 0.0,
                    "focus_top1_given_law_found_rate": 0.0,
                    "focus_top1_rate": 0.0,
                    "focus_top3_rate": 0.0,
                    "focus_top1_within_family_rate": 0.0,
                    "topical_doc_top10_rate": 0.0,
                    "answer_grounded_rate": 0.0,
                    "correct_article_top10_rate_on_article_cases": 0.0,
                    "focus_top1_rate_on_focus_cases": 0.0,
                    "focus_top3_rate_on_focus_cases": 0.0,
                    "hybrid_default_scope_fallback_added_law10_rate": 0.0,
                    "hybrid_default_scope_fallback_added_article10_rate": 0.0,
                    "hybrid_default_scope_fallback_added_focus1_rate": 0.0,
                },
                "stability": {
                    "vector_timed_out_rate": 0.0,
                    "vector_deadline_hit_rate": 0.0,
                    "vector_result_used_rate": 0.0,
                    "vector_cancelled_rate": 0.0,
                    "vector_skipped_by_route_rate": 0.0,
                    "vector_skipped_by_quality_gate_rate": 0.0,
                    "scoped_filter_broken_rate": 0.0,
                    "hybrid_default_scope_fallback_used_rate": 0.0,
                    "hybrid_default_scope_fallback_hits_avg": 0.0,
                    "hybrid_default_scope_fallback_bm25_only_used_rate": 0.0,
                    "hybrid_default_scope_fallback_bm25_only_hits_avg": 0.0,
                    "hybrid_default_scope_fallback_bm25_only_latency_avg": 0.0,
                    "hybrid_default_scope_fallback_vector_reused_rate": 0.0,
                    "hybrid_default_scope_fallback_second_vector_called_rate": 0.0,
                    "hybrid_doc_gate_applied_rate": 0.0,
                    "hybrid_doc_gate_removed_bm25_avg": 0.0,
                    "hybrid_doc_gate_removed_vector_avg": 0.0,
                    "hybrid_anchor_scope_used_rate": 0.0,
                },
                "route_quality": {
                    "route_correct_rate": 0.0,
                    "route_semantic_correct_rate": 0.0,
                    "expected_route_distribution": {},
                },
                "failure_labels": {},
                "failure_labels_true": {},
                "benchmark_semantics": {
                    "acceptable_route_promotion_count": 0,
                    "acceptable_alias_tie_count": 0,
                    "acceptable_scope_promotion_count": 0,
                    "true_retrieval_failure_count": 0,
                    "true_focus_failure_count": 0,
                },
                "hybrid_default_stage_diagnostics": {
                    "weak_query_abort_used_rate": 0.0,
                    "garbage_family_rejected_count": 0,
                    "garbage_family_rejected_rate": 0.0,
                    "shortlist_after_reject_zero_count": 0,
                    "shortlist_after_reject_nonzero_count": 0,
                    "trusted_family_count": 0,
                    "untrusted_family_count": 0,
                    "trusted_family_rate": 0.0,
                    "untrusted_family_rate": 0.0,
                },
            }
            continue

        bm25_vals = [int(r.get("bm25_latency_ms", 0)) for r in route_rows]
        vector_vals = [int(r.get("vector_latency_ms", 0)) for r in route_rows]
        vector_wall_vals = [int(r.get("vector_wall_ms", 0)) for r in route_rows]
        vector_join_wait_vals = [int(r.get("vector_join_wait_ms", 0)) for r in route_rows]
        retrieval_vals = [int(r.get("retrieval_latency_ms", 0)) for r in route_rows]

        article_cases = [r for r in route_rows if str(r.get("expected_article_num", ""))]
        focus_cases = [
            r
            for r in route_rows
            if list(r.get("expected_focus_terms", []))
            or list(r.get("expected_topical_terms", []))
        ]
        law_found_rows = [r for r in route_rows if int(r.get("law_found_top10", 0)) == 1]

        expected_dist = Counter(str(r.get("expected_route", "unknown")) for r in route_rows)
        failure_dist = Counter(str(r.get("primary_failure_label", "PASS")) for r in route_rows)
        failure_true_dist = Counter(str(r.get("primary_true_failure_label", "PASS")) for r in route_rows)
        is_hybrid_default = route == "hybrid_default"

        route_out[route] = {
            "count": len(route_rows),
            "latency_ms": {
                "bm25": _summary_stats(bm25_vals),
                "vector": _summary_stats(vector_vals),
                "vector_wall": _summary_stats(vector_wall_vals),
                "vector_join_wait": _summary_stats(vector_join_wait_vals),
                "retrieval": _summary_stats(retrieval_vals),
            },
            "quality": {
                "exact_doc_top10_rate": round(
                    sum(int(r.get("exact_doc_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "correct_law_top10_rate": round(
                    sum(int(r.get("correct_law_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "correct_family_top10_rate": round(
                    sum(int(r.get("correct_family_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "correct_article_top10_rate": round(
                    sum(int(r.get("correct_article_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "law_found_top10_rate": round(
                    sum(int(r.get("law_found_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "correct_article_top10_given_law_found_rate": round(
                    (
                        sum(int(r.get("correct_article_top10", 0)) for r in law_found_rows)
                        / len(law_found_rows)
                        if law_found_rows
                        else 0.0
                    ),
                    4,
                ),
                "focus_top1_given_law_found_rate": round(
                    (
                        sum(int(r.get("focus_top1", 0)) for r in law_found_rows)
                        / len(law_found_rows)
                        if law_found_rows
                        else 0.0
                    ),
                    4,
                ),
                "focus_top1_rate": round(
                    sum(int(r.get("focus_top1", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "focus_top3_rate": round(
                    sum(int(r.get("focus_top3", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "focus_top1_within_family_rate": round(
                    sum(int(r.get("focus_top1_within_family", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "topical_doc_top10_rate": round(
                    sum(int(r.get("topical_doc_top10", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "answer_grounded_rate": round(
                    sum(int(r.get("answer_grounded", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "correct_article_top10_rate_on_article_cases": round(
                    (
                        sum(int(r.get("correct_article_top10", 0)) for r in article_cases) / len(article_cases)
                        if article_cases
                        else 0.0
                    ),
                    4,
                ),
                "focus_top1_rate_on_focus_cases": round(
                    (
                        sum(int(r.get("focus_top1", 0)) for r in focus_cases) / len(focus_cases)
                        if focus_cases
                        else 0.0
                    ),
                    4,
                ),
                "focus_top3_rate_on_focus_cases": round(
                    (
                        sum(int(r.get("focus_top3", 0)) for r in focus_cases) / len(focus_cases)
                        if focus_cases
                        else 0.0
                    ),
                    4,
                ),
                "hybrid_default_scope_fallback_added_law10_rate": round(
                    sum(int(r.get("hybrid_default_scope_fallback_added_law10", 0)) for r in route_rows)
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_added_article10_rate": round(
                    sum(int(r.get("hybrid_default_scope_fallback_added_article10", 0)) for r in route_rows)
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_added_focus1_rate": round(
                    sum(int(r.get("hybrid_default_scope_fallback_added_focus1", 0)) for r in route_rows)
                    / len(route_rows),
                    4,
                ),
            },
            "stability": {
                "vector_timed_out_rate": round(
                    sum(int(bool(r.get("vector_timed_out", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "vector_deadline_hit_rate": round(
                    sum(int(bool(r.get("vector_deadline_hit", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "vector_result_used_rate": round(
                    sum(int(bool(r.get("vector_result_used", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "vector_cancelled_rate": round(
                    sum(int(bool(r.get("vector_cancelled", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "vector_skipped_by_route_rate": round(
                    sum(int(bool(r.get("vector_skipped_by_route", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "vector_skipped_by_quality_gate_rate": round(
                    sum(int(bool(r.get("vector_skipped_by_quality_gate", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "scoped_filter_broken_rate": round(
                    sum(int(bool(r.get("scoped_filter_broken", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_used_rate": round(
                    sum(int(bool(r.get("hybrid_default_scope_fallback_used", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_hits_avg": round(
                    sum(int(r.get("hybrid_default_scope_fallback_hits", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_bm25_only_used_rate": round(
                    sum(
                        int(bool(r.get("hybrid_default_scope_fallback_bm25_only_used", False)))
                        for r in route_rows
                    )
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_bm25_only_hits_avg": round(
                    sum(int(r.get("hybrid_default_scope_fallback_bm25_only_hits", 0)) for r in route_rows)
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_bm25_only_latency_avg": round(
                    sum(
                        int(r.get("hybrid_default_scope_fallback_bm25_only_latency_ms", 0))
                        for r in route_rows
                    )
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_vector_reused_rate": round(
                    sum(
                        int(bool(r.get("hybrid_default_scope_fallback_vector_reused", False)))
                        for r in route_rows
                    )
                    / len(route_rows),
                    4,
                ),
                "hybrid_default_scope_fallback_second_vector_called_rate": round(
                    sum(
                        int(bool(r.get("hybrid_default_scope_fallback_second_vector_called", False)))
                        for r in route_rows
                    )
                    / len(route_rows),
                    4,
                ),
                "hybrid_doc_gate_applied_rate": round(
                    sum(int(bool(r.get("hybrid_doc_gate_applied", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_doc_gate_removed_bm25_avg": round(
                    sum(int(r.get("hybrid_doc_gate_removed_bm25", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_doc_gate_removed_vector_avg": round(
                    sum(int(r.get("hybrid_doc_gate_removed_vector", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "hybrid_anchor_scope_used_rate": round(
                    sum(int(bool(r.get("hybrid_anchor_scope_used", False))) for r in route_rows) / len(route_rows),
                    4,
                ),
            },
            "route_quality": {
                "route_correct_rate": round(
                    sum(int(r.get("route_correct", 0)) for r in route_rows) / len(route_rows),
                    4,
                ),
                "route_semantic_correct_rate": round(
                    sum(int(r.get("route_semantic_correct", r.get("route_correct", 0))) for r in route_rows) / len(route_rows),
                    4,
                ),
                "expected_route_distribution": dict(expected_dist),
            },
            "failure_labels": dict(failure_dist),
            "failure_labels_true": dict(failure_true_dist),
            "benchmark_semantics": {
                "acceptable_route_promotion_count": int(
                    sum(int(r.get("acceptable_route_promotion", 0)) for r in route_rows)
                ),
                "acceptable_alias_tie_count": int(
                    sum(int(r.get("acceptable_alias_tie", 0)) for r in route_rows)
                ),
                "acceptable_scope_promotion_count": int(
                    sum(int(r.get("acceptable_scope_promotion", 0)) for r in route_rows)
                ),
                "true_retrieval_failure_count": int(
                    sum(int(r.get("is_true_failure", 0)) for r in route_rows)
                ),
                "true_focus_failure_count": int(
                    sum(int(r.get("is_true_focus_failure", 0)) for r in route_rows)
                ),
            },
            "hybrid_default_stage_diagnostics": {
                "weak_query_abort_used_rate": (
                    round(
                        sum(int(bool(r.get("hybrid_default_weak_query_abort_used", False))) for r in route_rows)
                        / len(route_rows),
                        4,
                    )
                    if is_hybrid_default
                    else 0.0
                ),
                "garbage_family_rejected_count": (
                    int(
                        sum(int(bool(r.get("hybrid_default_garbage_family_rejected", False))) for r in route_rows)
                    )
                    if is_hybrid_default
                    else 0
                ),
                "garbage_family_rejected_rate": (
                    round(
                        sum(int(bool(r.get("hybrid_default_garbage_family_rejected", False))) for r in route_rows)
                        / len(route_rows),
                        4,
                    )
                    if is_hybrid_default
                    else 0.0
                ),
                "shortlist_after_reject_zero_count": (
                    int(
                        sum(
                            int(bool(r.get("hybrid_default_shortlist_after_reject_empty", False)))
                            for r in route_rows
                        )
                    )
                    if is_hybrid_default
                    else 0
                ),
                "shortlist_after_reject_nonzero_count": (
                    int(
                        sum(
                            int(not bool(r.get("hybrid_default_shortlist_after_reject_empty", False)))
                            for r in route_rows
                        )
                    )
                    if is_hybrid_default
                    else 0
                ),
                "trusted_family_count": (
                    int(
                        sum(int(bool(r.get("hybrid_default_trusted_family", False))) for r in route_rows)
                    )
                    if is_hybrid_default
                    else 0
                ),
                "untrusted_family_count": (
                    int(
                        sum(int(not bool(r.get("hybrid_default_trusted_family", False))) for r in route_rows)
                    )
                    if is_hybrid_default
                    else 0
                ),
                "trusted_family_rate": (
                    round(
                        sum(int(bool(r.get("hybrid_default_trusted_family", False))) for r in route_rows)
                        / len(route_rows),
                        4,
                    )
                    if is_hybrid_default
                    else 0.0
                ),
                "untrusted_family_rate": (
                    round(
                        sum(int(not bool(r.get("hybrid_default_trusted_family", False))) for r in route_rows)
                        / len(route_rows),
                        4,
                    )
                    if is_hybrid_default
                    else 0.0
                ),
            },
        }
    return route_out


def _build_run_summary(rows: list[dict[str, Any]], elapsed_ms: int) -> dict[str, Any]:
    bm25_vals = [int(r.get("bm25_latency_ms", 0)) for r in rows]
    vector_vals = [int(r.get("vector_latency_ms", 0)) for r in rows]
    vector_wall_vals = [int(r.get("vector_wall_ms", 0)) for r in rows]
    vector_join_wait_vals = [int(r.get("vector_join_wait_ms", 0)) for r in rows]
    retrieval_vals = [int(r.get("retrieval_latency_ms", 0)) for r in rows]
    failure_dist = Counter(str(r.get("primary_failure_label", "PASS")) for r in rows)
    failure_true_dist = Counter(str(r.get("primary_true_failure_label", "PASS")) for r in rows)
    law_found_rows = [r for r in rows if int(r.get("law_found_top10", 0)) == 1]
    route_summary = _build_route_summary(rows)
    hybrid_default_summary = (route_summary.get("hybrid_default") or {})
    hybrid_default_quality = (hybrid_default_summary.get("quality") or {})
    hybrid_default_latency = ((hybrid_default_summary.get("latency_ms") or {}).get("retrieval") or {})
    acceptance_thresholds_v14 = {
        "correct_law_top10_rate": 0.40,
        "law_found_top10_rate": 0.50,
        "topical_doc_top10_rate": 0.50,
        "focus_top1_rate": 0.20,
        "answer_grounded_rate": 0.75,
        "p95_retrieval_ms": 5000,
    }
    acceptance_checks_v14 = {
        "correct_law_top10_rate": float(hybrid_default_quality.get("correct_law_top10_rate", 0.0)),
        "law_found_top10_rate": float(hybrid_default_quality.get("law_found_top10_rate", 0.0)),
        "topical_doc_top10_rate": float(hybrid_default_quality.get("topical_doc_top10_rate", 0.0)),
        "focus_top1_rate": float(hybrid_default_quality.get("focus_top1_rate", 0.0)),
        "answer_grounded_rate": float(hybrid_default_quality.get("answer_grounded_rate", 0.0)),
        "p95_retrieval_ms": float(hybrid_default_latency.get("p95", 0)),
    }
    acceptance_pass_v14 = bool(
        acceptance_checks_v14["correct_law_top10_rate"] >= acceptance_thresholds_v14["correct_law_top10_rate"]
        and acceptance_checks_v14["law_found_top10_rate"] >= acceptance_thresholds_v14["law_found_top10_rate"]
        and acceptance_checks_v14["topical_doc_top10_rate"] >= acceptance_thresholds_v14["topical_doc_top10_rate"]
        and acceptance_checks_v14["focus_top1_rate"] >= acceptance_thresholds_v14["focus_top1_rate"]
        and acceptance_checks_v14["answer_grounded_rate"] >= acceptance_thresholds_v14["answer_grounded_rate"]
        and acceptance_checks_v14["p95_retrieval_ms"] <= acceptance_thresholds_v14["p95_retrieval_ms"]
    )

    hybrid_default_rows = [r for r in rows if str(r.get("actual_route", "")) == "hybrid_default"]
    hybrid_default_count = len(hybrid_default_rows)
    mode_counter = Counter(str(r.get("hybrid_default_mode", "")) for r in hybrid_default_rows)
    vague_share = (
        float(mode_counter.get("vague_concept_recovery", 0)) / hybrid_default_count
        if hybrid_default_count
        else 0.0
    )
    garbage_rejected_count = sum(
        int(bool(r.get("hybrid_default_garbage_family_rejected", False)))
        for r in hybrid_default_rows
    )
    garbage_rejected_rate = (
        float(garbage_rejected_count) / hybrid_default_count
        if hybrid_default_count
        else 0.0
    )
    trusted_family_count = sum(
        int(bool(r.get("hybrid_default_trusted_family", False)))
        for r in hybrid_default_rows
    )
    trusted_family_rate = (
        float(trusted_family_count) / hybrid_default_count
        if hybrid_default_count
        else 0.0
    )
    v14_baseline_p95 = float(getattr(cfg, "HYBRID_DEFAULT_V14_BASELINE_P95_MS", 8931))
    acceptance_thresholds_v15 = {
        "correct_law_top10_rate": 0.0001,
        "law_found_top10_rate": 0.0001,
        "p95_retrieval_ms_max": round(v14_baseline_p95 * 1.10, 2),
        "vague_concept_recovery_share_max": 0.70,
        "garbage_family_rejected_rate_max": 0.50,
    }
    acceptance_checks_v15 = {
        "correct_law_top10_rate": float(hybrid_default_quality.get("correct_law_top10_rate", 0.0)),
        "law_found_top10_rate": float(hybrid_default_quality.get("law_found_top10_rate", 0.0)),
        "p95_retrieval_ms": float(hybrid_default_latency.get("p95", 0)),
        "vague_concept_recovery_share": float(round(vague_share, 4)),
        "garbage_family_rejected_rate": float(round(garbage_rejected_rate, 4)),
        "garbage_family_rejected_count": int(garbage_rejected_count),
        "trusted_family_rate": float(round(trusted_family_rate, 4)),
    }
    acceptance_pass_v15 = bool(
        acceptance_checks_v15["correct_law_top10_rate"] > acceptance_thresholds_v15["correct_law_top10_rate"]
        and acceptance_checks_v15["law_found_top10_rate"] > acceptance_thresholds_v15["law_found_top10_rate"]
        and acceptance_checks_v15["p95_retrieval_ms"] <= acceptance_thresholds_v15["p95_retrieval_ms_max"]
        and acceptance_checks_v15["vague_concept_recovery_share"] < acceptance_thresholds_v15["vague_concept_recovery_share_max"]
        and acceptance_checks_v15["garbage_family_rejected_rate"] <= acceptance_thresholds_v15["garbage_family_rejected_rate_max"]
    )

    return {
        "total_cases": len(rows),
        "elapsed_ms": int(elapsed_ms),
        "latency_ms": {
            "bm25": _summary_stats(bm25_vals),
            "vector": _summary_stats(vector_vals),
            "vector_wall": _summary_stats(vector_wall_vals),
            "vector_join_wait": _summary_stats(vector_join_wait_vals),
            "retrieval": _summary_stats(retrieval_vals),
        },
        "quality": {
            "route_correct_rate": round(
                sum(int(r.get("route_correct", 0)) for r in rows) / len(rows),
                4,
            ),
            "route_semantic_correct_rate": round(
                sum(int(r.get("route_semantic_correct", r.get("route_correct", 0))) for r in rows) / len(rows),
                4,
            ),
            "exact_doc_top10_rate": round(
                sum(int(r.get("exact_doc_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "correct_law_top10_rate": round(
                sum(int(r.get("correct_law_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "correct_family_top10_rate": round(
                sum(int(r.get("correct_family_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "correct_article_top10_rate": round(
                sum(int(r.get("correct_article_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "law_found_top10_rate": round(
                sum(int(r.get("law_found_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "correct_article_top10_given_law_found_rate": round(
                (
                    sum(int(r.get("correct_article_top10", 0)) for r in law_found_rows)
                    / len(law_found_rows)
                    if law_found_rows
                    else 0.0
                ),
                4,
            ),
            "focus_top1_given_law_found_rate": round(
                (
                    sum(int(r.get("focus_top1", 0)) for r in law_found_rows)
                    / len(law_found_rows)
                    if law_found_rows
                    else 0.0
                ),
                4,
            ),
            "focus_top1_rate": round(
                sum(int(r.get("focus_top1", 0)) for r in rows) / len(rows),
                4,
            ),
            "focus_top3_rate": round(
                sum(int(r.get("focus_top3", 0)) for r in rows) / len(rows),
                4,
            ),
            "focus_top1_within_family_rate": round(
                sum(int(r.get("focus_top1_within_family", 0)) for r in rows) / len(rows),
                4,
            ),
            "topical_doc_top10_rate": round(
                sum(int(r.get("topical_doc_top10", 0)) for r in rows) / len(rows),
                4,
            ),
            "answer_grounded_rate": round(
                sum(int(r.get("answer_grounded", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_added_law10_rate": round(
                sum(int(r.get("hybrid_default_scope_fallback_added_law10", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_added_article10_rate": round(
                sum(int(r.get("hybrid_default_scope_fallback_added_article10", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_added_focus1_rate": round(
                sum(int(r.get("hybrid_default_scope_fallback_added_focus1", 0)) for r in rows) / len(rows),
                4,
            ),
        },
        "stability": {
            "vector_timed_out_rate": round(
                sum(int(bool(r.get("vector_timed_out", False))) for r in rows) / len(rows),
                4,
            ),
            "vector_deadline_hit_rate": round(
                sum(int(bool(r.get("vector_deadline_hit", False))) for r in rows) / len(rows),
                4,
            ),
            "vector_result_used_rate": round(
                sum(int(bool(r.get("vector_result_used", False))) for r in rows) / len(rows),
                4,
            ),
            "vector_cancelled_rate": round(
                sum(int(bool(r.get("vector_cancelled", False))) for r in rows) / len(rows),
                4,
            ),
            "vector_skipped_by_route_rate": round(
                sum(int(bool(r.get("vector_skipped_by_route", False))) for r in rows) / len(rows),
                4,
            ),
            "vector_skipped_by_quality_gate_rate": round(
                sum(int(bool(r.get("vector_skipped_by_quality_gate", False))) for r in rows) / len(rows),
                4,
            ),
            "scoped_filter_broken_rate": round(
                sum(int(bool(r.get("scoped_filter_broken", False))) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_used_rate": round(
                sum(int(bool(r.get("hybrid_default_scope_fallback_used", False))) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_hits_avg": round(
                sum(int(r.get("hybrid_default_scope_fallback_hits", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_bm25_only_used_rate": round(
                sum(
                    int(bool(r.get("hybrid_default_scope_fallback_bm25_only_used", False)))
                    for r in rows
                )
                / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_bm25_only_hits_avg": round(
                sum(int(r.get("hybrid_default_scope_fallback_bm25_only_hits", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_bm25_only_latency_avg": round(
                sum(int(r.get("hybrid_default_scope_fallback_bm25_only_latency_ms", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_vector_reused_rate": round(
                sum(
                    int(bool(r.get("hybrid_default_scope_fallback_vector_reused", False)))
                    for r in rows
                )
                / len(rows),
                4,
            ),
            "hybrid_default_scope_fallback_second_vector_called_rate": round(
                sum(
                    int(bool(r.get("hybrid_default_scope_fallback_second_vector_called", False)))
                    for r in rows
                )
                / len(rows),
                4,
            ),
            "hybrid_doc_gate_applied_rate": round(
                sum(int(bool(r.get("hybrid_doc_gate_applied", False))) for r in rows) / len(rows),
                4,
            ),
            "hybrid_doc_gate_removed_bm25_avg": round(
                sum(int(r.get("hybrid_doc_gate_removed_bm25", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_doc_gate_removed_vector_avg": round(
                sum(int(r.get("hybrid_doc_gate_removed_vector", 0)) for r in rows) / len(rows),
                4,
            ),
            "hybrid_anchor_scope_used_rate": round(
                sum(int(bool(r.get("hybrid_anchor_scope_used", False))) for r in rows) / len(rows),
                4,
            ),
        },
        "failure_labels": dict(failure_dist),
        "failure_labels_true": dict(failure_true_dist),
        "benchmark_semantics": {
            "acceptable_route_promotion_count": int(
                sum(int(r.get("acceptable_route_promotion", 0)) for r in rows)
            ),
            "acceptable_alias_tie_count": int(
                sum(int(r.get("acceptable_alias_tie", 0)) for r in rows)
            ),
            "acceptable_scope_promotion_count": int(
                sum(int(r.get("acceptable_scope_promotion", 0)) for r in rows)
            ),
            "true_retrieval_failure_count": int(
                sum(int(r.get("is_true_failure", 0)) for r in rows)
            ),
            "true_focus_failure_count": int(
                sum(int(r.get("is_true_focus_failure", 0)) for r in rows)
            ),
        },
        "route_summary": route_summary,
        "hybrid_default_acceptance_gate_v14": {
            "pass": acceptance_pass_v14,
            "thresholds": acceptance_thresholds_v14,
            "observed": acceptance_checks_v14,
        },
        "hybrid_default_acceptance_gate_v15": {
            "pass": acceptance_pass_v15,
            "thresholds": acceptance_thresholds_v15,
            "observed": acceptance_checks_v15,
            "mode_distribution": dict(mode_counter),
        },
    }


def _resolve_exact_baseline(path: str | None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.exists() else None

    v3_candidates = sorted(Path("logs").glob("benchmark_patch_v3_*.json"))
    for p in reversed(v3_candidates):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            routes = set((payload.get("route_summary") or {}).keys())
            if routes == {"structured_exact"}:
                return p
        except Exception:
            continue

    v2_candidates = sorted(Path("logs").glob("benchmark_patch_v2_numeric_final_*.json"))
    if v2_candidates:
        return v2_candidates[-1]
    return None


def _extract_exact_metrics(payload: dict[str, Any]) -> dict[str, float]:
    route_summary = payload.get("route_summary") or {}
    if "structured_exact" in route_summary:
        exact = route_summary["structured_exact"]
        latency_root = exact.get("latency_ms") or {}
        if isinstance(latency_root.get("retrieval"), dict):
            latency = latency_root.get("retrieval", {})
            p95 = float(latency.get("p95", 0))
        else:
            p95 = float(latency_root.get("p95", 0))
        quality = exact.get("quality") or {}
        return {
            "p95_retrieval_ms": p95,
            "correct_article_top10_rate": float(quality.get("correct_article_top10_rate", 0.0)),
            "focus_top1_rate": float(quality.get("focus_top1_rate", 0.0)),
        }

    summary = payload.get("summary") or {}
    latency = summary.get("latency_ms") or {}
    quality = summary.get("quality") or {}

    # Old format compatibility.
    retrieval_p95 = latency.get("retrieval_p95")
    if retrieval_p95 is None and isinstance(latency.get("retrieval"), dict):
        retrieval_p95 = latency["retrieval"].get("p95", 0)

    return {
        "p95_retrieval_ms": float(retrieval_p95 or 0),
        "correct_article_top10_rate": float(quality.get("correct_article_top10_rate", 0.0)),
        "focus_top1_rate": float(quality.get("focus_top1_rate", 0.0)),
    }


def _compare_metrics(current: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("p95_retrieval_ms", "correct_article_top10_rate", "focus_top1_rate"):
        c = float(current.get(key, 0.0))
        b = float(baseline.get(key, 0.0))
        out[key] = {
            "baseline": b,
            "current": c,
            "delta": round(c - b, 4),
        }
    return out


def _load_cases_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Cases file must be a JSON array: {path}")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(dict(item))
    return rows


def _format_top_hits(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in list(hits or [])[:top_k]:
        out.append(
            {
                "rank": int(hit.get("rank", 0)),
                "chunk_id": str(hit.get("chunk_id", "")),
                "doc_id": str(hit.get("doc_id", "")),
                "document_number": str(hit.get("document_number", "")),
                "article": str(hit.get("article", "")),
                "clause": str(hit.get("clause", "")),
                "title": str(hit.get("title", "")),
                "path": str(hit.get("path", "")),
                "score": float(hit.get("score", 0.0)),
                "rrf_score": float(hit.get("rrf_score", 0.0)),
                "rerank_score": float(hit.get("rerank_score", 0.0)),
            }
        )
    return out


def _top_hit_fields(hits: list[dict[str, Any]], idx: int) -> dict[str, str]:
    if idx < 1 or idx > len(hits):
        return {
            "chunk_id": "",
            "doc_id": "",
            "document_number": "",
            "article": "",
            "clause": "",
            "title": "",
            "path": "",
            "score": "",
            "rerank_score": "",
        }
    hit = hits[idx - 1]
    return {
        "chunk_id": str(hit.get("chunk_id", "")),
        "doc_id": str(hit.get("doc_id", "")),
        "document_number": str(hit.get("document_number", "")),
        "article": str(hit.get("article", "")),
        "clause": str(hit.get("clause", "")),
        "title": str(hit.get("title", "")),
        "path": str(hit.get("path", "")),
        "score": str(hit.get("score", "")),
        "rerank_score": str(hit.get("rerank_score", "")),
    }


def _join_seq(value: Any, sep: str = ";") -> str:
    if isinstance(value, (list, tuple)):
        return sep.join(str(v) for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value)


def _write_failure_outputs(
    *,
    warm_rows: list[dict[str, Any]],
    cold_rows: list[dict[str, Any]],
    failure_csv_out: Path,
    failure_debug_out: Path,
) -> tuple[int, int]:
    fail_rows = [
        row
        for row in (list(warm_rows) + list(cold_rows))
        if str(row.get("primary_failure_label", "")) not in {"", "PASS"}
    ]

    failure_csv_out.parent.mkdir(parents=True, exist_ok=True)
    failure_debug_out.parent.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "run",
        "case_id",
        "expected_route",
        "actual_route",
        "primary_failure_label",
        "failure_labels",
        "primary_true_failure_label",
        "true_failure_labels",
        "acceptable_route_promotion",
        "acceptable_alias_tie",
        "acceptable_scope_promotion",
        "is_true_failure",
        "is_true_focus_failure",
        "route_correct",
        "route_semantic_correct",
        "retrieval_latency_ms",
        "bm25_latency_ms",
        "vector_top_k_effective",
        "vector_latency_ms",
        "vector_wall_ms",
        "vector_budget_ms",
        "vector_join_wait_ms",
        "vector_timed_out",
        "vector_deadline_hit",
        "vector_result_used",
        "vector_cancelled",
        "vector_skipped_by_route",
        "vector_skipped_by_quality_gate",
        "bm25_scoped",
        "vector_scoped",
        "allowed_doc_ids_count",
        "scoped_filter_broken",
        "scoped_filter_broken_doc_ids",
        "correct_law_top10",
        "correct_family_top10",
        "law_found_top10",
        "correct_article_top10",
        "correct_article_top10_given_law_found",
        "focus_top1",
        "focus_top1_given_law_found",
        "focus_top1_within_family",
        "expected_doc_id",
        "expected_doc_number",
        "expected_family_key",
        "expected_article_num",
        "expected_clause_num",
        "expected_focus_terms",
        "expected_topical_terms",
        "parser_document_number",
        "parser_document_loose",
        "parser_document_short",
        "parser_article_number",
        "parser_clause_number",
        "parser_law_name",
        "bm25_mode",
        "bm25_top_k_effective",
        "bm25_tokenized_query",
        "bm25_parsed_query",
        "bm25_query_fields",
        "bm25_anchor_tokens",
        "bm25_intent_tokens",
        "bm25_is_multifield",
        "narrow_doc_number_raw",
        "narrow_doc_number_canonical",
        "narrow_doc_number_loose_key",
        "narrow_doc_number_exact_candidates_count",
        "narrow_doc_number_exact_candidates",
        "narrow_doc_number_exact_pass_used",
        "narrow_doc_number_exact_pass_hits",
        "narrow_doc_number_bias_applied",
        "narrow_doc_number_top1_match_level",
        "narrow_doc_number_mismatch_reason",
        "document_lookup_used",
        "document_lookup_ms",
        "document_lookup_matched",
        "document_lookup_match_type",
        "document_lookup_confidence",
        "document_lookup_doc_ids_count",
        "document_lookup_top1_confidence",
        "document_lookup_top2_confidence",
        "document_lookup_margin",
        "document_lookup_selected_route",
        "document_lookup_selected_reason",
        "query_rewrite_used",
        "query_rewrite_ms",
        "query_rewrite_confidence",
        "query_too_vague",
        "query_rewrite_clean",
        "query_rewrite_lexical",
        "query_rewrite_semantic",
        "query_rewrite_focus_terms",
        "query_rewrite_fillers_removed",
        "query_rewrite_phrase_repairs",
        "query_rewrite_legal_anchor_guess",
        "query_rewrite_legal_anchor_guess_list",
        "query_rewrite_doc_type_prior",
        "query_rewrite_exclude_doc_type_hint",
        "query_rewrite_topic_class",
        "query_rewrite_subclass",
        "query_rewrite_legal_concept_tags",
        "query_rewrite_actor_terms",
        "query_rewrite_action_terms",
        "query_rewrite_object_terms",
        "query_rewrite_qualifier_terms",
        "query_rewrite_vagueness_level",
        "query_rewrite_concept_confidence",
        "query_rewrite_v3_used",
        "query_rewrite_is_concept_query",
        "query_rewrite_is_topic_broad",
        "query_rewrite_lexical_core",
        "query_rewrite_lexical_expanded",
        "query_rewrite_concept_seed_query",
        "query_rewrite_title_anchor_query",
        "query_rewrite_risk",
        "query_rewrite_lexical_is_weak",
        "query_rewrite_weak_query_abort",
        "query_rewrite_weak_query_abort_reasons",
        "query_rewrite_lexical_quality_flags",
        "query_rewrite_intent_template_hits",
        "query_rewrite_lexical_expansion_used",
        "hybrid_default_mode",
        "hybrid_default_candidate_family_count",
        "hybrid_default_candidate_doc_count",
        "hybrid_default_title_anchor_used",
        "hybrid_default_title_anchor_hits",
        "hybrid_default_doc_aggregation_used",
        "hybrid_default_doc_score_top1",
        "hybrid_default_doc_score_top2",
        "hybrid_default_doc_score_margin",
        "hybrid_default_doc_top1_family",
        "hybrid_default_doc_top2_family",
        "hybrid_default_vector_used_for_doc_recall",
        "hybrid_default_vector_support_same_family",
        "hybrid_default_genericity_penalty_top1",
        "hybrid_default_genericity_penalty_applied_count",
        "hybrid_default_focus_rerank_used",
        "hybrid_default_focus_rerank_candidate_chunks",
        "hybrid_default_focus_heading_match_top1",
        "hybrid_default_focus_actor_action_top1",
        "hybrid_default_focus_concept_match_top1",
        "hybrid_default_selected_family_keys",
        "hybrid_default_selected_doc_ids",
        "hybrid_default_family_search_used",
        "hybrid_default_family_search_hits",
        "hybrid_default_family_candidate_count_pre_filter",
        "hybrid_default_family_candidate_count_post_filter",
        "hybrid_default_family_score_top1",
        "hybrid_default_family_score_top2",
        "hybrid_default_family_score_margin",
        "hybrid_default_family_identity_score_top1",
        "hybrid_default_family_identity_score_top2",
        "hybrid_default_family_identity_margin",
        "hybrid_default_family_top1_doc_role",
        "hybrid_default_family_top1_is_implementation",
        "hybrid_default_family_recovery_used",
        "hybrid_default_family_recovery_promoted",
        "hybrid_default_weak_query_abort_used",
        "hybrid_default_no_confident_family_candidates",
        "hybrid_default_garbage_family_rejected",
        "hybrid_default_garbage_family_reject_reason",
        "hybrid_default_trusted_family",
        "hybrid_default_shortlist_after_reject_count",
        "hybrid_default_shortlist_after_reject_empty",
        "hybrid_default_doc_role_prior_applied",
        "hybrid_default_doc_role_prior_top1",
        "hybrid_default_wrong_doc_role_penalty_applied",
        "hybrid_default_implementation_bridge_used",
        "hybrid_default_implementation_bridge_hits",
        "hybrid_default_focus_rerank_blocked_by_low_family_confidence",
        "hybrid_default_focus_rerank_stage_entered_with_family_confidence",
        "rank_first_family_pre_focus",
        "rank_first_family_post_focus",
        "rank_first_doc_role_pre_focus",
        "rank_first_doc_role_post_focus",
        "hybrid_default_scope_fallback_used",
        "hybrid_default_scope_fallback_reason",
        "hybrid_default_scope_fallback_hits",
        "hybrid_default_scope_fallback_bm25_only_used",
        "hybrid_default_scope_fallback_bm25_only_hits",
        "hybrid_default_scope_fallback_bm25_only_latency_ms",
        "hybrid_default_scope_fallback_bm25_only_query",
        "hybrid_default_scope_fallback_bm25_only_query_reason",
        "hybrid_default_scope_fallback_vector_reused",
        "hybrid_default_scope_fallback_second_vector_called",
        "hybrid_default_scope_fallback_first_pass_hits",
        "hybrid_default_scope_fallback_added_law10",
        "hybrid_default_scope_fallback_added_article10",
        "hybrid_default_scope_fallback_added_focus1",
        "hybrid_default_hard_doc_type_filter",
        "hybrid_default_hard_doc_type_allowlist",
        "hybrid_default_concept_policy_applied",
        "hybrid_default_concept_vector_optional",
        "hybrid_default_intra_doc_focus_rerank_used",
        "hybrid_default_intra_doc_focus_target_doc_ids",
        "hybrid_default_intra_doc_focus_target_family_keys",
        "hybrid_default_intra_doc_focus_rank_first_doc",
        "hybrid_default_intra_doc_focus_rank_first_family",
        "hybrid_default_intra_doc_focus_candidates",
        "hybrid_default_intra_doc_focus_promoted",
        "hybrid_anchor_scope_used",
        "hybrid_anchor_scope_confidence",
        "hybrid_anchor_scope_doc_ids_count",
        "hybrid_anchor_scope_doc_ids",
        "hybrid_anchor_scope_reason",
        "document_lookup_candidate_doc_ids_original_count",
        "document_lookup_candidate_doc_ids_trimmed",
        "candidate_route_early_exit_used",
        "candidate_route_early_exit_reason",
        "candidate_route_early_exit_score_gap",
        "hybrid_doc_gate_applied",
        "hybrid_doc_gate_bm25_before",
        "hybrid_doc_gate_bm25_after",
        "hybrid_doc_gate_vector_before",
        "hybrid_doc_gate_vector_after",
        "hybrid_doc_gate_removed_bm25",
        "hybrid_doc_gate_removed_vector",
        "hybrid_doc_gate_override_kept_original",
        "exact_doc_top10",
        "focus_top3",
        "topical_doc_top10",
        "answer_grounded",
        "structured_lookup_match_type",
        "structured_lookup_confidence",
        "structured_lookup_candidates",
        "rerank_top1_doc_id",
        "rerank_top1_document_number",
        "rerank_top1_article",
        "rerank_top1_clause",
        "rerank_top1_path",
        "rerank_top1_family_key",
        "rerank_top2_path",
        "rerank_top3_path",
        "question",
    ]

    with failure_csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in fail_rows:
            writer.writerow(
                {
                    "run": row.get("run", ""),
                    "case_id": row.get("case_id", ""),
                    "expected_route": row.get("expected_route", ""),
                    "actual_route": row.get("actual_route", ""),
                    "primary_failure_label": row.get("primary_failure_label", ""),
                    "failure_labels": _join_seq(row.get("failure_labels", []), sep="|"),
                    "primary_true_failure_label": row.get("primary_true_failure_label", ""),
                    "true_failure_labels": _join_seq(row.get("true_failure_labels", []), sep="|"),
                    "acceptable_route_promotion": int(row.get("acceptable_route_promotion", 0)),
                    "acceptable_alias_tie": int(row.get("acceptable_alias_tie", 0)),
                    "acceptable_scope_promotion": int(row.get("acceptable_scope_promotion", 0)),
                    "is_true_failure": int(row.get("is_true_failure", 0)),
                    "is_true_focus_failure": int(row.get("is_true_focus_failure", 0)),
                    "route_correct": row.get("route_correct", 0),
                    "route_semantic_correct": row.get("route_semantic_correct", row.get("route_correct", 0)),
                    "retrieval_latency_ms": row.get("retrieval_latency_ms", 0),
                    "bm25_latency_ms": row.get("bm25_latency_ms", 0),
                    "vector_top_k_effective": row.get("vector_top_k_effective", 0),
                    "vector_latency_ms": row.get("vector_latency_ms", 0),
                    "vector_wall_ms": row.get("vector_wall_ms", 0),
                    "vector_budget_ms": row.get("vector_budget_ms", 0),
                    "vector_join_wait_ms": row.get("vector_join_wait_ms", 0),
                    "vector_timed_out": int(bool(row.get("vector_timed_out", False))),
                    "vector_deadline_hit": int(bool(row.get("vector_deadline_hit", False))),
                    "vector_result_used": int(bool(row.get("vector_result_used", False))),
                    "vector_cancelled": int(bool(row.get("vector_cancelled", False))),
                    "vector_skipped_by_route": int(bool(row.get("vector_skipped_by_route", False))),
                    "vector_skipped_by_quality_gate": int(bool(row.get("vector_skipped_by_quality_gate", False))),
                    "bm25_scoped": int(bool(row.get("bm25_scoped", False))),
                    "vector_scoped": int(bool(row.get("vector_scoped", False))),
                    "allowed_doc_ids_count": row.get("allowed_doc_ids_count", 0),
                    "scoped_filter_broken": int(bool(row.get("scoped_filter_broken", False))),
                    "scoped_filter_broken_doc_ids": _join_seq(row.get("scoped_filter_broken_doc_ids", [])),
                    "correct_law_top10": row.get("correct_law_top10", 0),
                    "correct_family_top10": row.get("correct_family_top10", 0),
                    "law_found_top10": row.get("law_found_top10", 0),
                    "correct_article_top10": row.get("correct_article_top10", 0),
                    "correct_article_top10_given_law_found": row.get(
                        "correct_article_top10_given_law_found",
                        0,
                    ),
                    "focus_top1": row.get("focus_top1", 0),
                    "focus_top1_given_law_found": row.get(
                        "focus_top1_given_law_found",
                        0,
                    ),
                    "focus_top1_within_family": row.get("focus_top1_within_family", 0),
                    "expected_doc_id": row.get("expected_doc_id", ""),
                    "expected_doc_number": row.get("expected_doc_number", ""),
                    "expected_family_key": row.get("expected_family_key", ""),
                    "expected_article_num": row.get("expected_article_num", ""),
                    "expected_clause_num": row.get("expected_clause_num", ""),
                    "expected_focus_terms": _join_seq(row.get("expected_focus_terms", [])),
                    "expected_topical_terms": _join_seq(row.get("expected_topical_terms", [])),
                    "parser_document_number": row.get("parser_document_number", ""),
                    "parser_document_loose": row.get("parser_document_loose", ""),
                    "parser_document_short": row.get("parser_document_short", ""),
                    "parser_article_number": row.get("parser_article_number", ""),
                    "parser_clause_number": row.get("parser_clause_number", ""),
                    "parser_law_name": row.get("parser_law_name", ""),
                    "bm25_mode": row.get("bm25_mode", ""),
                    "bm25_top_k_effective": row.get("bm25_top_k_effective", 0),
                    "bm25_tokenized_query": row.get("bm25_tokenized_query", ""),
                    "bm25_parsed_query": row.get("bm25_parsed_query", ""),
                    "bm25_query_fields": _join_seq(row.get("bm25_query_fields", [])),
                    "bm25_anchor_tokens": _join_seq(row.get("bm25_anchor_tokens", [])),
                    "bm25_intent_tokens": _join_seq(row.get("bm25_intent_tokens", [])),
                    "bm25_is_multifield": int(bool(row.get("bm25_is_multifield", False))),
                    "narrow_doc_number_raw": row.get("narrow_doc_number_raw", ""),
                    "narrow_doc_number_canonical": row.get("narrow_doc_number_canonical", ""),
                    "narrow_doc_number_loose_key": row.get("narrow_doc_number_loose_key", ""),
                    "narrow_doc_number_exact_candidates_count": row.get(
                        "narrow_doc_number_exact_candidates_count",
                        0,
                    ),
                    "narrow_doc_number_exact_candidates": _join_seq(
                        row.get("narrow_doc_number_exact_candidates", [])
                    ),
                    "narrow_doc_number_exact_pass_used": int(
                        bool(row.get("narrow_doc_number_exact_pass_used", False))
                    ),
                    "narrow_doc_number_exact_pass_hits": row.get(
                        "narrow_doc_number_exact_pass_hits",
                        0,
                    ),
                    "narrow_doc_number_bias_applied": int(
                        bool(row.get("narrow_doc_number_bias_applied", False))
                    ),
                    "narrow_doc_number_top1_match_level": row.get(
                        "narrow_doc_number_top1_match_level",
                        "",
                    ),
                    "narrow_doc_number_mismatch_reason": row.get(
                        "narrow_doc_number_mismatch_reason",
                        "",
                    ),
                    "document_lookup_used": int(bool(row.get("document_lookup_used", False))),
                    "document_lookup_ms": row.get("document_lookup_ms", 0),
                    "document_lookup_matched": int(bool(row.get("document_lookup_matched", False))),
                    "document_lookup_match_type": row.get("document_lookup_match_type", ""),
                    "document_lookup_confidence": row.get("document_lookup_confidence", 0.0),
                    "document_lookup_doc_ids_count": row.get("document_lookup_doc_ids_count", 0),
                    "document_lookup_top1_confidence": row.get("document_lookup_top1_confidence", 0.0),
                    "document_lookup_top2_confidence": row.get("document_lookup_top2_confidence", 0.0),
                    "document_lookup_margin": row.get("document_lookup_margin", 0.0),
                    "document_lookup_selected_route": row.get("document_lookup_selected_route", ""),
                    "document_lookup_selected_reason": row.get("document_lookup_selected_reason", ""),
                    "query_rewrite_used": int(bool(row.get("query_rewrite_used", False))),
                    "query_rewrite_ms": row.get("query_rewrite_ms", 0),
                    "query_rewrite_confidence": row.get("query_rewrite_confidence", 0.0),
                    "query_too_vague": int(bool(row.get("query_too_vague", False))),
                    "query_rewrite_clean": row.get("query_rewrite_clean", ""),
                    "query_rewrite_lexical": row.get("query_rewrite_lexical", ""),
                    "query_rewrite_semantic": row.get("query_rewrite_semantic", ""),
                    "query_rewrite_focus_terms": _join_seq(row.get("query_rewrite_focus_terms", [])),
                    "query_rewrite_fillers_removed": _join_seq(row.get("query_rewrite_fillers_removed", [])),
                    "query_rewrite_phrase_repairs": _join_seq(row.get("query_rewrite_phrase_repairs", [])),
                    "query_rewrite_legal_anchor_guess": row.get("query_rewrite_legal_anchor_guess", ""),
                    "query_rewrite_legal_anchor_guess_list": _join_seq(
                        row.get("query_rewrite_legal_anchor_guess_list", [])
                    ),
                    "query_rewrite_doc_type_prior": _join_seq(row.get("query_rewrite_doc_type_prior", [])),
                    "query_rewrite_exclude_doc_type_hint": _join_seq(
                        row.get("query_rewrite_exclude_doc_type_hint", [])
                    ),
                    "query_rewrite_topic_class": row.get("query_rewrite_topic_class", ""),
                    "query_rewrite_subclass": row.get("query_rewrite_subclass", "unknown"),
                    "query_rewrite_legal_concept_tags": _join_seq(
                        row.get("query_rewrite_legal_concept_tags", [])
                    ),
                    "query_rewrite_actor_terms": _join_seq(row.get("query_rewrite_actor_terms", [])),
                    "query_rewrite_action_terms": _join_seq(row.get("query_rewrite_action_terms", [])),
                    "query_rewrite_object_terms": _join_seq(row.get("query_rewrite_object_terms", [])),
                    "query_rewrite_qualifier_terms": _join_seq(row.get("query_rewrite_qualifier_terms", [])),
                    "query_rewrite_vagueness_level": row.get("query_rewrite_vagueness_level", "none"),
                    "query_rewrite_concept_confidence": row.get("query_rewrite_concept_confidence", 0.0),
                    "query_rewrite_v3_used": int(bool(row.get("query_rewrite_v3_used", False))),
                    "query_rewrite_is_concept_query": int(bool(row.get("query_rewrite_is_concept_query", False))),
                    "query_rewrite_is_topic_broad": int(bool(row.get("query_rewrite_is_topic_broad", False))),
                    "query_rewrite_lexical_core": row.get("query_rewrite_lexical_core", ""),
                    "query_rewrite_lexical_expanded": row.get("query_rewrite_lexical_expanded", ""),
                    "query_rewrite_concept_seed_query": row.get("query_rewrite_concept_seed_query", ""),
                    "query_rewrite_title_anchor_query": row.get("query_rewrite_title_anchor_query", ""),
                    "query_rewrite_risk": row.get("query_rewrite_risk", "medium"),
                    "query_rewrite_lexical_is_weak": int(bool(row.get("query_rewrite_lexical_is_weak", False))),
                    "query_rewrite_weak_query_abort": int(
                        bool(row.get("query_rewrite_weak_query_abort", False))
                    ),
                    "query_rewrite_weak_query_abort_reasons": _join_seq(
                        row.get("query_rewrite_weak_query_abort_reasons", [])
                    ),
                    "query_rewrite_lexical_quality_flags": _join_seq(
                        row.get("query_rewrite_lexical_quality_flags", [])
                    ),
                    "query_rewrite_intent_template_hits": _join_seq(
                        row.get("query_rewrite_intent_template_hits", [])
                    ),
                    "query_rewrite_lexical_expansion_used": _join_seq(
                        row.get("query_rewrite_lexical_expansion_used", [])
                    ),
                    "hybrid_default_mode": row.get("hybrid_default_mode", ""),
                    "hybrid_default_candidate_family_count": row.get(
                        "hybrid_default_candidate_family_count",
                        0,
                    ),
                    "hybrid_default_candidate_doc_count": row.get(
                        "hybrid_default_candidate_doc_count",
                        0,
                    ),
                    "hybrid_default_title_anchor_used": int(
                        bool(row.get("hybrid_default_title_anchor_used", False))
                    ),
                    "hybrid_default_title_anchor_hits": row.get("hybrid_default_title_anchor_hits", 0),
                    "hybrid_default_doc_aggregation_used": int(
                        bool(row.get("hybrid_default_doc_aggregation_used", False))
                    ),
                    "hybrid_default_doc_score_top1": row.get("hybrid_default_doc_score_top1", 0.0),
                    "hybrid_default_doc_score_top2": row.get("hybrid_default_doc_score_top2", 0.0),
                    "hybrid_default_doc_score_margin": row.get("hybrid_default_doc_score_margin", 0.0),
                    "hybrid_default_doc_top1_family": row.get("hybrid_default_doc_top1_family", ""),
                    "hybrid_default_doc_top2_family": row.get("hybrid_default_doc_top2_family", ""),
                    "hybrid_default_vector_used_for_doc_recall": int(
                        bool(row.get("hybrid_default_vector_used_for_doc_recall", False))
                    ),
                    "hybrid_default_vector_support_same_family": int(
                        bool(row.get("hybrid_default_vector_support_same_family", False))
                    ),
                    "hybrid_default_genericity_penalty_top1": row.get(
                        "hybrid_default_genericity_penalty_top1",
                        0.0,
                    ),
                    "hybrid_default_genericity_penalty_applied_count": row.get(
                        "hybrid_default_genericity_penalty_applied_count",
                        0,
                    ),
                    "hybrid_default_focus_rerank_used": int(
                        bool(row.get("hybrid_default_focus_rerank_used", False))
                    ),
                    "hybrid_default_focus_rerank_candidate_chunks": row.get(
                        "hybrid_default_focus_rerank_candidate_chunks",
                        0,
                    ),
                    "hybrid_default_focus_heading_match_top1": row.get(
                        "hybrid_default_focus_heading_match_top1",
                        0.0,
                    ),
                    "hybrid_default_focus_actor_action_top1": row.get(
                        "hybrid_default_focus_actor_action_top1",
                        0.0,
                    ),
                    "hybrid_default_focus_concept_match_top1": row.get(
                        "hybrid_default_focus_concept_match_top1",
                        0.0,
                    ),
                    "hybrid_default_selected_family_keys": _join_seq(
                        row.get("hybrid_default_selected_family_keys", [])
                    ),
                    "hybrid_default_selected_doc_ids": _join_seq(
                        row.get("hybrid_default_selected_doc_ids", [])
                    ),
                    "hybrid_default_family_search_used": int(
                        bool(row.get("hybrid_default_family_search_used", False))
                    ),
                    "hybrid_default_family_search_hits": row.get(
                        "hybrid_default_family_search_hits",
                        0,
                    ),
                    "hybrid_default_family_candidate_count_pre_filter": row.get(
                        "hybrid_default_family_candidate_count_pre_filter",
                        0,
                    ),
                    "hybrid_default_family_candidate_count_post_filter": row.get(
                        "hybrid_default_family_candidate_count_post_filter",
                        0,
                    ),
                    "hybrid_default_family_score_top1": row.get(
                        "hybrid_default_family_score_top1",
                        0.0,
                    ),
                    "hybrid_default_family_score_top2": row.get(
                        "hybrid_default_family_score_top2",
                        0.0,
                    ),
                    "hybrid_default_family_score_margin": row.get(
                        "hybrid_default_family_score_margin",
                        0.0,
                    ),
                    "hybrid_default_family_identity_score_top1": row.get(
                        "hybrid_default_family_identity_score_top1",
                        0.0,
                    ),
                    "hybrid_default_family_identity_score_top2": row.get(
                        "hybrid_default_family_identity_score_top2",
                        0.0,
                    ),
                    "hybrid_default_family_identity_margin": row.get(
                        "hybrid_default_family_identity_margin",
                        0.0,
                    ),
                    "hybrid_default_family_top1_doc_role": row.get(
                        "hybrid_default_family_top1_doc_role",
                        "",
                    ),
                    "hybrid_default_family_top1_is_implementation": int(
                        bool(row.get("hybrid_default_family_top1_is_implementation", False))
                    ),
                    "hybrid_default_family_recovery_used": int(
                        bool(row.get("hybrid_default_family_recovery_used", False))
                    ),
                    "hybrid_default_family_recovery_promoted": row.get(
                        "hybrid_default_family_recovery_promoted",
                        0,
                    ),
                    "hybrid_default_weak_query_abort_used": int(
                        bool(row.get("hybrid_default_weak_query_abort_used", False))
                    ),
                    "hybrid_default_no_confident_family_candidates": int(
                        bool(row.get("hybrid_default_no_confident_family_candidates", False))
                    ),
                    "hybrid_default_garbage_family_rejected": int(
                        bool(row.get("hybrid_default_garbage_family_rejected", False))
                    ),
                    "hybrid_default_garbage_family_reject_reason": row.get(
                        "hybrid_default_garbage_family_reject_reason",
                        "",
                    ),
                    "hybrid_default_trusted_family": int(
                        bool(row.get("hybrid_default_trusted_family", False))
                    ),
                    "hybrid_default_shortlist_after_reject_count": row.get(
                        "hybrid_default_shortlist_after_reject_count",
                        0,
                    ),
                    "hybrid_default_shortlist_after_reject_empty": int(
                        bool(row.get("hybrid_default_shortlist_after_reject_empty", False))
                    ),
                    "hybrid_default_doc_role_prior_applied": int(
                        bool(row.get("hybrid_default_doc_role_prior_applied", False))
                    ),
                    "hybrid_default_doc_role_prior_top1": row.get(
                        "hybrid_default_doc_role_prior_top1",
                        "",
                    ),
                    "hybrid_default_wrong_doc_role_penalty_applied": int(
                        bool(row.get("hybrid_default_wrong_doc_role_penalty_applied", False))
                    ),
                    "hybrid_default_implementation_bridge_used": int(
                        bool(row.get("hybrid_default_implementation_bridge_used", False))
                    ),
                    "hybrid_default_implementation_bridge_hits": row.get(
                        "hybrid_default_implementation_bridge_hits",
                        0,
                    ),
                    "hybrid_default_focus_rerank_blocked_by_low_family_confidence": int(
                        bool(row.get("hybrid_default_focus_rerank_blocked_by_low_family_confidence", False))
                    ),
                    "hybrid_default_focus_rerank_stage_entered_with_family_confidence": row.get(
                        "hybrid_default_focus_rerank_stage_entered_with_family_confidence",
                        0.0,
                    ),
                    "rank_first_family_pre_focus": row.get("rank_first_family_pre_focus", 0),
                    "rank_first_family_post_focus": row.get("rank_first_family_post_focus", 0),
                    "rank_first_doc_role_pre_focus": row.get("rank_first_doc_role_pre_focus", ""),
                    "rank_first_doc_role_post_focus": row.get("rank_first_doc_role_post_focus", ""),
                    "hybrid_default_scope_fallback_used": int(
                        bool(row.get("hybrid_default_scope_fallback_used", False))
                    ),
                    "hybrid_default_scope_fallback_reason": row.get(
                        "hybrid_default_scope_fallback_reason",
                        "",
                    ),
                    "hybrid_default_scope_fallback_hits": row.get(
                        "hybrid_default_scope_fallback_hits",
                        0,
                    ),
                    "hybrid_default_scope_fallback_bm25_only_used": int(
                        bool(row.get("hybrid_default_scope_fallback_bm25_only_used", False))
                    ),
                    "hybrid_default_scope_fallback_bm25_only_hits": row.get(
                        "hybrid_default_scope_fallback_bm25_only_hits",
                        0,
                    ),
                    "hybrid_default_scope_fallback_bm25_only_latency_ms": row.get(
                        "hybrid_default_scope_fallback_bm25_only_latency_ms",
                        0,
                    ),
                    "hybrid_default_scope_fallback_bm25_only_query": row.get(
                        "hybrid_default_scope_fallback_bm25_only_query",
                        "",
                    ),
                    "hybrid_default_scope_fallback_bm25_only_query_reason": row.get(
                        "hybrid_default_scope_fallback_bm25_only_query_reason",
                        "",
                    ),
                    "hybrid_default_scope_fallback_vector_reused": int(
                        bool(row.get("hybrid_default_scope_fallback_vector_reused", False))
                    ),
                    "hybrid_default_scope_fallback_second_vector_called": int(
                        bool(row.get("hybrid_default_scope_fallback_second_vector_called", False))
                    ),
                    "hybrid_default_scope_fallback_first_pass_hits": row.get(
                        "hybrid_default_scope_fallback_first_pass_hits",
                        0,
                    ),
                    "hybrid_default_scope_fallback_added_law10": row.get(
                        "hybrid_default_scope_fallback_added_law10",
                        0,
                    ),
                    "hybrid_default_scope_fallback_added_article10": row.get(
                        "hybrid_default_scope_fallback_added_article10",
                        0,
                    ),
                    "hybrid_default_scope_fallback_added_focus1": row.get(
                        "hybrid_default_scope_fallback_added_focus1",
                        0,
                    ),
                    "hybrid_default_hard_doc_type_filter": int(
                        bool(row.get("hybrid_default_hard_doc_type_filter", False))
                    ),
                    "hybrid_default_hard_doc_type_allowlist": _join_seq(
                        row.get("hybrid_default_hard_doc_type_allowlist", [])
                    ),
                    "hybrid_default_concept_policy_applied": int(
                        bool(row.get("hybrid_default_concept_policy_applied", False))
                    ),
                    "hybrid_default_concept_vector_optional": int(
                        bool(row.get("hybrid_default_concept_vector_optional", False))
                    ),
                    "hybrid_default_intra_doc_focus_rerank_used": int(
                        bool(row.get("hybrid_default_intra_doc_focus_rerank_used", False))
                    ),
                    "hybrid_default_intra_doc_focus_target_doc_ids": _join_seq(
                        row.get("hybrid_default_intra_doc_focus_target_doc_ids", [])
                    ),
                    "hybrid_default_intra_doc_focus_target_family_keys": _join_seq(
                        row.get("hybrid_default_intra_doc_focus_target_family_keys", [])
                    ),
                    "hybrid_default_intra_doc_focus_rank_first_doc": row.get(
                        "hybrid_default_intra_doc_focus_rank_first_doc",
                        0,
                    ),
                    "hybrid_default_intra_doc_focus_rank_first_family": row.get(
                        "hybrid_default_intra_doc_focus_rank_first_family",
                        0,
                    ),
                    "hybrid_default_intra_doc_focus_candidates": row.get(
                        "hybrid_default_intra_doc_focus_candidates",
                        0,
                    ),
                    "hybrid_default_intra_doc_focus_promoted": row.get(
                        "hybrid_default_intra_doc_focus_promoted",
                        0,
                    ),
                    "hybrid_anchor_scope_used": int(bool(row.get("hybrid_anchor_scope_used", False))),
                    "hybrid_anchor_scope_confidence": row.get("hybrid_anchor_scope_confidence", 0.0),
                    "hybrid_anchor_scope_doc_ids_count": row.get("hybrid_anchor_scope_doc_ids_count", 0),
                    "hybrid_anchor_scope_doc_ids": _join_seq(row.get("hybrid_anchor_scope_doc_ids", [])),
                    "hybrid_anchor_scope_reason": row.get("hybrid_anchor_scope_reason", ""),
                    "document_lookup_candidate_doc_ids_original_count": row.get(
                        "document_lookup_candidate_doc_ids_original_count",
                        0,
                    ),
                    "document_lookup_candidate_doc_ids_trimmed": int(
                        bool(row.get("document_lookup_candidate_doc_ids_trimmed", False))
                    ),
                    "candidate_route_early_exit_used": int(
                        bool(row.get("candidate_route_early_exit_used", False))
                    ),
                    "candidate_route_early_exit_reason": row.get("candidate_route_early_exit_reason", ""),
                    "candidate_route_early_exit_score_gap": row.get(
                        "candidate_route_early_exit_score_gap",
                        0.0,
                    ),
                    "hybrid_doc_gate_applied": int(bool(row.get("hybrid_doc_gate_applied", False))),
                    "hybrid_doc_gate_bm25_before": row.get("hybrid_doc_gate_bm25_before", 0),
                    "hybrid_doc_gate_bm25_after": row.get("hybrid_doc_gate_bm25_after", 0),
                    "hybrid_doc_gate_vector_before": row.get("hybrid_doc_gate_vector_before", 0),
                    "hybrid_doc_gate_vector_after": row.get("hybrid_doc_gate_vector_after", 0),
                    "hybrid_doc_gate_removed_bm25": row.get("hybrid_doc_gate_removed_bm25", 0),
                    "hybrid_doc_gate_removed_vector": row.get("hybrid_doc_gate_removed_vector", 0),
                    "hybrid_doc_gate_override_kept_original": int(
                        bool(row.get("hybrid_doc_gate_override_kept_original", False))
                    ),
                    "exact_doc_top10": row.get("exact_doc_top10", 0),
                    "focus_top3": row.get("focus_top3", 0),
                    "topical_doc_top10": row.get("topical_doc_top10", 0),
                    "answer_grounded": row.get("answer_grounded", 0),
                    "structured_lookup_match_type": row.get("structured_lookup_match_type", ""),
                    "structured_lookup_confidence": row.get("structured_lookup_confidence", 0.0),
                    "structured_lookup_candidates": row.get("structured_lookup_candidates", 0),
                    "rerank_top1_doc_id": row.get("rerank_top1_doc_id", ""),
                    "rerank_top1_document_number": row.get("rerank_top1_document_number", ""),
                    "rerank_top1_article": row.get("rerank_top1_article", ""),
                    "rerank_top1_clause": row.get("rerank_top1_clause", ""),
                    "rerank_top1_path": row.get("rerank_top1_path", ""),
                    "rerank_top1_family_key": row.get("rerank_top1_family_key", ""),
                    "rerank_top2_path": row.get("rerank_top2_path", ""),
                    "rerank_top3_path": row.get("rerank_top3_path", ""),
                    "question": row.get("question", ""),
                }
            )

    with failure_debug_out.open("w", encoding="utf-8") as f:
        for row in fail_rows:
            payload = {
                "run": row.get("run", ""),
                "case_id": row.get("case_id", ""),
                "primary_failure_label": row.get("primary_failure_label", ""),
                "failure_labels": row.get("failure_labels", []),
                "primary_true_failure_label": row.get("primary_true_failure_label", "PASS"),
                "true_failure_labels": row.get("true_failure_labels", ["PASS"]),
                "acceptable_route_promotion": int(row.get("acceptable_route_promotion", 0)),
                "acceptable_alias_tie": int(row.get("acceptable_alias_tie", 0)),
                "acceptable_scope_promotion": int(row.get("acceptable_scope_promotion", 0)),
                "is_true_failure": int(row.get("is_true_failure", 0)),
                "is_true_focus_failure": int(row.get("is_true_focus_failure", 0)),
                "question": row.get("question", ""),
                "expected": {
                    "route": row.get("expected_route", ""),
                    "doc_id": row.get("expected_doc_id", ""),
                    "doc_number": row.get("expected_doc_number", ""),
                    "family_key": row.get("expected_family_key", ""),
                    "article_num": row.get("expected_article_num", ""),
                    "clause_num": row.get("expected_clause_num", ""),
                    "focus_terms": row.get("expected_focus_terms", []),
                    "topical_terms": row.get("expected_topical_terms", []),
                },
                "actual": {
                    "route": row.get("actual_route", ""),
                    "route_correct": int(row.get("route_correct", 0)),
                    "route_semantic_correct": int(row.get("route_semantic_correct", row.get("route_correct", 0))),
                    "bm25_mode": row.get("bm25_mode", ""),
                    "bm25_top_k_effective": int(row.get("bm25_top_k_effective", 0)),
                    "vector_top_k_effective": int(row.get("vector_top_k_effective", 0)),
                    "latency_ms": {
                        "bm25": row.get("bm25_latency_ms", 0),
                        "vector": row.get("vector_latency_ms", 0),
                        "vector_wall": row.get("vector_wall_ms", 0),
                        "vector_budget": row.get("vector_budget_ms", 0),
                        "vector_join_wait": row.get("vector_join_wait_ms", 0),
                        "retrieval": row.get("retrieval_latency_ms", 0),
                    },
                    "vector_runtime": {
                        "timed_out": bool(row.get("vector_timed_out", False)),
                        "deadline_hit": bool(row.get("vector_deadline_hit", False)),
                        "result_used": bool(row.get("vector_result_used", False)),
                        "cancelled": bool(row.get("vector_cancelled", False)),
                        "skipped_by_route": bool(row.get("vector_skipped_by_route", False)),
                        "skipped_by_quality_gate": bool(row.get("vector_skipped_by_quality_gate", False)),
                    },
                    "quality": {
                        "exact_doc_top10": row.get("exact_doc_top10", 0),
                        "correct_law_top10": row.get("correct_law_top10", 0),
                        "correct_family_top10": row.get("correct_family_top10", 0),
                        "correct_article_top10": row.get("correct_article_top10", 0),
                        "focus_top1": row.get("focus_top1", 0),
                        "focus_top1_within_family": row.get("focus_top1_within_family", 0),
                        "focus_top3": row.get("focus_top3", 0),
                        "topical_doc_top10": row.get("topical_doc_top10", 0),
                        "answer_grounded": row.get("answer_grounded", 0),
                        "fallback_added_law10": row.get(
                            "hybrid_default_scope_fallback_added_law10",
                            0,
                        ),
                        "fallback_added_article10": row.get(
                            "hybrid_default_scope_fallback_added_article10",
                            0,
                        ),
                        "fallback_added_focus1": row.get(
                            "hybrid_default_scope_fallback_added_focus1",
                            0,
                        ),
                    },
                    "scoped": {
                        "bm25_scoped": bool(row.get("bm25_scoped", False)),
                        "vector_scoped": bool(row.get("vector_scoped", False)),
                        "allowed_doc_ids_count": int(row.get("allowed_doc_ids_count", 0)),
                        "scoped_filter_broken": bool(row.get("scoped_filter_broken", False)),
                        "scoped_filter_broken_doc_ids": row.get("scoped_filter_broken_doc_ids", []),
                    },
                    "document_lookup": {
                        "used": bool(row.get("document_lookup_used", False)),
                        "matched": bool(row.get("document_lookup_matched", False)),
                        "match_type": row.get("document_lookup_match_type", ""),
                        "confidence": row.get("document_lookup_confidence", 0.0),
                        "top1_confidence": row.get("document_lookup_top1_confidence", 0.0),
                        "top2_confidence": row.get("document_lookup_top2_confidence", 0.0),
                        "margin": row.get("document_lookup_margin", 0.0),
                        "selected_route": row.get("document_lookup_selected_route", ""),
                        "selected_reason": row.get("document_lookup_selected_reason", ""),
                        "doc_ids_count": int(row.get("document_lookup_doc_ids_count", 0)),
                        "latency_ms": int(row.get("document_lookup_ms", 0)),
                    },
                    "query_rewrite": {
                        "used": bool(row.get("query_rewrite_used", False)),
                        "confidence": row.get("query_rewrite_confidence", 0.0),
                        "too_vague": bool(row.get("query_too_vague", False)),
                        "lexical_is_weak": bool(row.get("query_rewrite_lexical_is_weak", False)),
                        "lexical_quality_flags": row.get("query_rewrite_lexical_quality_flags", []),
                        "intent_template_hits": row.get("query_rewrite_intent_template_hits", []),
                        "lexical_expansion_used": row.get(
                            "query_rewrite_lexical_expansion_used",
                            [],
                        ),
                        "latency_ms": int(row.get("query_rewrite_ms", 0)),
                        "clean": row.get("query_rewrite_clean", ""),
                        "lexical": row.get("query_rewrite_lexical", ""),
                        "semantic": row.get("query_rewrite_semantic", ""),
                        "focus_terms": row.get("query_rewrite_focus_terms", []),
                        "fillers_removed": row.get("query_rewrite_fillers_removed", []),
                        "phrase_repairs": row.get("query_rewrite_phrase_repairs", []),
                        "legal_anchor_guess": row.get("query_rewrite_legal_anchor_guess", ""),
                        "doc_type_prior": row.get("query_rewrite_doc_type_prior", []),
                        "exclude_doc_type_hint": row.get("query_rewrite_exclude_doc_type_hint", []),
                        "topic_class": row.get("query_rewrite_topic_class", ""),
                    },
                    "hybrid_anchor_scope": {
                        "used": bool(row.get("hybrid_anchor_scope_used", False)),
                        "confidence": float(row.get("hybrid_anchor_scope_confidence", 0.0)),
                        "doc_ids_count": int(row.get("hybrid_anchor_scope_doc_ids_count", 0)),
                        "doc_ids": row.get("hybrid_anchor_scope_doc_ids", []),
                        "reason": row.get("hybrid_anchor_scope_reason", ""),
                    },
                    "hybrid_doc_gate": {
                        "applied": bool(row.get("hybrid_doc_gate_applied", False)),
                        "hard_doc_type_filter": bool(
                            row.get("hybrid_default_hard_doc_type_filter", False)
                        ),
                        "hard_doc_type_allowlist": row.get(
                            "hybrid_default_hard_doc_type_allowlist",
                            [],
                        ),
                        "bm25_before": int(row.get("hybrid_doc_gate_bm25_before", 0)),
                        "bm25_after": int(row.get("hybrid_doc_gate_bm25_after", 0)),
                        "vector_before": int(row.get("hybrid_doc_gate_vector_before", 0)),
                        "vector_after": int(row.get("hybrid_doc_gate_vector_after", 0)),
                        "removed_bm25": int(row.get("hybrid_doc_gate_removed_bm25", 0)),
                        "removed_vector": int(row.get("hybrid_doc_gate_removed_vector", 0)),
                        "override_kept_original": bool(
                            row.get("hybrid_doc_gate_override_kept_original", False)
                        ),
                    },
                    "hybrid_default_scope_fallback": {
                        "used": bool(row.get("hybrid_default_scope_fallback_used", False)),
                        "reason": row.get("hybrid_default_scope_fallback_reason", ""),
                        "hits": int(row.get("hybrid_default_scope_fallback_hits", 0)),
                        "bm25_only_used": bool(
                            row.get("hybrid_default_scope_fallback_bm25_only_used", False)
                        ),
                        "bm25_only_hits": int(
                            row.get("hybrid_default_scope_fallback_bm25_only_hits", 0)
                        ),
                        "bm25_only_latency_ms": int(
                            row.get("hybrid_default_scope_fallback_bm25_only_latency_ms", 0)
                        ),
                        "bm25_only_query": row.get(
                            "hybrid_default_scope_fallback_bm25_only_query",
                            "",
                        ),
                        "bm25_only_query_reason": row.get(
                            "hybrid_default_scope_fallback_bm25_only_query_reason",
                            "",
                        ),
                        "vector_reused": bool(
                            row.get("hybrid_default_scope_fallback_vector_reused", False)
                        ),
                        "second_vector_called": bool(
                            row.get("hybrid_default_scope_fallback_second_vector_called", False)
                        ),
                        "first_pass_hits": int(
                            row.get("hybrid_default_scope_fallback_first_pass_hits", 0)
                        ),
                        "first_pass_top10": row.get(
                            "hybrid_default_scope_fallback_first_pass_top10",
                            [],
                        ),
                    },
                    "candidate_route_runtime": {
                        "doc_ids_original_count": int(
                            row.get("document_lookup_candidate_doc_ids_original_count", 0)
                        ),
                        "doc_ids_trimmed": bool(row.get("document_lookup_candidate_doc_ids_trimmed", False)),
                        "early_exit_used": bool(row.get("candidate_route_early_exit_used", False)),
                        "early_exit_reason": row.get("candidate_route_early_exit_reason", ""),
                        "early_exit_score_gap": float(row.get("candidate_route_early_exit_score_gap", 0.0)),
                    },
                },
                "parser": {
                    "document_number": row.get("parser_document_number", ""),
                    "document_loose": row.get("parser_document_loose", ""),
                    "document_short": row.get("parser_document_short", ""),
                    "article_number": row.get("parser_article_number", ""),
                    "clause_number": row.get("parser_clause_number", ""),
                    "law_name": row.get("parser_law_name", ""),
                },
                "bm25_debug": {
                    "tokenized_query": row.get("bm25_tokenized_query", ""),
                    "parsed_query": row.get("bm25_parsed_query", ""),
                    "query_fields": row.get("bm25_query_fields", []),
                    "is_multifield": row.get("bm25_is_multifield", False),
                    "anchor_tokens": row.get("bm25_anchor_tokens", []),
                    "intent_tokens": row.get("bm25_intent_tokens", []),
                    "narrow_doc_number_raw": row.get("narrow_doc_number_raw", ""),
                    "narrow_doc_number_canonical": row.get("narrow_doc_number_canonical", ""),
                    "narrow_doc_number_loose_key": row.get("narrow_doc_number_loose_key", ""),
                    "narrow_doc_number_exact_candidates_count": int(
                        row.get("narrow_doc_number_exact_candidates_count", 0)
                    ),
                    "narrow_doc_number_exact_candidates": row.get(
                        "narrow_doc_number_exact_candidates",
                        [],
                    ),
                    "narrow_doc_number_exact_pass_used": bool(
                        row.get("narrow_doc_number_exact_pass_used", False)
                    ),
                    "narrow_doc_number_exact_pass_hits": int(
                        row.get("narrow_doc_number_exact_pass_hits", 0)
                    ),
                    "narrow_doc_number_bias_applied": bool(
                        row.get("narrow_doc_number_bias_applied", False)
                    ),
                    "narrow_doc_number_top1_match_level": row.get(
                        "narrow_doc_number_top1_match_level",
                        "",
                    ),
                    "narrow_doc_number_mismatch_reason": row.get(
                        "narrow_doc_number_mismatch_reason",
                        "",
                    ),
                },
                "top_hits": {
                    "bm25": row.get("bm25_top_hits_debug", []),
                    "vector": row.get("vector_top_hits_debug", []),
                    "rerank": row.get("rerank_top_hits_debug", []),
                    "rerank_top1_family_key": row.get("rerank_top1_family_key", ""),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return len(fail_rows), len(fail_rows)


def _run_benchmark(
    *,
    label: str,
    cases: list[dict[str, Any]],
    warmup: bool,
    debug_top_k: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    retriever = HybridRetriever(
        bm25_index_dir=cfg.BM25_INDEX_DIR,
        vector_index_dir=cfg.VECTOR_INDEX_DIR,
        chunks_path=cfg.PROCESSED_DIR / "chunks.jsonl",
        ollama_url=cfg.OLLAMA_BASE_URL,
        embedding_model=cfg.EMBEDDING_MODEL,
        rrf_k=cfg.RRF_K,
    )

    if warmup and cases:
        _ = retriever.search_with_snapshot(
            query=cases[0]["question"],
            bm25_top_k=cfg.BM25_TOP_K,
            vector_top_k=cfg.VECTOR_TOP_K,
            final_top_k=max(10, cfg.HYBRID_TOP_K),
        )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        snapshot = retriever.search_with_snapshot(
            query=case["question"],
            bm25_top_k=cfg.BM25_TOP_K,
            vector_top_k=cfg.VECTOR_TOP_K,
            final_top_k=max(10, cfg.HYBRID_TOP_K),
        )
        reranked_top10 = list(snapshot.get("reranked_hits", [])[:10])
        quality = _evaluate_hit_quality(case, reranked_top10)

        lat = dict(snapshot.get("latencies_ms", {}))
        routing = dict(snapshot.get("routing", {}))
        execution = dict(snapshot.get("execution", {}))
        bm25_debug = dict(snapshot.get("bm25_debug", {}))
        legal_refs = dict(routing.get("legal_refs", {}))
        bm25_top_hits_debug = _format_top_hits(snapshot.get("bm25_hits", []), top_k=debug_top_k)
        vector_top_hits_debug = _format_top_hits(snapshot.get("vector_hits", []), top_k=debug_top_k)
        rerank_top_hits_debug = _format_top_hits(snapshot.get("reranked_hits", []), top_k=debug_top_k)
        rerank_top1 = _top_hit_fields(rerank_top_hits_debug, 1)
        rerank_top2 = _top_hit_fields(rerank_top_hits_debug, 2)
        rerank_top3 = _top_hit_fields(rerank_top_hits_debug, 3)
        expected_family_key = _expected_family_key(case)
        rerank_top1_family_key = _hit_family_key(reranked_top10[0]) if reranked_top10 else ""
        fallback_first_pass_top10 = list(
            execution.get(
                "hybrid_default_scope_fallback_first_pass_top10",
                routing.get("hybrid_default_scope_fallback_first_pass_top10", []),
            )
            or []
        )[:10]
        if fallback_first_pass_top10:
            fallback_first_pass_quality = _evaluate_hit_quality(case, fallback_first_pass_top10)
        else:
            fallback_first_pass_quality = {
                "correct_law_top10": 0,
                "correct_article_top10": 0,
                "focus_top1": 0,
            }
        fallback_bm25_only_used = bool(
            execution.get(
                "hybrid_default_scope_fallback_bm25_only_used",
                routing.get("hybrid_default_scope_fallback_bm25_only_used", False),
            )
        )
        fallback_added_law10 = int(
            fallback_bm25_only_used
            and int(quality.get("correct_law_top10", 0)) == 1
            and int(fallback_first_pass_quality.get("correct_law_top10", 0)) == 0
        )
        fallback_added_article10 = int(
            fallback_bm25_only_used
            and int(quality.get("correct_article_top10", 0)) == 1
            and int(fallback_first_pass_quality.get("correct_article_top10", 0)) == 0
        )
        fallback_added_focus1 = int(
            fallback_bm25_only_used
            and int(quality.get("focus_top1", 0)) == 1
            and int(fallback_first_pass_quality.get("focus_top1", 0)) == 0
        )

        row = {
            "run": label,
            "case_id": case.get("case_id", ""),
            "question": case.get("question", ""),
            "expected_route": case.get("expected_route", ""),
            "actual_route": routing.get("effective_route", ""),
            "route_correct": int(routing.get("effective_route", "") == case.get("expected_route", "")),
            "route_semantic_correct": int(routing.get("effective_route", "") == case.get("expected_route", "")),
            "expected_doc_id": case.get("expected_doc_id", ""),
            "expected_doc_number": case.get("expected_doc_number", ""),
            "expected_article_num": case.get("expected_article_num", ""),
            "expected_clause_num": case.get("expected_clause_num", ""),
            "expected_family_key": expected_family_key,
            "expected_focus_terms": list(case.get("expected_focus_terms", [])),
            "expected_topical_terms": list(
                case.get("expected_topical_terms", case.get("expected_focus_terms", []))
            ),
            "bm25_mode": routing.get("bm25_mode", ""),
            "bm25_top_k_effective": int(
                execution.get("bm25_top_k_effective", routing.get("bm25_top_k_effective", cfg.BM25_TOP_K))
            ),
            "vector_top_k_effective": int(
                execution.get("vector_top_k_effective", routing.get("vector_top_k_effective", cfg.VECTOR_TOP_K))
            ),
            "bm25_latency_ms": int(lat.get("bm25", 0)),
            "vector_latency_ms": int(lat.get("vector", 0)),
            "vector_wall_ms": int(execution.get("vector_wall_ms", routing.get("vector_wall_ms", lat.get("vector", 0)))),
            "vector_budget_ms": int(execution.get("vector_budget_ms", routing.get("vector_budget_ms", 0))),
            "vector_join_wait_ms": int(execution.get("vector_join_wait_ms", routing.get("vector_join_wait_ms", 0))),
            "retrieval_latency_ms": int(lat.get("total", 0)),
            "vector_timed_out": bool(execution.get("vector_timed_out", False)),
            "vector_deadline_hit": bool(execution.get("vector_deadline_hit", routing.get("vector_deadline_hit", False))),
            "vector_result_used": bool(execution.get("vector_result_used", routing.get("vector_result_used", False))),
            "vector_cancelled": bool(execution.get("vector_cancelled", routing.get("vector_cancelled", False))),
            "vector_skipped_by_route": bool(execution.get("vector_skipped_by_route", False)),
            "vector_skipped_by_quality_gate": bool(execution.get("vector_skipped_by_quality_gate", False)),
            "bm25_scoped": bool(execution.get("bm25_scoped", False)),
            "vector_scoped": bool(execution.get("vector_scoped", False)),
            "allowed_doc_ids_count": int(execution.get("allowed_doc_ids_count", 0)),
            "scoped_filter_broken": bool(execution.get("scoped_filter_broken", routing.get("scoped_filter_broken", False))),
            "scoped_filter_broken_doc_ids": list(
                execution.get("scoped_filter_broken_doc_ids", routing.get("scoped_filter_broken_doc_ids", []))
            ),
            "document_lookup_used": bool(routing.get("document_lookup_used", False)),
            "document_lookup_ms": int(routing.get("document_lookup_ms", 0)),
            "document_lookup_matched": bool(routing.get("document_lookup_matched", False)),
            "document_lookup_match_type": routing.get("document_lookup_match_type", ""),
            "document_lookup_confidence": float(routing.get("document_lookup_confidence", 0.0)),
            "document_lookup_doc_ids_count": int(routing.get("document_lookup_doc_ids_count", 0)),
            "document_lookup_top1_confidence": float(routing.get("document_lookup_top1_confidence", 0.0)),
            "document_lookup_top2_confidence": float(routing.get("document_lookup_top2_confidence", 0.0)),
            "document_lookup_margin": float(routing.get("document_lookup_margin", 0.0)),
            "document_lookup_selected_route": routing.get("document_lookup_selected_route", ""),
            "document_lookup_selected_reason": routing.get("document_lookup_selected_reason", ""),
            "structured_lookup_match_type": routing.get("structured_lookup_match_type", ""),
            "structured_lookup_confidence": routing.get("structured_lookup_confidence", 0.0),
            "structured_lookup_candidates": int(routing.get("structured_lookup_candidates", 0)),
            "query_rewrite_used": bool(routing.get("query_rewrite_used", False)),
            "query_rewrite_ms": int(routing.get("query_rewrite_ms", 0)),
            "query_rewrite_confidence": float(routing.get("query_rewrite_confidence", 0.0)),
            "query_too_vague": bool(routing.get("query_too_vague", False)),
            "query_rewrite_clean": routing.get("query_rewrite_clean", ""),
            "query_rewrite_lexical": routing.get("query_rewrite_lexical", ""),
            "query_rewrite_semantic": routing.get("query_rewrite_semantic", ""),
            "query_rewrite_focus_terms": list(routing.get("query_rewrite_focus_terms", [])),
            "query_rewrite_fillers_removed": list(routing.get("query_rewrite_fillers_removed", [])),
            "query_rewrite_phrase_repairs": list(routing.get("query_rewrite_phrase_repairs", [])),
            "query_rewrite_legal_anchor_guess": routing.get("query_rewrite_legal_anchor_guess", ""),
            "query_rewrite_legal_anchor_guess_list": list(
                routing.get("query_rewrite_legal_anchor_guess_list", [])
            ),
            "query_rewrite_doc_type_prior": list(routing.get("query_rewrite_doc_type_prior", [])),
            "query_rewrite_exclude_doc_type_hint": list(routing.get("query_rewrite_exclude_doc_type_hint", [])),
            "query_rewrite_topic_class": routing.get("query_rewrite_topic_class", ""),
            "query_rewrite_subclass": routing.get("query_rewrite_subclass", "unknown"),
            "query_rewrite_legal_concept_tags": list(routing.get("query_rewrite_legal_concept_tags", [])),
            "query_rewrite_actor_terms": list(routing.get("query_rewrite_actor_terms", [])),
            "query_rewrite_action_terms": list(routing.get("query_rewrite_action_terms", [])),
            "query_rewrite_object_terms": list(routing.get("query_rewrite_object_terms", [])),
            "query_rewrite_qualifier_terms": list(routing.get("query_rewrite_qualifier_terms", [])),
            "query_rewrite_vagueness_level": str(routing.get("query_rewrite_vagueness_level", "none")),
            "query_rewrite_concept_confidence": float(
                routing.get("query_rewrite_concept_confidence", 0.0)
            ),
            "query_rewrite_v3_used": bool(routing.get("query_rewrite_v3_used", False)),
            "query_rewrite_is_concept_query": bool(routing.get("query_rewrite_is_concept_query", False)),
            "query_rewrite_is_topic_broad": bool(routing.get("query_rewrite_is_topic_broad", False)),
            "query_rewrite_lexical_core": str(routing.get("query_rewrite_lexical_core", "")),
            "query_rewrite_lexical_expanded": str(routing.get("query_rewrite_lexical_expanded", "")),
            "query_rewrite_concept_seed_query": str(
                routing.get("query_rewrite_concept_seed_query", "")
            ),
            "query_rewrite_title_anchor_query": str(routing.get("query_rewrite_title_anchor_query", "")),
            "query_rewrite_risk": routing.get("query_rewrite_risk", "medium"),
            "query_rewrite_lexical_is_weak": bool(routing.get("query_rewrite_lexical_is_weak", False)),
            "query_rewrite_weak_query_abort": bool(
                routing.get("query_rewrite_weak_query_abort", False)
            ),
            "query_rewrite_weak_query_abort_reasons": list(
                routing.get("query_rewrite_weak_query_abort_reasons", [])
            ),
            "query_rewrite_lexical_quality_flags": list(routing.get("query_rewrite_lexical_quality_flags", [])),
            "query_rewrite_intent_template_hits": list(routing.get("query_rewrite_intent_template_hits", [])),
            "query_rewrite_lexical_expansion_used": list(
                routing.get("query_rewrite_lexical_expansion_used", [])
            ),
            "hybrid_default_mode": str(
                execution.get("hybrid_default_mode", routing.get("hybrid_default_mode", ""))
            ),
            "hybrid_default_candidate_family_count": int(
                execution.get(
                    "hybrid_default_candidate_family_count",
                    routing.get("hybrid_default_candidate_family_count", 0),
                )
            ),
            "hybrid_default_candidate_doc_count": int(
                execution.get(
                    "hybrid_default_candidate_doc_count",
                    routing.get("hybrid_default_candidate_doc_count", 0),
                )
            ),
            "hybrid_default_title_anchor_used": bool(
                execution.get(
                    "hybrid_default_title_anchor_used",
                    routing.get("hybrid_default_title_anchor_used", False),
                )
            ),
            "hybrid_default_title_anchor_hits": int(
                execution.get(
                    "hybrid_default_title_anchor_hits",
                    routing.get("hybrid_default_title_anchor_hits", 0),
                )
            ),
            "hybrid_default_doc_aggregation_used": bool(
                execution.get(
                    "hybrid_default_doc_aggregation_used",
                    routing.get("hybrid_default_doc_aggregation_used", False),
                )
            ),
            "hybrid_default_doc_score_top1": float(
                execution.get(
                    "hybrid_default_doc_score_top1",
                    routing.get("hybrid_default_doc_score_top1", 0.0),
                )
            ),
            "hybrid_default_doc_score_top2": float(
                execution.get(
                    "hybrid_default_doc_score_top2",
                    routing.get("hybrid_default_doc_score_top2", 0.0),
                )
            ),
            "hybrid_default_doc_score_margin": float(
                execution.get(
                    "hybrid_default_doc_score_margin",
                    routing.get("hybrid_default_doc_score_margin", 0.0),
                )
            ),
            "hybrid_default_doc_top1_family": str(
                execution.get(
                    "hybrid_default_doc_top1_family",
                    routing.get("hybrid_default_doc_top1_family", ""),
                )
            ),
            "hybrid_default_doc_top2_family": str(
                execution.get(
                    "hybrid_default_doc_top2_family",
                    routing.get("hybrid_default_doc_top2_family", ""),
                )
            ),
            "hybrid_default_vector_used_for_doc_recall": bool(
                execution.get(
                    "hybrid_default_vector_used_for_doc_recall",
                    routing.get("hybrid_default_vector_used_for_doc_recall", False),
                )
            ),
            "hybrid_default_vector_support_same_family": bool(
                execution.get(
                    "hybrid_default_vector_support_same_family",
                    routing.get("hybrid_default_vector_support_same_family", False),
                )
            ),
            "hybrid_default_genericity_penalty_top1": float(
                execution.get(
                    "hybrid_default_genericity_penalty_top1",
                    routing.get("hybrid_default_genericity_penalty_top1", 0.0),
                )
            ),
            "hybrid_default_genericity_penalty_applied_count": int(
                execution.get(
                    "hybrid_default_genericity_penalty_applied_count",
                    routing.get("hybrid_default_genericity_penalty_applied_count", 0),
                )
            ),
            "hybrid_default_focus_rerank_used": bool(
                execution.get(
                    "hybrid_default_focus_rerank_used",
                    routing.get("hybrid_default_focus_rerank_used", False),
                )
            ),
            "hybrid_default_focus_rerank_candidate_chunks": int(
                execution.get(
                    "hybrid_default_focus_rerank_candidate_chunks",
                    routing.get("hybrid_default_focus_rerank_candidate_chunks", 0),
                )
            ),
            "hybrid_default_focus_heading_match_top1": float(
                execution.get(
                    "hybrid_default_focus_heading_match_top1",
                    routing.get("hybrid_default_focus_heading_match_top1", 0.0),
                )
            ),
            "hybrid_default_focus_actor_action_top1": float(
                execution.get(
                    "hybrid_default_focus_actor_action_top1",
                    routing.get("hybrid_default_focus_actor_action_top1", 0.0),
                )
            ),
            "hybrid_default_focus_concept_match_top1": float(
                execution.get(
                    "hybrid_default_focus_concept_match_top1",
                    routing.get("hybrid_default_focus_concept_match_top1", 0.0),
                )
            ),
            "hybrid_default_selected_family_keys": list(
                execution.get(
                    "hybrid_default_selected_family_keys",
                    routing.get("hybrid_default_selected_family_keys", []),
                )
            ),
            "hybrid_default_selected_doc_ids": list(
                execution.get(
                    "hybrid_default_selected_doc_ids",
                    routing.get("hybrid_default_selected_doc_ids", []),
                )
            ),
            "hybrid_default_family_search_used": bool(
                execution.get(
                    "hybrid_default_family_search_used",
                    routing.get("hybrid_default_family_search_used", False),
                )
            ),
            "hybrid_default_family_search_hits": int(
                execution.get(
                    "hybrid_default_family_search_hits",
                    routing.get("hybrid_default_family_search_hits", 0),
                )
            ),
            "hybrid_default_family_candidate_count_pre_filter": int(
                execution.get(
                    "hybrid_default_family_candidate_count_pre_filter",
                    routing.get("hybrid_default_family_candidate_count_pre_filter", 0),
                )
            ),
            "hybrid_default_family_candidate_count_post_filter": int(
                execution.get(
                    "hybrid_default_family_candidate_count_post_filter",
                    routing.get("hybrid_default_family_candidate_count_post_filter", 0),
                )
            ),
            "hybrid_default_family_score_top1": float(
                execution.get(
                    "hybrid_default_family_score_top1",
                    routing.get("hybrid_default_family_score_top1", 0.0),
                )
            ),
            "hybrid_default_family_score_top2": float(
                execution.get(
                    "hybrid_default_family_score_top2",
                    routing.get("hybrid_default_family_score_top2", 0.0),
                )
            ),
            "hybrid_default_family_score_margin": float(
                execution.get(
                    "hybrid_default_family_score_margin",
                    routing.get("hybrid_default_family_score_margin", 0.0),
                )
            ),
            "hybrid_default_family_identity_score_top1": float(
                execution.get(
                    "hybrid_default_family_identity_score_top1",
                    routing.get("hybrid_default_family_identity_score_top1", 0.0),
                )
            ),
            "hybrid_default_family_identity_score_top2": float(
                execution.get(
                    "hybrid_default_family_identity_score_top2",
                    routing.get("hybrid_default_family_identity_score_top2", 0.0),
                )
            ),
            "hybrid_default_family_identity_margin": float(
                execution.get(
                    "hybrid_default_family_identity_margin",
                    routing.get("hybrid_default_family_identity_margin", 0.0),
                )
            ),
            "hybrid_default_family_top1_doc_role": str(
                execution.get(
                    "hybrid_default_family_top1_doc_role",
                    routing.get("hybrid_default_family_top1_doc_role", ""),
                )
            ),
            "hybrid_default_family_top1_is_implementation": bool(
                execution.get(
                    "hybrid_default_family_top1_is_implementation",
                    routing.get("hybrid_default_family_top1_is_implementation", False),
                )
            ),
            "hybrid_default_family_recovery_used": bool(
                execution.get(
                    "hybrid_default_family_recovery_used",
                    routing.get("hybrid_default_family_recovery_used", False),
                )
            ),
            "hybrid_default_family_recovery_promoted": int(
                execution.get(
                    "hybrid_default_family_recovery_promoted",
                    routing.get("hybrid_default_family_recovery_promoted", 0),
                )
            ),
            "hybrid_default_weak_query_abort_used": bool(
                execution.get(
                    "hybrid_default_weak_query_abort_used",
                    routing.get("hybrid_default_weak_query_abort_used", False),
                )
            ),
            "hybrid_default_no_confident_family_candidates": bool(
                execution.get(
                    "hybrid_default_no_confident_family_candidates",
                    routing.get("hybrid_default_no_confident_family_candidates", False),
                )
            ),
            "hybrid_default_garbage_family_rejected": bool(
                execution.get(
                    "hybrid_default_garbage_family_rejected",
                    routing.get("hybrid_default_garbage_family_rejected", False),
                )
            ),
            "hybrid_default_garbage_family_reject_reason": str(
                execution.get(
                    "hybrid_default_garbage_family_reject_reason",
                    routing.get("hybrid_default_garbage_family_reject_reason", ""),
                )
            ),
            "hybrid_default_trusted_family": bool(
                execution.get(
                    "hybrid_default_trusted_family",
                    routing.get("hybrid_default_trusted_family", False),
                )
            ),
            "hybrid_default_shortlist_after_reject_count": int(
                execution.get(
                    "hybrid_default_shortlist_after_reject_count",
                    routing.get("hybrid_default_shortlist_after_reject_count", 0),
                )
            ),
            "hybrid_default_shortlist_after_reject_empty": bool(
                execution.get(
                    "hybrid_default_shortlist_after_reject_empty",
                    routing.get("hybrid_default_shortlist_after_reject_empty", False),
                )
            ),
            "hybrid_default_doc_role_prior_applied": bool(
                execution.get(
                    "hybrid_default_doc_role_prior_applied",
                    routing.get("hybrid_default_doc_role_prior_applied", False),
                )
            ),
            "hybrid_default_doc_role_prior_top1": str(
                execution.get(
                    "hybrid_default_doc_role_prior_top1",
                    routing.get("hybrid_default_doc_role_prior_top1", ""),
                )
            ),
            "hybrid_default_wrong_doc_role_penalty_applied": bool(
                execution.get(
                    "hybrid_default_wrong_doc_role_penalty_applied",
                    routing.get("hybrid_default_wrong_doc_role_penalty_applied", False),
                )
            ),
            "hybrid_default_implementation_bridge_used": bool(
                execution.get(
                    "hybrid_default_implementation_bridge_used",
                    routing.get("hybrid_default_implementation_bridge_used", False),
                )
            ),
            "hybrid_default_implementation_bridge_hits": int(
                execution.get(
                    "hybrid_default_implementation_bridge_hits",
                    routing.get("hybrid_default_implementation_bridge_hits", 0),
                )
            ),
            "hybrid_default_focus_rerank_blocked_by_low_family_confidence": bool(
                execution.get(
                    "hybrid_default_focus_rerank_blocked_by_low_family_confidence",
                    routing.get("hybrid_default_focus_rerank_blocked_by_low_family_confidence", False),
                )
            ),
            "hybrid_default_focus_rerank_stage_entered_with_family_confidence": float(
                execution.get(
                    "hybrid_default_focus_rerank_stage_entered_with_family_confidence",
                    routing.get("hybrid_default_focus_rerank_stage_entered_with_family_confidence", 0.0),
                )
            ),
            "rank_first_family_pre_focus": int(
                execution.get(
                    "rank_first_family_pre_focus",
                    routing.get("rank_first_family_pre_focus", 0),
                )
            ),
            "rank_first_family_post_focus": int(
                execution.get(
                    "rank_first_family_post_focus",
                    routing.get("rank_first_family_post_focus", 0),
                )
            ),
            "rank_first_doc_role_pre_focus": str(
                execution.get(
                    "rank_first_doc_role_pre_focus",
                    routing.get("rank_first_doc_role_pre_focus", ""),
                )
            ),
            "rank_first_doc_role_post_focus": str(
                execution.get(
                    "rank_first_doc_role_post_focus",
                    routing.get("rank_first_doc_role_post_focus", ""),
                )
            ),
            "hybrid_default_scope_fallback_used": bool(
                execution.get(
                    "hybrid_default_scope_fallback_used",
                    routing.get("hybrid_default_scope_fallback_used", False),
                )
            ),
            "hybrid_default_scope_fallback_reason": str(
                execution.get(
                    "hybrid_default_scope_fallback_reason",
                    routing.get("hybrid_default_scope_fallback_reason", ""),
                )
            ),
            "hybrid_default_scope_fallback_hits": int(
                execution.get(
                    "hybrid_default_scope_fallback_hits",
                    routing.get("hybrid_default_scope_fallback_hits", 0),
                )
            ),
            "hybrid_default_scope_fallback_bm25_only_used": bool(
                execution.get(
                    "hybrid_default_scope_fallback_bm25_only_used",
                    routing.get("hybrid_default_scope_fallback_bm25_only_used", False),
                )
            ),
            "hybrid_default_scope_fallback_bm25_only_hits": int(
                execution.get(
                    "hybrid_default_scope_fallback_bm25_only_hits",
                    routing.get("hybrid_default_scope_fallback_bm25_only_hits", 0),
                )
            ),
            "hybrid_default_scope_fallback_bm25_only_latency_ms": int(
                execution.get(
                    "hybrid_default_scope_fallback_bm25_only_latency_ms",
                    routing.get("hybrid_default_scope_fallback_bm25_only_latency_ms", 0),
                )
            ),
            "hybrid_default_scope_fallback_bm25_only_query": str(
                execution.get(
                    "hybrid_default_scope_fallback_bm25_only_query",
                    routing.get("hybrid_default_scope_fallback_bm25_only_query", ""),
                )
            ),
            "hybrid_default_scope_fallback_bm25_only_query_reason": str(
                execution.get(
                    "hybrid_default_scope_fallback_bm25_only_query_reason",
                    routing.get("hybrid_default_scope_fallback_bm25_only_query_reason", ""),
                )
            ),
            "hybrid_default_scope_fallback_vector_reused": bool(
                execution.get(
                    "hybrid_default_scope_fallback_vector_reused",
                    routing.get("hybrid_default_scope_fallback_vector_reused", False),
                )
            ),
            "hybrid_default_scope_fallback_second_vector_called": bool(
                execution.get(
                    "hybrid_default_scope_fallback_second_vector_called",
                    routing.get("hybrid_default_scope_fallback_second_vector_called", False),
                )
            ),
            "hybrid_default_scope_fallback_first_pass_hits": int(
                execution.get(
                    "hybrid_default_scope_fallback_first_pass_hits",
                    routing.get("hybrid_default_scope_fallback_first_pass_hits", 0),
                )
            ),
            "hybrid_default_scope_fallback_first_pass_top10": list(fallback_first_pass_top10),
            "hybrid_default_scope_fallback_added_law10": int(fallback_added_law10),
            "hybrid_default_scope_fallback_added_article10": int(fallback_added_article10),
            "hybrid_default_scope_fallback_added_focus1": int(fallback_added_focus1),
            "hybrid_anchor_scope_used": bool(
                execution.get("hybrid_anchor_scope_used", routing.get("hybrid_anchor_scope_used", False))
            ),
            "hybrid_anchor_scope_confidence": float(
                execution.get(
                    "hybrid_anchor_scope_confidence",
                    routing.get("hybrid_anchor_scope_confidence", 0.0),
                )
            ),
            "hybrid_anchor_scope_doc_ids_count": int(
                execution.get(
                    "hybrid_anchor_scope_doc_ids_count",
                    routing.get("hybrid_anchor_scope_doc_ids_count", 0),
                )
            ),
            "hybrid_anchor_scope_doc_ids": list(
                execution.get(
                    "hybrid_anchor_scope_doc_ids",
                    routing.get("hybrid_anchor_scope_doc_ids", []),
                )
            ),
            "hybrid_anchor_scope_reason": str(
                execution.get("hybrid_anchor_scope_reason", routing.get("hybrid_anchor_scope_reason", ""))
            ),
            "hybrid_doc_gate_applied": bool(
                execution.get("hybrid_doc_gate_applied", routing.get("hybrid_doc_gate_applied", False))
            ),
            "hybrid_doc_gate_bm25_before": int(
                execution.get("hybrid_doc_gate_bm25_before", routing.get("hybrid_doc_gate_bm25_before", 0))
            ),
            "hybrid_doc_gate_bm25_after": int(
                execution.get("hybrid_doc_gate_bm25_after", routing.get("hybrid_doc_gate_bm25_after", 0))
            ),
            "hybrid_doc_gate_vector_before": int(
                execution.get("hybrid_doc_gate_vector_before", routing.get("hybrid_doc_gate_vector_before", 0))
            ),
            "hybrid_doc_gate_vector_after": int(
                execution.get("hybrid_doc_gate_vector_after", routing.get("hybrid_doc_gate_vector_after", 0))
            ),
            "hybrid_doc_gate_removed_bm25": int(
                execution.get("hybrid_doc_gate_removed_bm25", routing.get("hybrid_doc_gate_removed_bm25", 0))
            ),
            "hybrid_doc_gate_removed_vector": int(
                execution.get("hybrid_doc_gate_removed_vector", routing.get("hybrid_doc_gate_removed_vector", 0))
            ),
            "hybrid_doc_gate_override_kept_original": bool(
                execution.get(
                    "hybrid_doc_gate_override_kept_original",
                    routing.get("hybrid_doc_gate_override_kept_original", False),
                )
            ),
            "hybrid_default_hard_doc_type_filter": bool(
                execution.get(
                    "hybrid_default_hard_doc_type_filter",
                    routing.get("hybrid_default_hard_doc_type_filter", False),
                )
            ),
            "hybrid_default_hard_doc_type_allowlist": list(
                execution.get(
                    "hybrid_default_hard_doc_type_allowlist",
                    routing.get("hybrid_default_hard_doc_type_allowlist", []),
                )
            ),
            "hybrid_default_concept_policy_applied": bool(
                execution.get(
                    "hybrid_default_concept_policy_applied",
                    routing.get("hybrid_default_concept_policy_applied", False),
                )
            ),
            "hybrid_default_concept_vector_optional": bool(
                execution.get(
                    "hybrid_default_concept_vector_optional",
                    routing.get("hybrid_default_concept_vector_optional", False),
                )
            ),
            "hybrid_default_intra_doc_focus_rerank_used": bool(
                execution.get(
                    "hybrid_default_intra_doc_focus_rerank_used",
                    routing.get("hybrid_default_intra_doc_focus_rerank_used", False),
                )
            ),
            "hybrid_default_intra_doc_focus_target_doc_ids": list(
                execution.get(
                    "hybrid_default_intra_doc_focus_target_doc_ids",
                    routing.get("hybrid_default_intra_doc_focus_target_doc_ids", []),
                )
            ),
            "hybrid_default_intra_doc_focus_target_family_keys": list(
                execution.get(
                    "hybrid_default_intra_doc_focus_target_family_keys",
                    routing.get("hybrid_default_intra_doc_focus_target_family_keys", []),
                )
            ),
            "hybrid_default_intra_doc_focus_rank_first_doc": int(
                execution.get(
                    "hybrid_default_intra_doc_focus_rank_first_doc",
                    routing.get("hybrid_default_intra_doc_focus_rank_first_doc", 0),
                )
            ),
            "hybrid_default_intra_doc_focus_rank_first_family": int(
                execution.get(
                    "hybrid_default_intra_doc_focus_rank_first_family",
                    routing.get("hybrid_default_intra_doc_focus_rank_first_family", 0),
                )
            ),
            "hybrid_default_intra_doc_focus_candidates": int(
                execution.get(
                    "hybrid_default_intra_doc_focus_candidates",
                    routing.get("hybrid_default_intra_doc_focus_candidates", 0),
                )
            ),
            "hybrid_default_intra_doc_focus_promoted": int(
                execution.get(
                    "hybrid_default_intra_doc_focus_promoted",
                    routing.get("hybrid_default_intra_doc_focus_promoted", 0),
                )
            ),
            "document_lookup_candidate_doc_ids_original_count": int(
                execution.get(
                    "document_lookup_candidate_doc_ids_original_count",
                    routing.get("document_lookup_candidate_doc_ids_original_count", 0),
                )
            ),
            "document_lookup_candidate_doc_ids_trimmed": bool(
                execution.get(
                    "document_lookup_candidate_doc_ids_trimmed",
                    routing.get("document_lookup_candidate_doc_ids_trimmed", False),
                )
            ),
            "candidate_route_early_exit_used": bool(
                execution.get(
                    "candidate_route_early_exit_used",
                    routing.get("candidate_route_early_exit_used", False),
                )
            ),
            "candidate_route_early_exit_reason": str(
                execution.get(
                    "candidate_route_early_exit_reason",
                    routing.get("candidate_route_early_exit_reason", ""),
                )
            ),
            "candidate_route_early_exit_score_gap": float(
                execution.get(
                    "candidate_route_early_exit_score_gap",
                    routing.get("candidate_route_early_exit_score_gap", 0.0),
                )
            ),
            "parser_document_number": legal_refs.get("document_number", ""),
            "parser_document_loose": legal_refs.get("document_loose", ""),
            "parser_document_short": legal_refs.get("document_short", ""),
            "parser_article_number": legal_refs.get("article_number", ""),
            "parser_clause_number": legal_refs.get("clause_number", ""),
            "parser_law_name": legal_refs.get("law_name", ""),
            "bm25_tokenized_query": bm25_debug.get("tokenized_query", ""),
            "bm25_parsed_query": bm25_debug.get("parsed_query", ""),
            "bm25_query_fields": list(bm25_debug.get("query_fields", []) or []),
            "bm25_is_multifield": bool(bm25_debug.get("is_multifield", False)),
            "bm25_anchor_tokens": list(bm25_debug.get("anchor_tokens", []) or []),
            "bm25_intent_tokens": list(bm25_debug.get("intent_tokens", []) or []),
            "narrow_doc_number_raw": bm25_debug.get("narrow_doc_number_raw", ""),
            "narrow_doc_number_canonical": bm25_debug.get("narrow_doc_number_canonical", ""),
            "narrow_doc_number_loose_key": bm25_debug.get("narrow_doc_number_loose_key", ""),
            "narrow_doc_number_exact_candidates_count": int(
                bm25_debug.get("narrow_doc_number_exact_candidates_count", 0)
            ),
            "narrow_doc_number_exact_candidates": list(
                bm25_debug.get("narrow_doc_number_exact_candidates", [])
            ),
            "narrow_doc_number_exact_pass_used": bool(
                bm25_debug.get("narrow_doc_number_exact_pass_used", False)
            ),
            "narrow_doc_number_exact_pass_hits": int(
                bm25_debug.get("narrow_doc_number_exact_pass_hits", 0)
            ),
            "narrow_doc_number_bias_applied": bool(
                bm25_debug.get("narrow_doc_number_bias_applied", False)
            ),
            "narrow_doc_number_top1_match_level": bm25_debug.get(
                "narrow_doc_number_top1_match_level",
                "",
            ),
            "narrow_doc_number_mismatch_reason": bm25_debug.get(
                "narrow_doc_number_mismatch_reason",
                "",
            ),
            "rerank_top1_doc_id": rerank_top1["doc_id"],
            "rerank_top1_document_number": rerank_top1["document_number"],
            "rerank_top1_article": rerank_top1["article"],
            "rerank_top1_clause": rerank_top1["clause"],
            "rerank_top1_path": rerank_top1["path"],
            "rerank_top1_family_key": rerank_top1_family_key,
            "rerank_top2_path": rerank_top2["path"],
            "rerank_top3_path": rerank_top3["path"],
            "bm25_top_hits_debug": bm25_top_hits_debug,
            "vector_top_hits_debug": vector_top_hits_debug,
            "rerank_top_hits_debug": rerank_top_hits_debug,
            **quality,
        }
        failure_labels, primary_failure = _classify_failure(case, row, routing)
        true_failure_labels = [
            label
            for label in failure_labels
            if label not in {"PASS"} and label not in ACCEPTABLE_ROUTE_PROMOTION_LABELS
        ]
        primary_true_failure = true_failure_labels[0] if true_failure_labels else "PASS"
        acceptable_route_promotion = any(
            label in ACCEPTABLE_ROUTE_PROMOTION_LABELS for label in failure_labels
        )
        row["failure_labels"] = failure_labels
        row["primary_failure_label"] = primary_failure
        row["true_failure_labels"] = true_failure_labels or ["PASS"]
        row["primary_true_failure_label"] = primary_true_failure
        row["acceptable_route_promotion"] = int(acceptable_route_promotion)
        row["acceptable_alias_tie"] = int("ROUTE_ACCEPTABLE_ALIAS_TIE" in failure_labels)
        row["acceptable_scope_promotion"] = int("ROUTE_ACCEPTABLE_SCOPE_PROMOTION" in failure_labels)
        row["is_true_failure"] = int(primary_true_failure != "PASS")
        row["is_true_focus_failure"] = int(
            any(label in FOCUS_FAILURE_LABELS for label in true_failure_labels)
        )
        row["route_semantic_correct"] = int(
            bool(row.get("route_correct", 0))
            or any(label in ACCEPTABLE_ROUTE_PROMOTION_LABELS for label in failure_labels)
        )
        rows.append(row)

        print(
            f"[{label} {idx:02d}/{len(cases)}] {row['case_id']} "
            f"route={row['actual_route']} expected={row['expected_route']} "
            f"retrieval={row['retrieval_latency_ms']}ms "
            f"law={row['correct_law_top10']} article={row['correct_article_top10']} focus={row['focus_top1']}"
        )

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    try:
        retriever._bm25.close()
    except Exception:
        pass
    return rows, elapsed_ms


def _interleave_cases_by_route(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("expected_route", ""))].append(case)

    out: list[dict[str, Any]] = []
    while any(grouped.get(route) for route in EXPECTED_ROUTES):
        for route in EXPECTED_ROUTES:
            if grouped.get(route):
                out.append(grouped[route].pop(0))
    return out


def _apply_cfg_overrides(overrides: dict[str, Any]) -> dict[str, tuple[bool, Any]]:
    backup: dict[str, tuple[bool, Any]] = {}
    for key, value in overrides.items():
        existed = hasattr(cfg, key)
        backup[key] = (existed, getattr(cfg, key, None))
        setattr(cfg, key, value)
    return backup


def _restore_cfg_overrides(backup: dict[str, tuple[bool, Any]]) -> None:
    for key, state in backup.items():
        existed, old_value = state
        if existed:
            setattr(cfg, key, old_value)
        elif hasattr(cfg, key):
            delattr(cfg, key)


def _run_hybrid_default_ablation(
    *,
    cases: list[dict[str, Any]],
    debug_top_k: int,
) -> dict[str, Any]:
    target_cases = [c for c in cases if str(c.get("expected_route", "")) == "hybrid_default"][:15]
    if not target_cases:
        return {
            "enabled": False,
            "reason": "no_hybrid_default_cases",
            "case_count": 0,
            "variants": [],
        }

    variants = [
        {
            "id": "A",
            "name": "v14_baseline",
            "overrides": {
                "HYBRID_DEFAULT_V15_VAGUE_GATE_ENABLED": False,
                "HYBRID_DEFAULT_V15_GARBAGE_REJECTION_ENABLED": False,
                "HYBRID_DEFAULT_V15_FAMILY_IDENTITY_ENABLED": False,
            },
        },
        {
            "id": "B",
            "name": "v14_plus_vague_gate",
            "overrides": {
                "HYBRID_DEFAULT_V15_VAGUE_GATE_ENABLED": True,
                "HYBRID_DEFAULT_V15_GARBAGE_REJECTION_ENABLED": False,
                "HYBRID_DEFAULT_V15_FAMILY_IDENTITY_ENABLED": False,
            },
        },
        {
            "id": "C",
            "name": "B_plus_garbage_reject",
            "overrides": {
                "HYBRID_DEFAULT_V15_VAGUE_GATE_ENABLED": True,
                "HYBRID_DEFAULT_V15_GARBAGE_REJECTION_ENABLED": True,
                "HYBRID_DEFAULT_V15_FAMILY_IDENTITY_ENABLED": False,
            },
        },
        {
            "id": "D",
            "name": "C_plus_family_identity",
            "overrides": {
                "HYBRID_DEFAULT_V15_VAGUE_GATE_ENABLED": True,
                "HYBRID_DEFAULT_V15_GARBAGE_REJECTION_ENABLED": True,
                "HYBRID_DEFAULT_V15_FAMILY_IDENTITY_ENABLED": True,
            },
        },
    ]

    out_variants: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        backup = _apply_cfg_overrides(variant["overrides"])
        try:
            rows, elapsed = _run_benchmark(
                label=f"ablation_{variant['id']}",
                cases=target_cases,
                warmup=(idx == 0),
                debug_top_k=max(3, int(debug_top_k)),
            )
        finally:
            _restore_cfg_overrides(backup)

        summary = _build_run_summary(rows, elapsed)
        route_summary = (summary.get("route_summary") or {}).get("hybrid_default", {})
        quality = route_summary.get("quality", {})
        latency = (route_summary.get("latency_ms") or {}).get("retrieval", {})
        diagnostics = route_summary.get("hybrid_default_stage_diagnostics", {})
        mode_dist = (
            (summary.get("hybrid_default_acceptance_gate_v15") or {}).get("mode_distribution")
            or {}
        )
        out_variants.append(
            {
                "id": variant["id"],
                "name": variant["name"],
                "overrides": dict(variant["overrides"]),
                "summary": {
                    "count": len(rows),
                    "correct_law_top10_rate": float(quality.get("correct_law_top10_rate", 0.0)),
                    "law_found_top10_rate": float(quality.get("law_found_top10_rate", 0.0)),
                    "focus_top1_rate": float(quality.get("focus_top1_rate", 0.0)),
                    "topical_doc_top10_rate": float(quality.get("topical_doc_top10_rate", 0.0)),
                    "answer_grounded_rate": float(quality.get("answer_grounded_rate", 0.0)),
                    "p95_retrieval_ms": int(latency.get("p95", 0)),
                    "garbage_family_rejected_count": int(
                        diagnostics.get("garbage_family_rejected_count", 0)
                    ),
                    "shortlist_after_reject_zero_count": int(
                        diagnostics.get("shortlist_after_reject_zero_count", 0)
                    ),
                    "shortlist_after_reject_nonzero_count": int(
                        diagnostics.get("shortlist_after_reject_nonzero_count", 0)
                    ),
                    "trusted_family_count": int(diagnostics.get("trusted_family_count", 0)),
                    "untrusted_family_count": int(diagnostics.get("untrusted_family_count", 0)),
                    "mode_distribution": dict(mode_dist),
                },
                "rows": rows,
            }
        )

    return {
        "enabled": True,
        "case_count": len(target_cases),
        "case_ids": [str(case.get("case_id", "")) for case in target_cases],
        "variants": out_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-in", default="")
    parser.add_argument("--cases-out", default="processed/eval_cases_patch_v4_expanded.json")
    parser.add_argument("--summary-out", default="logs/benchmark_expanded_summary.json")
    parser.add_argument("--results-out", default="logs/benchmark_expanded_results.json")
    parser.add_argument("--compare-out", default="logs/benchmark_expanded_compare_exact.json")
    parser.add_argument("--failure-csv-out", default="logs/benchmark_expanded_failures.csv")
    parser.add_argument("--failure-debug-out", default="logs/benchmark_expanded_failure_debug.jsonl")
    parser.add_argument("--debug-top-k", type=int, default=5)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--no-sync-db", action="store_true")
    parser.add_argument("--skip-hybrid-ablation", action="store_true")
    args = parser.parse_args()

    db_path = Path("logs/observability.sqlite3")
    docs_path = Path("processed/documents.jsonl")
    chunks_path = Path("processed/chunks.jsonl")
    docs_map, law_candidates = _load_documents(docs_path)
    global DOC_META_BY_ID
    DOC_META_BY_ID = {_norm_text(doc_id): dict(meta) for doc_id, meta in docs_map.items()}

    reused_cases_file = ""
    if args.cases_in:
        in_path = Path(args.cases_in)
        if not in_path.exists():
            raise SystemExit(f"--cases-in file not found: {in_path}")
        expanded_cases = _load_cases_file(in_path)
        reused_cases_file = str(in_path)
    else:
        base_exact = _load_base_exact_cases(db_path=db_path, limit=20)
        if len(base_exact) < 20:
            raise SystemExit("Need at least 20 base exact cases in eval_cases (debug_bm25_*)")

        exact_targets: set[tuple[str, str, str]] = set()
        for row in base_exact:
            doc_id = str(row.get("expected_doc_id", ""))
            if not doc_id:
                continue
            article_num, clause_num = _parse_expected_article_clause(row.get("expected_article_ref", ""))
            exact_targets.add((doc_id, article_num, clause_num))
            if article_num:
                exact_targets.add((doc_id, article_num, ""))

        law_doc_ids = {str(meta.get("doc_id", "")) for meta in law_candidates[:80] if meta.get("doc_id")}
        exact_samples, law_samples = _collect_chunk_samples(
            chunks_path=chunks_path,
            exact_targets=exact_targets,
            law_doc_ids=law_doc_ids,
        )

        expanded_cases = _build_expanded_cases(
            base_exact=base_exact,
            docs_map=docs_map,
            law_candidates=law_candidates,
            exact_samples=exact_samples,
            law_samples=law_samples,
        )
        expanded_cases = _interleave_cases_by_route(expanded_cases)

    if len(expanded_cases) < 65:
        raise SystemExit(
            f"Expanded eval set too small: {len(expanded_cases)} (need at least 65). "
            "Need more valid law_anchored/hybrid cases."
        )

    route_counts = Counter(str(c.get("expected_route", "")) for c in expanded_cases)
    for route, min_count in CASE_MIN_COUNTS.items():
        if route_counts.get(route, 0) < int(min_count):
            raise SystemExit(
                f"Route '{route}' has too few cases: {route_counts.get(route, 0)} (need >= {min_count})."
            )

    cases_out = Path(args.cases_out)
    cases_out.parent.mkdir(parents=True, exist_ok=True)
    cases_out.write_text(json.dumps(expanded_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_sync_db:
        _sync_eval_cases_to_db(db_path, expanded_cases)

    warm_rows, warm_elapsed = _run_benchmark(
        label="warm",
        cases=expanded_cases,
        warmup=True,
        debug_top_k=max(3, int(args.debug_top_k)),
    )
    cold_rows, cold_elapsed = _run_benchmark(
        label="coldish",
        cases=expanded_cases,
        warmup=False,
        debug_top_k=max(3, int(args.debug_top_k)),
    )

    warm_summary = _build_run_summary(warm_rows, warm_elapsed)
    cold_summary = _build_run_summary(cold_rows, cold_elapsed)
    hybrid_ablation_payload: dict[str, Any] = {
        "enabled": False,
        "reason": "skipped_by_flag",
        "case_count": 0,
        "variants": [],
    }
    if not args.skip_hybrid_ablation:
        hybrid_ablation_payload = _run_hybrid_default_ablation(
            cases=expanded_cases,
            debug_top_k=max(3, int(args.debug_top_k)),
        )

    summary_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_cases_file": str(cases_out),
        "reused_cases_file": reused_cases_file,
        "source_cases_count": len(expanded_cases),
        "route_distribution": dict(route_counts),
        "config": {
            "bm25_top_k": cfg.BM25_TOP_K,
            "vector_top_k": cfg.VECTOR_TOP_K,
            "final_top_k": max(10, cfg.HYBRID_TOP_K),
            "vector_timeout_ms": cfg.VECTOR_TIMEOUT_MS,
            "vector_skip_by_quality_gate": cfg.VECTOR_SKIP_BY_QUALITY_GATE,
        },
        "runs": {
            "warm": warm_summary,
            "coldish": cold_summary,
        },
        "hybrid_default_ablation_v15": hybrid_ablation_payload,
    }

    results_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cases_file": str(cases_out),
        "cases": expanded_cases,
        "runs": {
            "warm": warm_rows,
            "coldish": cold_rows,
        },
        "hybrid_default_ablation_v15": hybrid_ablation_payload,
    }

    baseline_payload = None
    baseline_path = _resolve_exact_baseline(args.baseline or None)
    if baseline_path and baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    warm_exact_metrics = _extract_exact_metrics(
        {
            "route_summary": {
                "structured_exact": (warm_summary.get("route_summary") or {}).get("structured_exact", {}),
            },
            "summary": warm_summary,
        }
    )
    cold_exact_metrics = _extract_exact_metrics(
        {
            "route_summary": {
                "structured_exact": (cold_summary.get("route_summary") or {}).get("structured_exact", {}),
            },
            "summary": cold_summary,
        }
    )

    compare_payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "baseline_file": str(baseline_path) if baseline_path else "",
        "expanded_run_compare": {
            "warm_vs_coldish": _compare_metrics(
                {
                    "p95_retrieval_ms": float(((warm_summary.get("latency_ms") or {}).get("retrieval") or {}).get("p95", 0)),
                    "correct_article_top10_rate": float((warm_summary.get("quality") or {}).get("correct_article_top10_rate", 0.0)),
                    "focus_top1_rate": float((warm_summary.get("quality") or {}).get("focus_top1_rate", 0.0)),
                },
                {
                    "p95_retrieval_ms": float(((cold_summary.get("latency_ms") or {}).get("retrieval") or {}).get("p95", 0)),
                    "correct_article_top10_rate": float((cold_summary.get("quality") or {}).get("correct_article_top10_rate", 0.0)),
                    "focus_top1_rate": float((cold_summary.get("quality") or {}).get("focus_top1_rate", 0.0)),
                },
            ),
        },
        "exact_route_compare": {},
    }

    if baseline_payload:
        baseline_exact_metrics = _extract_exact_metrics(baseline_payload)
        compare_payload["exact_route_compare"] = {
            "warm_vs_baseline_exact": _compare_metrics(warm_exact_metrics, baseline_exact_metrics),
            "coldish_vs_baseline_exact": _compare_metrics(cold_exact_metrics, baseline_exact_metrics),
        }

    summary_out = Path(args.summary_out)
    results_out = Path(args.results_out)
    compare_out = Path(args.compare_out)
    failure_csv_out = Path(args.failure_csv_out)
    failure_debug_out = Path(args.failure_debug_out)
    for p in (summary_out, results_out, compare_out, failure_csv_out, failure_debug_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    summary_out.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    results_out.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_out.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fail_csv_count, fail_debug_count = _write_failure_outputs(
        warm_rows=warm_rows,
        cold_rows=cold_rows,
        failure_csv_out=failure_csv_out,
        failure_debug_out=failure_debug_out,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_ts = summary_out.with_name(f"{summary_out.stem}_{ts}{summary_out.suffix}")
    results_ts = results_out.with_name(f"{results_out.stem}_{ts}{results_out.suffix}")
    compare_ts = compare_out.with_name(f"{compare_out.stem}_{ts}{compare_out.suffix}")
    failure_csv_ts = failure_csv_out.with_name(f"{failure_csv_out.stem}_{ts}{failure_csv_out.suffix}")
    failure_debug_ts = failure_debug_out.with_name(f"{failure_debug_out.stem}_{ts}{failure_debug_out.suffix}")
    summary_ts.write_text(summary_out.read_text(encoding="utf-8"), encoding="utf-8")
    results_ts.write_text(results_out.read_text(encoding="utf-8"), encoding="utf-8")
    compare_ts.write_text(compare_out.read_text(encoding="utf-8"), encoding="utf-8")
    failure_csv_ts.write_text(failure_csv_out.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    failure_debug_ts.write_text(failure_debug_out.read_text(encoding="utf-8"), encoding="utf-8")

    print("\nDONE")
    print(f"cases: {cases_out}")
    print(f"summary: {summary_out}")
    print(f"results: {results_out}")
    print(f"comparison: {compare_out}")
    print(f"failures_csv: {failure_csv_out} (rows={fail_csv_count})")
    print(f"failures_debug: {failure_debug_out} (rows={fail_debug_count})")
    print(
        "timestamped: "
        f"{summary_ts.name}, {results_ts.name}, {compare_ts.name}, "
        f"{failure_csv_ts.name}, {failure_debug_ts.name}"
    )
    print("\nROUTE DISTRIBUTION")
    print(json.dumps(dict(route_counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
