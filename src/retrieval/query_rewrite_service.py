"""
Query rewrite service for non-exact hybrid retrieval routes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.retrieval.text_tokenizer import tokenize_for_query

_SPACE_RE = re.compile(r"\s+")

_FILLER_PHRASES: tuple[str, ...] = (
    "quy dinh hien hanh",
    "quy dinh chung",
    "nhu the nao",
    "la gi",
    "trong thuc te",
    "khi ap dung thuc te",
    "cho toi biet",
    "tom tat",
    "mot cach ngan gon",
    "duoc xu ly nhu the nao",
    "nhung gi",
    "ra sao",
)

_FILLER_TOKENS = {
    "khi",
    "nhu",
    "the",
    "nao",
    "can",
    "nhung",
    "chung",
    "ap",
    "thuc",
    "te",
    "cho",
    "toi",
    "biet",
    "tom",
    "tat",
    "hien",
    "nay",
    "ra",
    "sao",
}

_GENERIC_LEGAL_TOKENS = {
    "quy",
    "dinh",
    "phap",
    "luat",
    "ve",
    "trong",
    "cac",
    "noi",
    "dung",
    "lien",
    "quan",
}

_LEXICAL_BLACKLIST_TOKENS = _FILLER_TOKENS | {
    "luu",
    "gi",
    "la",
    "ve",
    "quy",
    "dinh",
    "duoc",
    "theo",
    "cua",
    "va",
    "mot",
    "so",
}

_STOP_TOKENS = _LEXICAL_BLACKLIST_TOKENS | _GENERIC_LEGAL_TOKENS

_LEGAL_KEEP_TOKENS = {
    "dung",
    "co",
    "vu",
    "nghia",
    "quyen",
    "tuyen",
    "cong",
    "chuc",
    "lao",
    "dong",
    "hop",
    "tham",
    "pham",
    "vi",
    "hanh",
    "chinh",
    "an",
    "toan",
    "hieu",
    "luc",
    "trach",
    "nhiem",
    "thu",
    "tuc",
    "xet",
    "xu",
    "ly",
}

_PHRASE_REPAIRS: tuple[tuple[str, str], ...] = (
    (r"\bnhiem\s+quyen\s+han\b", "nhiem vu quyen han"),
    (r"\btham\s+quyen\s+cong\b", "tham quyen"),
    (r"\bpham\s+chinh\b", "pham vi dieu chinh"),
    (r"\bbao\s+dam\s+toan\b", "bao dam an toan"),
    (r"\bquyen\s+nghia\s+ban\s+cong\s+chuc\b", "quyen va nghia vu co ban cua cong chuc"),
    (r"\bquyet\s+hieu\s+luc\b", "thoi diem hieu luc"),
    (r"\blap\s+kinh\b", "lap ke hoach"),
)

_AMBIGUOUS_PATTERNS: tuple[str, ...] = (
    r"\blap\s+quan\s+kinh\b",
    r"\bkiem\s+sat\s+giai\b",
    r"\bquyet\s+hieu\s+luc\b",
)

_UNSTABLE_BIGRAMS = {
    "lap quan",
    "quan kinh",
    "kiem sat",
    "sat giai",
    "cong chuc thanh",
}

_STRONG_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdon\s+phuong\s+cham\s+dut\s+hop\s+dong\b", "don phuong cham dut hop dong"),
    (r"\bdieu\s+kien\s+tuyen\s+dung\s+cong\s+chuc\b", "dieu kien tuyen dung cong chuc"),
    (
        r"\btham\s+quyen\s+xu\s+phat\s+vi\s+pham\s+hanh\s+chinh\b",
        "tham quyen xu phat vi pham hanh chinh",
    ),
    (r"\bquyen\s+va\s+nghia\s+vu\s+co\s+ban\s+cua\s+cong\s+chuc\b", "quyen va nghia vu co ban cua cong chuc"),
    (r"\bnhiem\s+vu\s+quyen\s+han\b", "nhiem vu quyen han"),
    (r"\bpham\s+vi\s+dieu\s+chinh\b", "pham vi dieu chinh"),
    (r"\bdoi\s+tuong\s+ap\s+dung\b", "doi tuong ap dung"),
    (r"\bhieu\s+luc\s+thi\s+hanh\b", "hieu luc thi hanh"),
    (r"\btham\s+quyen\b", "tham quyen"),
    (r"\bdieu\s+kien\b", "dieu kien"),
)

_ENTITY_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcong\s+chuc\b", "cong chuc"),
    (r"\bcan\s+bo\b", "can bo"),
    (r"\bvien\s+chuc\b", "vien chuc"),
    (r"\bthanh\s+tra\b", "thanh tra"),
    (r"\bnguoi\s+lao\s+dong\b", "nguoi lao dong"),
    (r"\bnguoi\s+su\s+dung\s+lao\s+dong\b", "nguoi su dung lao dong"),
    (r"\bco\s+quan\b", "co quan"),
    (r"\bto\s+chuc\b", "to chuc"),
    (r"\bca\s+nhan\b", "ca nhan"),
)

_PROCEDURAL_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdieu\s+kien\b", "dieu kien"),
    (r"\bthu\s+tuc\b", "thu tuc"),
    (r"\bthoi\s+han\b", "thoi han"),
    (r"\btrinh\s+tu\b", "trinh tu"),
    (r"\bho\s+so\b", "ho so"),
    (r"\btham\s+quyen\b", "tham quyen"),
)

_ACTOR_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcong\s+chuc\b", "cong chuc"),
    (r"\bcan\s+bo\b", "can bo"),
    (r"\bvien\s+chuc\b", "vien chuc"),
    (r"\bnguoi\s+lao\s+dong\b", "nguoi lao dong"),
    (r"\bnguoi\s+su\s+dung\s+lao\s+dong\b", "nguoi su dung lao dong"),
    (r"\bco\s+quan\b", "co quan"),
    (r"\bto\s+chuc\b", "to chuc"),
)

_ACTION_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\btham\s+quyen\s+xu\s+phat\b", "tham quyen xu phat"),
    (r"\bxu\s+phat\s+vi\s+pham\s+hanh\s+chinh\b", "xu phat vi pham hanh chinh"),
    (r"\bdon\s+phuong\s+cham\s+dut\b", "don phuong cham dut"),
    (r"\bdieu\s+kien\s+tuyen\s+dung\b", "dieu kien tuyen dung"),
    (r"\bquyen\s+va\s+nghia\s+vu\b", "quyen va nghia vu"),
    (r"\bnhiem\s+vu\s+quyen\s+han\b", "nhiem vu quyen han"),
    (r"\btrach\s+nhiem\b", "trach nhiem"),
    (r"\bhieu\s+luc\s+thi\s+hanh\b", "hieu luc thi hanh"),
)

_ACTION_TOKENS = {
    "tuyen",
    "dung",
    "xu",
    "phat",
    "cham",
    "dut",
    "tham",
    "quyen",
    "ky",
    "luat",
    "pham",
    "vi",
    "dieu",
    "chinh",
    "quyen",
    "nghia",
    "vu",
    "dieu",
    "kien",
    "thu",
    "tuc",
    "trinh",
    "tu",
    "hieu",
    "luc",
    "trach",
    "nhiem",
}

_ACTOR_TOKENS = {
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

_DOC_HINT_TOKENS = {"luat", "bo", "nghi", "dinh", "vbhn", "thong", "tu"}

_GENERIC_WEAK_PHRASES = {
    "trach nhiem",
    "bao dam an toan",
    "bao dam toan",
    "thoi diem hieu luc",
    "hieu luc",
    "lap ke hoach",
}

_WEAK_LEXICAL_ABORT_SINGLE_TOKENS = {
    "can",
    "chuc",
    "quan",
    "ve",
}

_WEAK_LEXICAL_ABORT_PHRASES = {
    "la gi",
    "quy dinh chung",
    "ve kiem sat giai",
}

_TITLE_ANCHOR_BLACKLIST_PHRASES = {
    "la gi",
    "quy dinh chung",
    "ve",
}

_CONCEPT_RULES: tuple[dict[str, Any], ...] = (
    {
        "tag": "pham_vi_dieu_chinh",
        "pattern": r"\bpham\s+vi\s+dieu\s+chinh\b",
        "expansions": (
            "pham vi dieu chinh",
            "doi tuong ap dung",
            "ap dung doi voi",
            "dieu chinh cac hoat dong",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "doi_tuong_ap_dung",
        "pattern": r"\bdoi\s+tuong\s+ap\s+dung\b",
        "expansions": (
            "doi tuong ap dung",
            "pham vi ap dung",
            "ap dung doi voi",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "hieu_luc_thi_hanh",
        "pattern": r"\b(?:thoi\s+diem\s+)?hieu\s+luc(?:\s+thi\s+hanh)?\b",
        "expansions": (
            "co hieu luc",
            "hieu luc thi hanh",
            "ngay co hieu luc",
            "thoi diem co hieu luc",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "trach_nhiem",
        "pattern": r"\btrach\s+nhiem\b",
        "expansions": (
            "trach nhiem cua",
            "co trach nhiem",
            "chiu trach nhiem",
            "nhiem vu quyen han",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "nhiem_vu_quyen_han",
        "pattern": r"\bnhiem\s+vu\s+quyen\s+han\b",
        "expansions": (
            "nhiem vu quyen han",
            "quyen va nghia vu",
            "co trach nhiem",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "quyen_va_nghia_vu",
        "pattern": r"\bquyen\s+va\s+nghia\s+vu\b",
        "expansions": (
            "quyen va nghia vu",
            "quyen cua",
            "nghia vu cua",
        ),
        "subclass": "concept_generic",
    },
    {
        "tag": "tham_quyen_xu_phat",
        "pattern": r"\btham\s+quyen\s+xu\s+phat(?:\s+vi\s+pham\s+hanh\s+chinh)?\b",
        "expansions": (
            "tham quyen xu phat",
            "xu phat vi pham hanh chinh",
            "nguoi co tham quyen",
            "quyet dinh xu phat",
        ),
        "subclass": "sanction_power",
    },
    {
        "tag": "xu_ly_ky_luat",
        "pattern": r"\bxu\s+ly\s+ky\s+luat\b",
        "expansions": (
            "xu ly ky luat",
            "hinh thuc ky luat",
            "trinh tu xu ly ky luat",
        ),
        "subclass": "subject_action",
    },
    {
        "tag": "dieu_kien_tuyen_dung",
        "pattern": r"\bdieu\s+kien\s+tuyen\s+dung\b",
        "expansions": (
            "dieu kien tuyen dung",
            "tieu chuan tuyen dung",
            "ho so tuyen dung",
        ),
        "subclass": "subject_action",
    },
    {
        "tag": "don_phuong_cham_dut_hop_dong",
        "pattern": r"\bdon\s+phuong\s+cham\s+dut\s+hop\s+dong\b",
        "expansions": (
            "don phuong cham dut hop dong",
            "truong hop cham dut hop dong",
            "thoi han bao truoc",
        ),
        "subclass": "subject_action",
    },
    {
        "tag": "trinh_tu_thu_tuc",
        "pattern": r"\btrinh\s+tu\s+thu\s+tuc\b",
        "expansions": (
            "trinh tu thu tuc",
            "ho so thu tuc",
            "co quan giai quyet",
        ),
        "subclass": "procedural",
    },
    {
        "tag": "thoi_han_giai_quyet",
        "pattern": r"\bthoi\s+han\s+giai\s+quyet\b",
        "expansions": (
            "thoi han giai quyet",
            "thoi han xu ly",
            "thoi han thuc hien",
        ),
        "subclass": "procedural",
    },
)

_CONCEPT_TAG_TO_EXPANSIONS: dict[str, tuple[str, ...]] = {
    str(rule["tag"]): tuple(rule.get("expansions") or ())
    for rule in _CONCEPT_RULES
}

_CONCEPT_TAG_TO_ANCHOR: dict[str, str] = {
    "tham_quyen_xu_phat": "xu ly vi pham hanh chinh",
    "xu_ly_ky_luat": "can bo cong chuc",
    "dieu_kien_tuyen_dung": "can bo cong chuc",
    "don_phuong_cham_dut_hop_dong": "lao dong",
    "quyen_va_nghia_vu": "can bo cong chuc",
}

_CONCEPT_TAG_PHRASES: dict[str, tuple[str, ...]] = {
    "pham_vi_dieu_chinh": (
        "pham vi dieu chinh",
        "doi tuong ap dung",
        "ap dung doi voi",
        "pham vi ap dung",
    ),
    "hieu_luc_thi_hanh": (
        "hieu luc",
        "hieu luc thi hanh",
        "co hieu luc",
        "thi hanh ke tu",
    ),
    "trach_nhiem": (
        "trach nhiem",
        "chiu trach nhiem",
        "co trach nhiem",
    ),
    "nhiem_vu_quyen_han": (
        "nhiem vu",
        "quyen han",
        "nhiem vu quyen han",
        "chuc nang nhiem vu quyen han",
    ),
    "quyen_va_nghia_vu": (
        "quyen va nghia vu",
        "quyen loi va nghia vu",
        "nghia vu",
        "quyen",
    ),
    "dieu_kien_tuyen_dung": (
        "dieu kien tuyen dung",
        "tieu chuan tuyen dung",
        "dieu kien du tuyen",
    ),
    "xu_ly_ky_luat": (
        "xu ly ky luat",
        "bi ky luat",
        "hinh thuc ky luat",
        "truong hop bi ky luat",
    ),
    "tham_quyen_xu_phat": (
        "tham quyen xu phat",
        "xu phat vi pham hanh chinh",
        "nguoi co tham quyen xu phat",
    ),
    "don_phuong_cham_dut_hop_dong": (
        "don phuong cham dut",
        "cham dut hop dong",
        "duoc quyen don phuong cham dut",
    ),
    "dieu_kien_huong": (
        "dieu kien huong",
        "duoc huong khi nao",
        "truong hop duoc huong",
    ),
    "thu_tuc_ho_so": (
        "thu tuc",
        "ho so",
        "trinh tu",
        "thanh phan ho so",
    ),
    "tham_quyen": (
        "tham quyen",
        "co quan co tham quyen",
        "quyet dinh boi ai",
    ),
}

_CONCEPT_EXPANSION_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "pham_vi_dieu_chinh": {
        "core": ("pham vi dieu chinh",),
        "expanded": ("pham vi dieu chinh", "doi tuong ap dung", "ap dung doi voi"),
        "title": ("pham vi dieu chinh", "doi tuong ap dung"),
        "semantic": ("quy dinh ve pham vi dieu chinh va doi tuong ap dung",),
    },
    "hieu_luc_thi_hanh": {
        "core": ("hieu luc thi hanh",),
        "expanded": ("hieu luc thi hanh", "co hieu luc ke tu", "thi hanh ke tu"),
        "title": ("hieu luc thi hanh",),
        "semantic": ("quy dinh ve thoi diem co hieu luc thi hanh cua van ban",),
    },
    "trach_nhiem": {
        "core": ("trach nhiem",),
        "expanded": ("trach nhiem", "co trach nhiem", "chiu trach nhiem"),
        "title": ("trach nhiem",),
        "semantic": ("quy dinh ve trach nhiem cua chu the lien quan",),
    },
    "nhiem_vu_quyen_han": {
        "core": ("nhiem vu quyen han",),
        "expanded": (
            "nhiem vu quyen han",
            "nhiem vu",
            "quyen han",
            "chuc nang nhiem vu quyen han",
        ),
        "title": ("nhiem vu quyen han",),
        "semantic": ("quy dinh ve nhiem vu va quyen han cua chu the",),
    },
    "quyen_va_nghia_vu": {
        "core": ("quyen va nghia vu",),
        "expanded": ("quyen va nghia vu", "nghia vu", "quyen co ban", "nghia vu co ban"),
        "title": ("quyen va nghia vu",),
        "semantic": ("quy dinh ve quyen va nghia vu cua chu the",),
    },
    "tham_quyen_xu_phat": {
        "core": ("tham quyen xu phat vi pham hanh chinh",),
        "expanded": (
            "tham quyen xu phat",
            "xu phat vi pham hanh chinh",
            "nguoi co tham quyen xu phat",
        ),
        "title": ("xu phat vi pham hanh chinh", "tham quyen xu phat"),
        "semantic": ("quy dinh ve chu the va tham quyen xu phat vi pham hanh chinh",),
    },
    "don_phuong_cham_dut_hop_dong": {
        "core": ("don phuong cham dut hop dong",),
        "expanded": (
            "don phuong cham dut hop dong",
            "cham dut hop dong lao dong",
            "quyen don phuong cham dut",
        ),
        "title": ("cham dut hop dong", "don phuong cham dut hop dong"),
        "semantic": ("quy dinh ve truong hop duoc don phuong cham dut hop dong",),
    },
    "dieu_kien_tuyen_dung": {
        "core": ("dieu kien tuyen dung",),
        "expanded": ("dieu kien tuyen dung", "tieu chuan tuyen dung", "dieu kien du tuyen"),
        "title": ("dieu kien tuyen dung", "tieu chuan tuyen dung"),
        "semantic": ("quy dinh ve dieu kien va tieu chuan tuyen dung",),
    },
    "xu_ly_ky_luat": {
        "core": ("xu ly ky luat",),
        "expanded": ("xu ly ky luat", "hinh thuc ky luat", "truong hop bi ky luat"),
        "title": ("xu ly ky luat", "hinh thuc ky luat"),
        "semantic": ("quy dinh ve xu ly ky luat va cac hinh thuc ky luat",),
    },
    "dieu_kien_huong": {
        "core": ("dieu kien huong",),
        "expanded": ("dieu kien huong", "duoc huong khi nao", "truong hop duoc huong"),
        "title": ("dieu kien huong", "truong hop duoc huong"),
        "semantic": ("quy dinh ve dieu kien va truong hop duoc huong",),
    },
    "thu_tuc_ho_so": {
        "core": ("thu tuc ho so",),
        "expanded": ("thu tuc", "ho so", "trinh tu", "thanh phan ho so"),
        "title": ("thu tuc", "ho so", "trinh tu"),
        "semantic": ("quy dinh ve thu tuc trinh tu va thanh phan ho so",),
    },
    "tham_quyen": {
        "core": ("tham quyen",),
        "expanded": ("tham quyen", "co quan co tham quyen", "quyet dinh boi ai"),
        "title": ("tham quyen", "co quan co tham quyen"),
        "semantic": ("quy dinh ve co quan va chu the co tham quyen",),
    },
}

_CONCEPT_SEED_QUERY_RULES: dict[str, tuple[str, ...]] = {
    "pham_vi_dieu_chinh": (
        "pham vi dieu chinh",
        "doi tuong ap dung",
        "ap dung doi voi",
    ),
    "trach_nhiem": (
        "trach nhiem",
        "chiu trach nhiem",
        "co trach nhiem",
    ),
    "nhiem_vu_quyen_han": (
        "nhiem vu quyen han",
        "chuc nang nhiem vu quyen han",
    ),
    "quyen_va_nghia_vu": (
        "quyen va nghia vu",
        "quyen loi va nghia vu",
    ),
    "dieu_kien_tuyen_dung": (
        "dieu kien tuyen dung",
        "tieu chuan tuyen dung",
    ),
    "xu_ly_ky_luat": (
        "xu ly ky luat",
        "hinh thuc ky luat",
    ),
    "don_phuong_cham_dut_hop_dong": (
        "don phuong cham dut hop dong",
        "thoi han bao truoc",
    ),
    "tham_quyen_xu_phat": (
        "tham quyen xu phat vi pham hanh chinh",
        "nguoi co tham quyen xu phat",
    ),
}

_DOC_TYPE_PRIOR_BY_TOPIC: dict[str, list[str]] = {
    "general_legal": ["bo_luat", "luat", "vbhn", "nghi_dinh", "thong_tu"],
    "labor": ["bo_luat", "luat", "vbhn", "nghi_dinh", "thong_tu"],
    "civil_service": ["luat", "vbhn", "bo_luat", "nghi_dinh", "thong_tu"],
    "administrative_sanction": ["luat", "vbhn", "nghi_dinh", "thong_tu"],
    "social_insurance": ["luat", "vbhn", "nghi_dinh", "thong_tu"],
    "banking": ["luat", "vbhn", "nghi_dinh", "thong_tu"],
}

_OBJECT_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhop\s+dong\s+lao\s+dong\b", "hop dong lao dong"),
    (r"\bvi\s+pham\s+hanh\s+chinh\b", "vi pham hanh chinh"),
    (r"\bcong\s+viec\b", "cong viec"),
    (r"\bho\s+so\b", "ho so"),
    (r"\btro\s+cap\s+that\s+nghiep\b", "tro cap that nghiep"),
    (r"\bbao\s+hiem\s+xa\s+hoi\b", "bao hiem xa hoi"),
    (r"\btuyen\s+dung\b", "tuyen dung"),
)

_QUALIFIER_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhien\s+nay\b", "hien nay"),
    (r"\bco\s+ban\b", "co ban"),
    (r"\bcu\s+the\b", "cu the"),
    (r"\bkhi\s+nao\b", "khi nao"),
    (r"\btruong\s+hop\b", "truong hop"),
    (r"\bthuc\s+te\b", "thuc te"),
)

_LOCAL_ANCHOR_PATTERN = re.compile(
    r"\b(ubnd|hdnd|so\b|tinh\b|thanh pho|quan\b|huyen\b|xa\b|phuong\b)\b"
)


def _to_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or ""))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("\u0111", "d").replace("\u0110", "D")
    return value


def _normalize_text(text: str) -> str:
    value = _to_ascii(str(text or "")).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def _extract_phrases(text: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pattern, phrase in patterns:
        if re.search(pattern, text):
            normalized_phrase = _normalize_text(phrase)
            if not normalized_phrase or normalized_phrase in seen:
                continue
            seen.add(normalized_phrase)
            out.append(normalized_phrase)
    return out


def _dedup_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = _normalize_text(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _dedup_raw_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _is_short_non_keep_token(token: str) -> bool:
    if token.isdigit():
        return False
    return len(token) < 3 and token not in _LEGAL_KEEP_TOKENS


class QueryRewriteService:
    def rewrite(self, query: str, route: str = "hybrid_default") -> dict[str, Any]:
        raw = str(query or "").strip()
        clean = _normalize_text(raw)
        if not clean:
            return {
                "route": route,
                "query_raw": raw,
                "normalized_query": "",
                "query_clean": "",
                "lexical_core": "",
                "lexical_expanded": "",
                "concept_seed_query": "",
                "title_anchor_query": "",
                "lexical_query": "",
                "semantic_query": "",
                "focus_terms": [],
                "legal_anchor_guess": [],
                "legal_anchor_guess_text": "",
                "legal_anchor_guess_list": [],
                "doc_type_prior": ["luat", "bo_luat", "nghi_dinh", "vbhn"],
                "exclude_doc_type_hint": ["quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"],
                "topic_class": "vague",
                "subclass": "unknown",
                "query_subclass": "unknown",
                "concept_tags": [],
                "legal_concept_tags": [],
                "concept_confidence": 0.0,
                "actor_terms": [],
                "action_terms": [],
                "object_terms": [],
                "qualifier_terms": [],
                "is_concept_query": False,
                "is_topic_broad": False,
                "vagueness_level": "hard",
                "rewrite_risk": "high",
                "rewrite_confidence": 0.0,
                "query_too_vague": True,
                "lexical_is_weak": True,
                "weak_query_abort": True,
                "weak_query_abort_reasons": ["empty_query"],
                "lexical_quality_flags": ["empty_query"],
                "intent_template_hits": [],
                "lexical_expansion_used": [],
                "fillers_removed": [],
                "phrase_repairs": [],
                "token_classes": {},
            }

        fillers_removed: list[str] = []
        for phrase in _FILLER_PHRASES:
            pattern = rf"\b{re.escape(phrase)}\b"
            if re.search(pattern, clean):
                clean = re.sub(pattern, " ", clean)
                fillers_removed.append(phrase)

        phrase_repairs: list[str] = []
        for pattern, replacement in _PHRASE_REPAIRS:
            if re.search(pattern, clean):
                clean = re.sub(pattern, replacement, clean)
                phrase_repairs.append(f"{pattern}->{replacement}")
        clean = _SPACE_RE.sub(" ", clean).strip()

        ambiguous_hit = any(re.search(pattern, clean) for pattern in _AMBIGUOUS_PATTERNS)

        strong_phrases = _extract_phrases(clean, _STRONG_PHRASE_PATTERNS)
        entity_phrases = _extract_phrases(clean, _ENTITY_PHRASE_PATTERNS)
        procedural_phrases = _extract_phrases(clean, _PROCEDURAL_PHRASE_PATTERNS)
        actor_terms, action_terms = self._extract_actor_action_terms(
            clean=clean,
            entity_phrases=entity_phrases,
            strong_phrases=strong_phrases,
        )
        object_terms = _extract_phrases(clean, _OBJECT_PHRASE_PATTERNS)
        qualifier_terms = _extract_phrases(clean, _QUALIFIER_PHRASE_PATTERNS)

        intent_template_hits = self._intent_template_expansion(
            clean=clean,
            strong_phrases=strong_phrases,
            entity_phrases=entity_phrases,
            procedural_phrases=procedural_phrases,
        )

        legal_concept_tags, concept_expansions, concept_subclasses = self._extract_legal_concepts(
            clean=clean,
            strong_phrases=strong_phrases,
            intent_template_hits=intent_template_hits,
        )
        concept_tags_v3 = self._detect_concept_tags_v3(clean)
        legal_concept_tags = _dedup_raw_keep_order(legal_concept_tags + concept_tags_v3)
        concept_expansions = _dedup_keep_order(
            concept_expansions
            + [
                phrase
                for tag in legal_concept_tags
                for phrase in _CONCEPT_TAG_PHRASES.get(tag, ())
            ]
        )

        query_subclass = self._classify_query_subclass(
            clean=clean,
            concept_subclasses=concept_subclasses,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            procedural_phrases=procedural_phrases,
            strong_phrases=strong_phrases,
        )

        all_tokens = tokenize_for_query(clean).split()
        topical_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in all_tokens:
            if token in _STOP_TOKENS and token not in _LEGAL_KEEP_TOKENS:
                continue
            if token.isdigit():
                continue
            if _is_short_non_keep_token(token):
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            topical_tokens.append(token)

        topic_class = self._classify_topic_class(
            clean=clean,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            legal_concept_tags=legal_concept_tags,
        )
        concept_confidence = self._estimate_concept_confidence(
            clean=clean,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            strong_phrases=strong_phrases,
        )
        legal_anchor_guess_list = self._guess_legal_anchor_guesses(
            clean=clean,
            topic_class=topic_class,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            object_terms=object_terms,
        )
        legal_anchor_guess_text = legal_anchor_guess_list[0] if legal_anchor_guess_list else ""
        doc_type_prior = list(_DOC_TYPE_PRIOR_BY_TOPIC.get(topic_class, _DOC_TYPE_PRIOR_BY_TOPIC["general_legal"]))
        exclude_doc_type_hint = ["quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"]
        if _LOCAL_ANCHOR_PATTERN.search(clean):
            exclude_doc_type_hint = []

        lexical_core = self._build_lexical_core(
            clean=clean,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            strong_phrases=strong_phrases,
        )
        lexical_expanded = self._build_lexical_expanded(
            clean=clean,
            lexical_core=lexical_core,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            legal_anchor_guess_list=legal_anchor_guess_list,
            intent_template_hits=intent_template_hits,
            topical_tokens=topical_tokens,
        )
        title_anchor_query = self._build_title_anchor_query(
            legal_concept_tags=legal_concept_tags,
            legal_anchor_guess_list=legal_anchor_guess_list,
            actor_terms=actor_terms,
            object_terms=object_terms,
        )
        concept_seed_query = self._build_concept_seed_query(
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            intent_template_hits=intent_template_hits,
            strong_phrases=strong_phrases,
        )

        lexical_tokens_raw = tokenize_for_query(lexical_expanded).split()
        lexical_tokens, lexical_noise_flags = self._sanitize_lexical_tokens(lexical_tokens_raw)
        if not lexical_tokens:
            lexical_tokens = tokenize_for_query(clean).split()
        lexical_query = " ".join(lexical_tokens[:18]).strip() or clean
        lexical_expanded = lexical_query
        if not lexical_core:
            lexical_core = " ".join(lexical_tokens[:10]).strip() or clean
        if not title_anchor_query:
            title_anchor_query = self._build_title_anchor_query(
                legal_concept_tags=legal_concept_tags,
                legal_anchor_guess_list=legal_anchor_guess_list,
                actor_terms=actor_terms,
                object_terms=object_terms,
            )
        if not concept_seed_query:
            concept_seed_query = self._build_concept_seed_query(
                legal_concept_tags=legal_concept_tags,
                actor_terms=actor_terms,
                action_terms=action_terms,
                object_terms=object_terms,
                intent_template_hits=intent_template_hits,
                strong_phrases=strong_phrases,
            )

        lexical_is_weak, lexical_quality_flags = self._is_weak_lexical_query(
            lexical_query=lexical_query,
            strong_phrases=strong_phrases,
            entity_phrases=entity_phrases,
            intent_template_hits=intent_template_hits,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            query_subclass=query_subclass,
        )
        lexical_quality_flags = _dedup_keep_order(lexical_quality_flags + lexical_noise_flags)

        lexical_expansion_used: list[str] = []
        if lexical_is_weak:
            expanded_query, expanded_hints = self._expand_weak_lexical_query(
                clean=clean,
                lexical_query=lexical_query,
                legal_concept_tags=legal_concept_tags,
                query_subclass=query_subclass,
                strong_phrases=strong_phrases,
                entity_phrases=entity_phrases,
                topical_tokens=topical_tokens,
                actor_terms=actor_terms,
                action_terms=action_terms,
            )
            if expanded_query and expanded_query != lexical_query:
                lexical_query = expanded_query
                lexical_expansion_used = expanded_hints
                lexical_quality_flags = _dedup_keep_order(
                    list(lexical_quality_flags) + ["expanded_weak_lexical"]
                )
                lexical_is_weak, lexical_quality_flags_after = self._is_weak_lexical_query(
                    lexical_query=lexical_query,
                    strong_phrases=strong_phrases,
                    entity_phrases=entity_phrases,
                    intent_template_hits=intent_template_hits + lexical_expansion_used,
                    legal_concept_tags=legal_concept_tags,
                    actor_terms=actor_terms,
                    action_terms=action_terms,
                    query_subclass=query_subclass,
                )
                lexical_quality_flags = _dedup_keep_order(
                    lexical_quality_flags + lexical_quality_flags_after + lexical_noise_flags
                )

        if query_subclass == "concept_generic" and legal_concept_tags:
            concept_seed: list[str] = []
            for tag in legal_concept_tags[:3]:
                concept_seed.extend(list(_CONCEPT_TAG_TO_EXPANSIONS.get(tag, ())[:2]))
            lexical_query = self._merge_query_tokens(
                lexical_query,
                concept_seed + action_terms[:2],
                limit=18,
            )

        lexical_norm = " ".join(tokenize_for_query(lexical_query).split())
        if lexical_norm in _GENERIC_WEAK_PHRASES:
            lexical_is_weak = True
            lexical_quality_flags = _dedup_keep_order(
                list(lexical_quality_flags) + ["generic_weak_phrase"]
            )

        if any(flag.startswith("unstable_bigram") for flag in lexical_noise_flags):
            lexical_is_weak = True
            lexical_quality_flags = _dedup_keep_order(
                lexical_quality_flags + ["unstable_bigram"]
            )

        weak_query_abort, weak_query_abort_reasons = self._detect_weak_query_abort(
            lexical_core=lexical_core,
            lexical_query=lexical_query,
            concept_seed_query=concept_seed_query,
            title_anchor_query=title_anchor_query,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            lexical_quality_flags=lexical_quality_flags,
        )
        if weak_query_abort:
            lexical_is_weak = True
            lexical_quality_flags = _dedup_keep_order(
                lexical_quality_flags + weak_query_abort_reasons + ["weak_query_abort"]
            )

        semantic_query = self._build_semantic_query_v3(
            clean=clean,
            topic_class=topic_class,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            legal_anchor_guess_list=legal_anchor_guess_list,
            strong_phrases=strong_phrases,
        )

        legacy_query_too_vague = self._is_query_too_vague(
            strong_phrases=strong_phrases,
            entity_phrases=entity_phrases,
            topical_tokens=topical_tokens,
            lexical_query=lexical_query,
            legal_anchor_guess=legal_anchor_guess_text,
            ambiguous_hit=ambiguous_hit,
            lexical_is_weak=lexical_is_weak,
            query_subclass=query_subclass,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
        )
        vagueness_level, vagueness_flags = self._assess_vagueness_level(
            lexical_tokens=lexical_tokens,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            object_terms=object_terms,
            legal_anchor_guess_list=legal_anchor_guess_list,
            ambiguous_hit=ambiguous_hit,
            lexical_is_weak=lexical_is_weak,
        )
        query_too_vague = vagueness_level == "hard" or legacy_query_too_vague
        lexical_quality_flags = _dedup_keep_order(
            lexical_quality_flags + vagueness_flags
        )
        if weak_query_abort:
            query_too_vague = True
            if vagueness_level != "hard":
                vagueness_level = "hard"
            lexical_quality_flags = _dedup_keep_order(
                lexical_quality_flags + ["weak_query_abort"]
            )
        lexical_is_weak = bool(
            lexical_is_weak
            or ("weak_structure" in lexical_quality_flags)
            or vagueness_level in {"medium", "hard"}
        )

        has_doc_or_structured_anchor = bool(
            re.search(r"\b\d{1,4}/\d{2,4}\b", clean)
            or re.search(r"\b(dieu|khoan|chuong)\s+\d+", clean)
        )
        has_law_name_anchor = bool(re.search(r"\b(bo\s+luat|luat)\b", clean))
        is_concept_query = bool(legal_concept_tags) and not has_doc_or_structured_anchor and not has_law_name_anchor
        if query_subclass in {"concept_generic", "sanction_power"} and concept_confidence >= 0.45 and not has_doc_or_structured_anchor:
            is_concept_query = True
        is_topic_broad = topic_class in {"general_legal", "civil_service", "labor", "administrative_sanction"}

        rewrite_risk = self._assess_rewrite_risk(
            query_subclass=query_subclass,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
            lexical_query=lexical_query,
            lexical_is_weak=lexical_is_weak,
            lexical_quality_flags=lexical_quality_flags,
            legal_anchor_guess=legal_anchor_guess_text,
            ambiguous_hit=ambiguous_hit,
            query_too_vague=query_too_vague,
        )

        if rewrite_risk == "high":
            query_too_vague = True
        elif query_subclass == "concept_generic" and legal_concept_tags and rewrite_risk != "high":
            query_too_vague = False

        rewrite_confidence = self._estimate_confidence(
            strong_phrases=strong_phrases,
            topical_tokens=topical_tokens,
            fillers_removed=fillers_removed,
            legal_anchor_guess=legal_anchor_guess_text,
            ambiguous_hit=ambiguous_hit,
            query_too_vague=query_too_vague,
            lexical_is_weak=lexical_is_weak,
            rewrite_risk=rewrite_risk,
            legal_concept_tags=legal_concept_tags,
            actor_terms=actor_terms,
            action_terms=action_terms,
        )
        rewrite_confidence = max(
            rewrite_confidence,
            min(0.97, 0.25 + concept_confidence * 0.72),
        )
        if vagueness_level == "medium":
            rewrite_confidence = max(0.12, rewrite_confidence - 0.08)
        elif vagueness_level == "hard":
            rewrite_confidence = max(0.08, rewrite_confidence - 0.22)

        focus_terms = _dedup_keep_order(
            strong_phrases[:3]
            + concept_expansions[:4]
            + action_terms[:3]
            + topical_tokens[:4]
        )[:8]

        return {
            "route": route,
            "query_raw": raw,
            "normalized_query": clean,
            "query_clean": clean,
            "lexical_core": lexical_core,
            "lexical_expanded": lexical_expanded,
            "concept_seed_query": concept_seed_query,
            "title_anchor_query": title_anchor_query,
            "lexical_query": lexical_query,
            "semantic_query": semantic_query,
            "focus_terms": focus_terms,
            "legal_anchor_guess": list(legal_anchor_guess_list),
            "legal_anchor_guess_text": legal_anchor_guess_text,
            "legal_anchor_guess_list": list(legal_anchor_guess_list),
            "doc_type_prior": doc_type_prior,
            "exclude_doc_type_hint": exclude_doc_type_hint,
            "topic_class": topic_class,
            "subclass": query_subclass,
            "query_subclass": query_subclass,
            "concept_tags": list(legal_concept_tags),
            "legal_concept_tags": legal_concept_tags,
            "concept_confidence": round(float(concept_confidence), 4),
            "actor_terms": actor_terms,
            "action_terms": action_terms,
            "object_terms": object_terms,
            "qualifier_terms": qualifier_terms,
            "is_concept_query": bool(is_concept_query),
            "is_topic_broad": bool(is_topic_broad),
            "vagueness_level": vagueness_level,
            "rewrite_risk": rewrite_risk,
            "rewrite_confidence": round(float(rewrite_confidence), 4),
            "query_too_vague": bool(query_too_vague),
            "lexical_is_weak": bool(lexical_is_weak),
            "weak_query_abort": bool(weak_query_abort),
            "weak_query_abort_reasons": weak_query_abort_reasons,
            "lexical_quality_flags": lexical_quality_flags,
            "intent_template_hits": intent_template_hits,
            "lexical_expansion_used": lexical_expansion_used,
            "fillers_removed": fillers_removed,
            "phrase_repairs": phrase_repairs,
            "token_classes": {
                "strong_phrases": strong_phrases,
                "entity_phrases": entity_phrases,
                "procedural_phrases": procedural_phrases,
                "topical_tokens": topical_tokens,
                "legal_concept_tags": legal_concept_tags,
                "actor_terms": actor_terms,
                "action_terms": action_terms,
                "object_terms": object_terms,
                "qualifier_terms": qualifier_terms,
                "ambiguous_hit": bool(ambiguous_hit),
                "lexical_is_weak": bool(lexical_is_weak),
                "weak_query_abort": bool(weak_query_abort),
                "weak_query_abort_reasons": weak_query_abort_reasons,
                "lexical_quality_flags": lexical_quality_flags,
                "intent_template_hits": intent_template_hits,
                "lexical_expansion_used": lexical_expansion_used,
                "rewrite_risk": rewrite_risk,
                "query_subclass": query_subclass,
                "concept_confidence": round(float(concept_confidence), 4),
                "vagueness_level": vagueness_level,
            },
        }

    @staticmethod
    def _detect_concept_tags_v3(clean: str) -> list[str]:
        hay = _normalize_text(clean)
        tags: list[str] = []
        for tag, phrases in _CONCEPT_TAG_PHRASES.items():
            if any(phrase in hay for phrase in phrases):
                tags.append(tag)
        return _dedup_raw_keep_order(tags)

    @staticmethod
    def _classify_topic_class(
        *,
        clean: str,
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        legal_concept_tags: list[str],
    ) -> str:
        hay = _normalize_text(" ".join([clean] + actor_terms + action_terms + object_terms + legal_concept_tags))
        if re.search(r"\b(lao dong|hop dong lao dong|nguoi lao dong)\b", hay):
            return "labor"
        if re.search(r"\b(cong chuc|can bo|vien chuc|thanh tra)\b", hay):
            return "civil_service"
        if re.search(r"\b(xu phat|vi pham hanh chinh|tham quyen xu phat)\b", hay):
            return "administrative_sanction"
        if re.search(r"\b(bao hiem xa hoi|tro cap that nghiep)\b", hay):
            return "social_insurance"
        if re.search(r"\b(ngan hang|tin dung|lai suat)\b", hay):
            return "banking"
        return "general_legal"

    @staticmethod
    def _estimate_concept_confidence(
        *,
        clean: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        strong_phrases: list[str],
    ) -> float:
        score = 0.08
        score += min(0.48, 0.16 * len(legal_concept_tags))
        if strong_phrases:
            score += min(0.20, 0.06 * len(strong_phrases))
        if actor_terms:
            score += 0.08
        if action_terms:
            score += 0.10
        if object_terms:
            score += 0.06
        if re.search(r"\b(dieu|khoan|chuong)\s+\d+", clean):
            score -= 0.10
        return max(0.0, min(1.0, score))

    @staticmethod
    def _guess_legal_anchor_guesses(
        *,
        clean: str,
        topic_class: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        object_terms: list[str],
    ) -> list[str]:
        hay = _normalize_text(" ".join([clean] + actor_terms + object_terms + legal_concept_tags))
        anchors: list[str] = []
        if "cong chuc" in hay or "can bo" in hay or "vien chuc" in hay:
            anchors.extend(["can bo cong chuc", "cong chuc"])
        if "lao dong" in hay or "hop dong lao dong" in hay:
            anchors.extend(["lao dong", "hop dong lao dong"])
        if "xu phat" in hay or "vi pham hanh chinh" in hay:
            anchors.extend(["xu ly vi pham hanh chinh", "xu phat vi pham hanh chinh"])
        if "bao hiem xa hoi" in hay:
            anchors.append("bao hiem xa hoi")
        if "tham_quyen_xu_phat" in legal_concept_tags:
            anchors.append("xu phat vi pham hanh chinh")
        if topic_class == "labor":
            anchors.append("lao dong")
        elif topic_class == "civil_service":
            anchors.append("can bo cong chuc")
        elif topic_class == "administrative_sanction":
            anchors.append("xu ly vi pham hanh chinh")
        if not anchors:
            for tag in legal_concept_tags:
                hint = _CONCEPT_TAG_TO_ANCHOR.get(tag, "")
                if hint:
                    anchors.append(hint)
        return _dedup_keep_order(anchors)[:5]

    @staticmethod
    def _build_lexical_core(
        *,
        clean: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        strong_phrases: list[str],
    ) -> str:
        parts: list[str] = []
        for tag in legal_concept_tags:
            parts.extend(list(_CONCEPT_EXPANSION_RULES.get(tag, {}).get("core", ())))
        parts.extend(strong_phrases[:2])
        parts.extend(actor_terms[:2])
        parts.extend(action_terms[:2])
        parts.extend(object_terms[:2])
        parts = _dedup_keep_order(parts)
        if not parts:
            parts = tokenize_for_query(clean).split()[:10]
        return " ".join(tokenize_for_query(" ".join(parts)).split()[:14]).strip()

    @staticmethod
    def _build_lexical_expanded(
        *,
        clean: str,
        lexical_core: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        legal_anchor_guess_list: list[str],
        intent_template_hits: list[str],
        topical_tokens: list[str],
    ) -> str:
        parts: list[str] = [lexical_core]
        for tag in legal_concept_tags:
            parts.extend(list(_CONCEPT_EXPANSION_RULES.get(tag, {}).get("expanded", ())))
        parts.extend(intent_template_hits[:3])
        parts.extend(actor_terms[:3])
        parts.extend(action_terms[:3])
        parts.extend(object_terms[:3])
        parts.extend(legal_anchor_guess_list[:3])
        parts.extend(topical_tokens[:8])
        merged = " ".join(_dedup_keep_order(parts))
        return " ".join(tokenize_for_query(merged).split()[:20]).strip() or lexical_core or clean

    @staticmethod
    def _build_title_anchor_query(
        *,
        legal_concept_tags: list[str],
        legal_anchor_guess_list: list[str],
        actor_terms: list[str],
        object_terms: list[str],
    ) -> str:
        parts: list[str] = []
        for tag in legal_concept_tags:
            parts.extend(list(_CONCEPT_EXPANSION_RULES.get(tag, {}).get("title", ())))
        parts.extend(legal_anchor_guess_list[:3])
        parts.extend(actor_terms[:2])
        parts.extend(object_terms[:2])
        filtered_parts: list[str] = []
        for part in _dedup_keep_order(parts):
            norm = _normalize_text(part)
            if not norm:
                continue
            if norm in _TITLE_ANCHOR_BLACKLIST_PHRASES:
                continue
            if len(norm.split()) < 2 and norm in _WEAK_LEXICAL_ABORT_SINGLE_TOKENS:
                continue
            filtered_parts.append(norm)
        return " ".join(tokenize_for_query(" ".join(filtered_parts)).split()[:12]).strip()

    @staticmethod
    def _build_concept_seed_query(
        *,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        intent_template_hits: list[str],
        strong_phrases: list[str],
    ) -> str:
        seed_parts: list[str] = []
        for tag in legal_concept_tags[:4]:
            seed_parts.extend(list(_CONCEPT_SEED_QUERY_RULES.get(tag, ())[:2]))
        seed_parts.extend(intent_template_hits[:2])
        seed_parts.extend(strong_phrases[:2])
        seed_parts.extend(actor_terms[:2])
        seed_parts.extend(action_terms[:3])
        seed_parts.extend(object_terms[:2])
        merged = " ".join(_dedup_keep_order(seed_parts))
        merged_tokens, _ = QueryRewriteService._sanitize_lexical_tokens(
            tokenize_for_query(merged).split()
        )
        return " ".join(merged_tokens[:18]).strip()

    @staticmethod
    def _detect_weak_query_abort(
        *,
        lexical_core: str,
        lexical_query: str,
        concept_seed_query: str,
        title_anchor_query: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        lexical_quality_flags: list[str],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        lexical_core_norm = _normalize_text(lexical_core)
        lexical_norm = _normalize_text(lexical_query)
        lexical_tokens = tokenize_for_query(lexical_norm).split()
        lexical_unique = [tok for tok in lexical_tokens if tok and tok not in _STOP_TOKENS]
        token_variety = len(set(lexical_tokens))

        if lexical_core_norm in _WEAK_LEXICAL_ABORT_SINGLE_TOKENS:
            reasons.append("weak_single_token_core")
        if lexical_norm in _WEAK_LEXICAL_ABORT_PHRASES or lexical_core_norm in _WEAK_LEXICAL_ABORT_PHRASES:
            reasons.append("weak_generic_phrase")
        if len(lexical_unique) <= 2:
            reasons.append("lexical_content_too_short")
        if token_variety <= 2:
            reasons.append("low_token_variety")

        weak_structure_flags = {
            "too_short",
            "weak_structure",
            "low_token_variety",
            "missing_action",
            "missing_actor",
        }
        if any(flag in weak_structure_flags for flag in lexical_quality_flags):
            reasons.append("weak_quality_flags")

        has_non_trivial_tuple = bool(
            legal_concept_tags
            and (
                (actor_terms and action_terms)
                or (action_terms and object_terms)
                or (actor_terms and object_terms)
            )
        )
        concept_seed_tokens = tokenize_for_query(concept_seed_query).split()
        title_anchor_tokens = tokenize_for_query(title_anchor_query).split()
        title_anchor_is_usable = len(title_anchor_tokens) >= 4 and not any(
            token in _WEAK_LEXICAL_ABORT_SINGLE_TOKENS
            for token in title_anchor_tokens
        )

        allow_recovery = bool(
            len(lexical_unique) >= 3
            or has_non_trivial_tuple
            or len(concept_seed_tokens) >= 4
            or title_anchor_is_usable
        )
        abort = bool(reasons) and not allow_recovery
        return abort, _dedup_keep_order(reasons)

    @staticmethod
    def _build_semantic_query_v3(
        *,
        clean: str,
        topic_class: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        legal_anchor_guess_list: list[str],
        strong_phrases: list[str],
    ) -> str:
        clauses: list[str] = []
        if topic_class:
            clauses.append(f"chu de phap ly: {topic_class}")
        if legal_concept_tags:
            clauses.append("concept phap ly: " + ", ".join(legal_concept_tags[:5]))
            for tag in legal_concept_tags[:2]:
                semantic_terms = list(_CONCEPT_EXPANSION_RULES.get(tag, {}).get("semantic", ()))
                if semantic_terms:
                    clauses.append(semantic_terms[0])
        if actor_terms:
            clauses.append("chu the: " + ", ".join(actor_terms[:3]))
        if action_terms:
            clauses.append("hanh vi: " + ", ".join(action_terms[:3]))
        if object_terms:
            clauses.append("doi tuong: " + ", ".join(object_terms[:3]))
        if legal_anchor_guess_list:
            clauses.append("anchor van ban du kien: " + ", ".join(legal_anchor_guess_list[:3]))
        if strong_phrases:
            clauses.append("cum tu trong tam: " + "; ".join(strong_phrases[:2]))
        if not clauses:
            return clean
        return ". ".join(clauses)

    @staticmethod
    def _assess_vagueness_level(
        *,
        lexical_tokens: list[str],
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        object_terms: list[str],
        legal_anchor_guess_list: list[str],
        ambiguous_hit: bool,
        lexical_is_weak: bool,
    ) -> tuple[str, list[str]]:
        content_tokens = [
            tok for tok in lexical_tokens
            if tok and tok not in _STOP_TOKENS and not tok.isdigit()
        ]
        flags: list[str] = []
        if len(content_tokens) < 2:
            flags.append("too_short")
        if not actor_terms:
            flags.append("missing_actor")
        if not action_terms:
            flags.append("missing_action")
        if not object_terms:
            flags.append("missing_object")
        if lexical_is_weak:
            flags.append("weak_structure")
        if ambiguous_hit:
            flags.append("ambiguous_noise")

        if (
            len(content_tokens) < 2
            and not legal_concept_tags
            and not actor_terms
            and not action_terms
            and not legal_anchor_guess_list
        ):
            return "hard", _dedup_keep_order(flags)

        if ambiguous_hit and not legal_concept_tags and not legal_anchor_guess_list:
            return "hard", _dedup_keep_order(flags)

        if legal_concept_tags and (not actor_terms or not action_terms) and not object_terms:
            return "medium", _dedup_keep_order(flags)

        if lexical_is_weak and (not legal_anchor_guess_list or len(content_tokens) <= 3):
            return "medium", _dedup_keep_order(flags)

        return "none", _dedup_keep_order(flags)

    @staticmethod
    def _intent_template_expansion(
        *,
        clean: str,
        strong_phrases: list[str],
        entity_phrases: list[str],
        procedural_phrases: list[str],
    ) -> list[str]:
        hay = tokenize_for_query(" ".join([clean] + strong_phrases + entity_phrases + procedural_phrases))
        tokens = set(hay.split())
        out: list[str] = []

        if "cong chuc" in clean and ("dieu kien" in clean or {"tuyen", "dung"} & tokens):
            out.append("dieu kien tuyen dung cong chuc")

        if "cong chuc" in clean and ("quyen va nghia vu" in clean or {"quyen", "nghia", "vu"} <= tokens):
            out.append("quyen va nghia vu co ban cua cong chuc")

        if ("lao dong" in clean or "hop dong" in clean) and (
            "don phuong cham dut" in clean or {"cham", "dut"} <= tokens
        ):
            out.append("don phuong cham dut hop dong lao dong")

        if (
            ("tham quyen" in clean or {"tham", "quyen"} <= tokens)
            and ("xu phat" in clean or {"xu", "phat"} <= tokens)
            and ("vi pham hanh chinh" in clean or {"vi", "pham", "hanh", "chinh"} <= tokens)
        ):
            out.append("tham quyen xu phat vi pham hanh chinh")

        if ("pham vi dieu chinh" in clean) or ({"pham", "vi", "dieu", "chinh"} <= tokens):
            out.append("pham vi dieu chinh")

        return _dedup_keep_order(out)

    @staticmethod
    def _extract_legal_concepts(
        *,
        clean: str,
        strong_phrases: list[str],
        intent_template_hits: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        hay = _normalize_text(" ".join([clean] + strong_phrases + intent_template_hits))
        concept_tags: list[str] = []
        expansions: list[str] = []
        subclasses: list[str] = []

        for rule in _CONCEPT_RULES:
            pattern = str(rule.get("pattern", ""))
            if not pattern:
                continue
            if not re.search(pattern, hay):
                continue
            tag = str(rule.get("tag", "")).strip().lower()
            if tag:
                concept_tags.append(tag)
            subclasses.append(str(rule.get("subclass", "unknown")))
            for exp in (rule.get("expansions") or ()):  # type: ignore[arg-type]
                expansions.append(str(exp))

        return (
            _dedup_raw_keep_order(concept_tags),
            _dedup_keep_order(expansions),
            _dedup_raw_keep_order(subclasses),
        )

    @staticmethod
    def _extract_actor_action_terms(
        *,
        clean: str,
        entity_phrases: list[str],
        strong_phrases: list[str],
    ) -> tuple[list[str], list[str]]:
        actors = _extract_phrases(clean, _ACTOR_PHRASE_PATTERNS)
        actions = _extract_phrases(clean, _ACTION_PHRASE_PATTERNS)

        tokens = tokenize_for_query(" ".join([clean] + entity_phrases + strong_phrases)).split()
        for token in tokens:
            if token in _ACTOR_TOKENS:
                actors.append(token)
            if token in _ACTION_TOKENS:
                actions.append(token)

        return _dedup_keep_order(actors)[:8], _dedup_keep_order(actions)[:8]

    @staticmethod
    def _classify_query_subclass(
        *,
        clean: str,
        concept_subclasses: list[str],
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        procedural_phrases: list[str],
        strong_phrases: list[str],
    ) -> str:
        subclasses = {str(v).strip() for v in concept_subclasses if str(v).strip()}
        if "sanction_power" in subclasses:
            return "sanction_power"

        if actor_terms and action_terms:
            return "subject_action"

        if "procedural" in subclasses:
            return "procedural"

        if procedural_phrases and not (actor_terms and action_terms):
            if any(term in clean for term in ["thu tuc", "trinh tu", "ho so", "thoi han"]):
                return "procedural"

        if "subject_action" in subclasses:
            return "subject_action"

        if legal_concept_tags or strong_phrases:
            return "concept_generic"

        if action_terms and len(action_terms) >= 2:
            return "subject_action"

        return "unknown"

    @staticmethod
    def _expand_weak_lexical_query(
        *,
        clean: str,
        lexical_query: str,
        legal_concept_tags: list[str],
        query_subclass: str,
        strong_phrases: list[str],
        entity_phrases: list[str],
        topical_tokens: list[str],
        actor_terms: list[str],
        action_terms: list[str],
    ) -> tuple[str, list[str]]:
        tokens = set(tokenize_for_query(clean).split())
        additions: list[str] = []

        if query_subclass == "subject_action":
            additions.extend(actor_terms[:3])
            additions.extend(action_terms[:4])
        elif query_subclass == "procedural":
            additions.extend(["trinh tu thu tuc", "thoi han giai quyet", "ho so thu tuc"])
        elif query_subclass == "sanction_power":
            additions.extend(["tham quyen xu phat", "xu phat vi pham hanh chinh", "nguoi co tham quyen"])

        for tag in legal_concept_tags:
            additions.extend(_CONCEPT_TAG_TO_EXPANSIONS.get(tag, ()))

        if "cong chuc" in clean:
            additions.append("can bo cong chuc")
            if {"tuyen", "dung"} & tokens:
                additions.append("dieu kien tuyen dung cong chuc")
            if {"quyen", "nghia", "vu"} & tokens:
                additions.append("quyen va nghia vu co ban cua cong chuc")
        elif "lao dong" in clean:
            additions.append("hop dong lao dong")
            if {"cham", "dut"} & tokens:
                additions.append("don phuong cham dut hop dong lao dong")

        additions.extend(strong_phrases[:2])
        additions.extend(entity_phrases[:2])
        additions.extend(actor_terms[:2])
        additions.extend(action_terms[:2])
        additions.extend(topical_tokens[:8])

        merged_tokens = tokenize_for_query(" ".join([lexical_query] + additions)).split()
        merged_tokens, _ = QueryRewriteService._sanitize_lexical_tokens(merged_tokens)
        expanded = " ".join(_dedup_keep_order(merged_tokens)[:18]).strip() or lexical_query
        used = _dedup_keep_order(additions)[:8]
        return expanded, used

    @staticmethod
    def _is_weak_lexical_query(
        *,
        lexical_query: str,
        strong_phrases: list[str],
        entity_phrases: list[str],
        intent_template_hits: list[str],
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        query_subclass: str,
    ) -> tuple[bool, list[str]]:
        lexical_tokens = [tok for tok in tokenize_for_query(lexical_query).split() if tok]
        unique_tokens = {tok for tok in lexical_tokens if tok not in _STOP_TOKENS}
        flags: list[str] = []

        has_action = bool(set(lexical_tokens) & _ACTION_TOKENS) or bool(action_terms)
        has_entity = bool(set(lexical_tokens) & _ACTOR_TOKENS) or bool(entity_phrases) or bool(actor_terms)
        has_doc_hint = bool(set(lexical_tokens) & _DOC_HINT_TOKENS)
        has_concept = bool(legal_concept_tags)

        if len(unique_tokens) <= 3:
            flags.append("too_short")
        if not has_action and not (strong_phrases or intent_template_hits or has_doc_hint or has_concept):
            flags.append("missing_action")
        if not has_entity and not has_doc_hint and query_subclass not in {"concept_generic", "procedural"}:
            flags.append("missing_entity")
        if len(unique_tokens) <= 5 and not (has_action and (has_entity or has_concept)) and not has_doc_hint:
            flags.append("weak_structure")
        if len(set(lexical_tokens)) <= 2:
            flags.append("low_token_variety")

        if query_subclass == "concept_generic" and has_concept:
            flags = [flag for flag in flags if flag not in {"missing_entity", "weak_structure"}]
        if query_subclass == "procedural":
            flags = [flag for flag in flags if flag != "missing_entity"]

        if strong_phrases or intent_template_hits:
            flags = [flag for flag in flags if flag != "weak_structure"]
            if strong_phrases and len(unique_tokens) >= 3:
                flags = [flag for flag in flags if flag != "missing_action"]

        weak = any(flag in {"too_short", "weak_structure", "low_token_variety"} for flag in flags) or (
            ("missing_action" in flags and "missing_entity" in flags)
        )
        return bool(weak), _dedup_keep_order(flags)

    def _infer_anchors_and_doc_policy(
        self,
        *,
        clean: str,
        strong_phrases: list[str],
        entity_phrases: list[str],
        topical_tokens: list[str],
        legal_concept_tags: list[str],
        query_subclass: str,
    ) -> tuple[str, str, list[str], list[str]]:
        hay = " ".join([clean, " ".join(strong_phrases), " ".join(entity_phrases), " ".join(topical_tokens)])
        local_anchor = bool(_LOCAL_ANCHOR_PATTERN.search(hay))

        if "cong chuc" in hay or "can bo" in hay or "thanh tra" in hay:
            anchor = "can bo cong chuc"
            topic = "civil_service"
            prior = ["luat", "bo_luat", "vbhn", "nghi_dinh"]
        elif "lao dong" in hay or "hop dong" in hay:
            anchor = "lao dong"
            topic = "labor"
            prior = ["bo_luat", "luat", "vbhn", "nghi_dinh", "thong_tu"]
        elif "xu phat" in hay or "vi pham hanh chinh" in hay or query_subclass == "sanction_power":
            anchor = "xu ly vi pham hanh chinh"
            topic = "administrative_sanction"
            prior = ["luat", "vbhn", "nghi_dinh", "thong_tu"]
        elif "quy hoach" in hay:
            anchor = "quy hoach"
            topic = "planning"
            prior = ["luat", "vbhn", "nghi_dinh"]
        else:
            anchor = ""
            topic = "general_legal"
            prior = ["luat", "bo_luat", "vbhn", "nghi_dinh"]

        if not anchor:
            for tag in legal_concept_tags:
                hint = _CONCEPT_TAG_TO_ANCHOR.get(tag, "")
                if hint:
                    anchor = hint
                    break

        if query_subclass == "concept_generic":
            prior = ["luat", "bo_luat", "nghi_dinh", "vbhn"]
            if topic in {"labor", "administrative_sanction"}:
                prior.append("thong_tu")

        exclude = [] if local_anchor else ["quyet_dinh_ubnd", "cong_van", "van_ban_dia_phuong"]
        return anchor, topic, _dedup_keep_order(prior), _dedup_keep_order(exclude)

    @staticmethod
    def _estimate_confidence(
        *,
        strong_phrases: list[str],
        topical_tokens: list[str],
        fillers_removed: list[str],
        legal_anchor_guess: str,
        ambiguous_hit: bool,
        query_too_vague: bool,
        lexical_is_weak: bool,
        rewrite_risk: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
    ) -> float:
        score = 0.54
        if strong_phrases:
            score += min(0.20, 0.08 + (0.05 * len(strong_phrases)))
        if legal_concept_tags:
            score += min(0.10, 0.03 + (0.02 * len(legal_concept_tags)))
        if len(topical_tokens) >= 3:
            score += 0.08
        if legal_anchor_guess:
            score += 0.08
        if actor_terms and action_terms:
            score += 0.08
        elif action_terms:
            score += 0.04
        if fillers_removed:
            score += 0.03
        if ambiguous_hit:
            score -= 0.14
        if lexical_is_weak:
            score -= 0.12
        if query_too_vague:
            score -= 0.25
        if rewrite_risk == "high":
            score -= 0.15
        elif rewrite_risk == "medium":
            score -= 0.07
        return max(0.05, min(0.97, score))

    @staticmethod
    def _is_query_too_vague(
        *,
        strong_phrases: list[str],
        entity_phrases: list[str],
        topical_tokens: list[str],
        lexical_query: str,
        legal_anchor_guess: str,
        ambiguous_hit: bool,
        lexical_is_weak: bool,
        query_subclass: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
    ) -> bool:
        if ambiguous_hit and not strong_phrases and not legal_anchor_guess:
            return True

        lexical_tokens = tokenize_for_query(lexical_query).split()
        unique_tokens = {tok for tok in lexical_tokens if tok and tok not in _STOP_TOKENS}

        if len(unique_tokens) <= 2:
            return True

        if query_subclass == "concept_generic" and legal_concept_tags:
            if lexical_is_weak and len(unique_tokens) <= 3 and not legal_anchor_guess:
                return True
            return False

        if query_subclass == "procedural" and (action_terms or "thu" in unique_tokens):
            if len(unique_tokens) >= 3:
                return False

        if lexical_is_weak and not strong_phrases and not legal_anchor_guess and len(topical_tokens) <= 3:
            return True

        if legal_anchor_guess and (len(unique_tokens) >= 3 or bool(entity_phrases) or bool(actor_terms)):
            return False

        if action_terms and (actor_terms or legal_concept_tags) and len(unique_tokens) >= 4:
            return False

        if not strong_phrases and not legal_anchor_guess and len(topical_tokens) <= 2:
            return True

        if not entity_phrases and not strong_phrases and not actor_terms and len(topical_tokens) <= 3:
            return True

        return False

    @staticmethod
    def _assess_rewrite_risk(
        *,
        query_subclass: str,
        legal_concept_tags: list[str],
        actor_terms: list[str],
        action_terms: list[str],
        lexical_query: str,
        lexical_is_weak: bool,
        lexical_quality_flags: list[str],
        legal_anchor_guess: str,
        ambiguous_hit: bool,
        query_too_vague: bool,
    ) -> str:
        score = 0

        lexical_tokens = tokenize_for_query(lexical_query).split()
        unique_tokens = {tok for tok in lexical_tokens if tok and tok not in _STOP_TOKENS}

        if ambiguous_hit:
            score += 2
        if lexical_is_weak:
            score += 2
        if query_too_vague:
            score += 3
        if len(unique_tokens) <= 3:
            score += 2
        if not legal_anchor_guess:
            score += 1
        if not legal_concept_tags:
            score += 1
        if not actor_terms and not action_terms:
            score += 1
        if query_subclass == "unknown":
            score += 1
        if any(flag.startswith("unstable_bigram") for flag in lexical_quality_flags):
            score += 2
        if "junk_token" in lexical_quality_flags:
            score += 2
        if "missing_action" in lexical_quality_flags and "missing_entity" in lexical_quality_flags:
            score += 1

        if score >= 8:
            return "high"
        if score >= 4:
            return "medium"
        return "low"

    @staticmethod
    def _sanitize_lexical_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
        sanitized: list[str] = []
        flags: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            tok = _normalize_text(token)
            if not tok:
                continue
            if tok in _LEXICAL_BLACKLIST_TOKENS and tok not in _LEGAL_KEEP_TOKENS:
                flags.append(f"blacklist:{tok}")
                continue
            if _is_short_non_keep_token(tok):
                flags.append(f"short:{tok}")
                continue
            if re.fullmatch(r"(.)\1{3,}", tok):
                flags.append("junk_token")
                continue
            if tok in seen:
                continue
            if sanitized:
                bigram = f"{sanitized[-1]} {tok}"
                if bigram in _UNSTABLE_BIGRAMS:
                    flags.append(f"unstable_bigram:{bigram}")
                    continue
            seen.add(tok)
            sanitized.append(tok)

        return sanitized, _dedup_keep_order(flags)

    @staticmethod
    def _merge_query_tokens(base_query: str, additions: list[str], *, limit: int = 18) -> str:
        merged = tokenize_for_query(" ".join([base_query] + additions)).split()
        merged, _ = QueryRewriteService._sanitize_lexical_tokens(merged)
        return " ".join(_dedup_keep_order(merged)[: max(4, int(limit))]).strip() or base_query
