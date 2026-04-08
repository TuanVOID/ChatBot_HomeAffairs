"""
Vietnamese-friendly text tokenizer utilities cho BM25 indexing/search.

Thiết kế:
- Không phụ thuộc underthesea/pyvi để tránh lỗi model/tokenization runtime.
- Dùng regex Unicode ổn định để tạo token nhất quán giữa lúc index và query.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", flags=re.UNICODE)
_SEP_RE = re.compile(r"[\/\\|:;,.!?()\[\]{}\"'“”‘’`~@#$%^&*=+<>-]+", flags=re.UNICODE)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("_", " ")
    text = _SEP_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _tokenize(text: str, min_len: int) -> list[str]:
    if not text:
        return []

    normalized = _normalize_text(text)
    tokens = _WORD_RE.findall(normalized)
    if min_len <= 1:
        return tokens

    return [tok for tok in tokens if tok.isdigit() or len(tok) >= min_len]


def tokenize_for_index(text: str) -> str:
    """Tokenize dùng cho dữ liệu index (giữ token 1 ký tự để không mất thông tin)."""
    return " ".join(_tokenize(text, min_len=1))


def tokenize_for_query(text: str) -> str:
    """Tokenize dùng cho query (lọc token quá ngắn để giảm nhiễu)."""
    return " ".join(_tokenize(text, min_len=2))

