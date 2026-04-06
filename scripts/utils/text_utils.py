"""
Text normalization utilities cho văn bản pháp luật Việt Nam.
"""

import re
import unicodedata
from html import unescape


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode NFC — quan trọng cho tiếng Việt (ă, ơ, ứ...)."""
    return unicodedata.normalize("NFC", text)


def strip_html(text: str) -> str:
    """Loại bỏ HTML tags, giữ lại text content."""
    text = unescape(text)                           # &amp; → &
    text = re.sub(r"<br\s*/?>", "\n", text)          # <br> → newline
    text = re.sub(r"<[^>]+>", "", text)              # <tag>...</tag>
    return text


def collapse_whitespace(text: str) -> str:
    """Gộp nhiều khoảng trắng liên tiếp thành 1, trim."""
    text = re.sub(r"[ \t]+", " ", text)              # spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)           # max 2 newlines liên tiếp
    return text.strip()


def remove_control_chars(text: str) -> str:
    """Loại ký tự điều khiển trừ \\n \\t \\r."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def clean_legal_text(text: str) -> str:
    """
    Pipeline chuẩn hóa đầy đủ cho 1 đoạn văn bản pháp luật.
    Gọi: cleaned = clean_legal_text(raw_content)
    """
    if not text or not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = remove_control_chars(text)
    text = collapse_whitespace(text)
    return text


def estimate_tokens(text: str) -> int:
    """
    Ước tính số tokens (rough). 
    Tiếng Việt ~1.5 tokens/word. Dùng word count * 1.5.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.5)
