"""
Import Raw-Data → documents.jsonl
==================================
Parse 55+ file .txt từ data/raw-data/ thành processed/documents.jsonl.

Mỗi file .txt có format:
  Dòng 1: URL thuvienphapluat.vn (optional)
  Dòng 2+: Nội dung văn bản

Usage:
    python scripts/import_raw_data.py
    python scripts/import_raw_data.py --dry-run
    python scripts/import_raw_data.py --input data/raw-data --output processed/documents.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

# ── Logger ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


# ══════════════════════════════════════════════════════════
# Metadata Extraction
# ══════════════════════════════════════════════════════════

# Regex patterns cho các loại VB
DOC_TYPE_PATTERNS = [
    (r"HIẾN\s*PHÁP", "Hiến pháp"),
    (r"BỘ\s*LUẬT", "Bộ luật"),
    (r"\bLUẬT\b", "Luật"),
    (r"NGHỊ\s*QUYẾT", "Nghị quyết"),
    (r"NGHỊ\s*ĐỊNH", "Nghị định"),
    (r"QUYẾT\s*ĐỊNH", "Quyết định"),
    (r"THÔNG\s*TƯ", "Thông tư"),
    (r"CHỈ\s*THỊ", "Chỉ thị"),
    (r"CÔNG\s*VĂN", "Công văn"),
]

# Regex cho số hiệu VB
NUMBER_PATTERNS = [
    # "Luật số: 80/2025/QH15" hoặc "Số: 07/2026/NĐ-CP"
    r"(?:Luật\s*số|Số)\s*:\s*([\d]+/[\d]{4}/[A-ZĐa-zđ\-]+\d*)",
    # "số 80/2025/QH15"
    r"số\s+([\d]+/[\d]{4}/[A-ZĐa-zđ\-]+\d*)",
]

# Regex cho ngày ban hành
DATE_PATTERN = r"(?:ngày|Ngày)\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"

# Regex cho cơ quan ban hành
ISSUER_PATTERNS = [
    (r"QUỐC\s*HỘI", "Quốc hội"),
    (r"ỦY\s*BAN\s*THƯỜNG\s*VỤ\s*QUỐC\s*HỘI", "Ủy ban Thường vụ Quốc hội"),
    (r"CHÍNH\s*PHỦ", "Chính phủ"),
    (r"THỦ\s*TƯỚNG\s*CHÍNH\s*PHỦ", "Thủ tướng Chính phủ"),
    (r"BỘ\s*NỘI\s*VỤ", "Bộ Nội vụ"),
    (r"BỘ\s*TRƯỞNG\s*BỘ\s*NỘI\s*VỤ", "Bộ trưởng Bộ Nội vụ"),
    (r"BAN\s*CHẤP\s*HÀNH\s*TRUNG\s*ƯƠNG", "Ban Chấp hành Trung ương"),
]

# Sector detection từ URL hoặc nội dung
SECTOR_KEYWORDS = {
    "Cán bộ, công chức": ["cán bộ", "công chức", "tuyển dụng công chức", "biên chế"],
    "Viên chức": ["viên chức"],
    "Tổ chức bộ máy": ["tổ chức bộ máy", "tổ chức chính phủ", "cơ cấu tổ chức"],
    "Chính quyền địa phương": ["chính quyền địa phương", "hội đồng nhân dân", "ủy ban nhân dân", "đơn vị hành chính"],
    "Thi đua, khen thưởng": ["thi đua", "khen thưởng", "huân chương", "danh hiệu"],
    "Tiền lương": ["tiền lương", "lương cơ sở", "phụ cấp", "hệ số lương"],
    "Tôn giáo, tín ngưỡng": ["tôn giáo", "tín ngưỡng", "nhà thờ", "chùa"],
    "Văn thư, lưu trữ": ["văn thư", "lưu trữ", "tài liệu"],
    "Thanh niên": ["thanh niên", "tuổi trẻ"],
    "Bảo hiểm xã hội": ["bảo hiểm xã hội", "bhxh", "bảo hiểm y tế"],
    "Dân chủ cơ sở": ["dân chủ cơ sở", "dân chủ ở cơ sở"],
    "Cải cách hành chính": ["cải cách hành chính", "thủ tục hành chính"],
    "Hội, tổ chức phi chính phủ": ["hội", "tổ chức phi chính phủ", "tổ chức xã hội"],
    "Thanh tra": ["thanh tra"],
    "Người có công": ["người có công", "thương binh", "liệt sĩ"],
}


def extract_url(lines: list[str]) -> tuple[str, int]:
    """Extract URL từ dòng đầu tiên. Trả về (url, start_line)."""
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line, i + 1
    return "", 0


def extract_doc_number(text: str) -> str:
    """Extract số hiệu văn bản."""
    for pattern in NUMBER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_doc_type(text: str, filename: str) -> str:
    """Xác định loại văn bản. Ưu tiên filename → standalone header line."""
    fn = filename.lower()

    # 1. Detect từ filename (đáng tin nhất)
    if "hien-phap" in fn:
        return "Hiến pháp"
    elif fn.startswith("nghi-dinh-") or fn.startswith("nd-") or fn.startswith("nd="):
        return "Nghị định"
    elif fn.startswith("nghi-quyet-") or fn.startswith("nq-"):
        return "Nghị quyết"
    elif fn.startswith("thong-tu-") or fn.startswith("tt-"):
        return "Thông tư"
    elif fn.startswith("quyet-dinh-") or fn.startswith("qd-"):
        return "Quyết định"
    elif "luat" in fn:
        return "Luật"

    # 2. Tìm standalone header line (ví dụ dòng chỉ chứa "NGHỊ ĐỊNH")
    lines = text.split("\n")
    for line in lines[:40]:
        stripped = line.strip().upper()
        if not stripped or len(stripped) > 30:
            continue
        for pattern, doc_type in DOC_TYPE_PATTERNS:
            if re.fullmatch(pattern, stripped):
                return doc_type

    return "Văn bản QPPL"


def extract_issuer(text: str) -> str:
    """Xác định cơ quan ban hành."""
    header = text[:3000].upper()

    # Check specific patterns first
    if re.search(r"ỦY\s*BAN\s*THƯỜNG\s*VỤ\s*QUỐC\s*HỘI", header):
        return "Ủy ban Thường vụ Quốc hội"
    if re.search(r"QUỐC\s*HỘI", header):
        return "Quốc hội"

    # Check footer for signer
    footer = text[-2000:].upper()
    if "THỦ TƯỚNG" in footer:
        return "Chính phủ"
    if "BỘ TRƯỞNG BỘ NỘI VỤ" in footer or "BỘ NỘI VỤ" in footer:
        return "Bộ Nội vụ"

    # Check header
    for pattern, issuer in ISSUER_PATTERNS:
        if re.search(pattern, header):
            return issuer

    # Default by doc_type
    return ""


def extract_date(text: str) -> str:
    """Extract ngày ban hành (YYYY-MM-DD)."""
    # Tìm ngày đầu tiên trong header (thường là ngày ban hành)
    match = re.search(DATE_PATTERN, text[:3000])
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return ""


def extract_title(text: str, doc_type: str, doc_number: str) -> str:
    """Extract tiêu đề văn bản."""
    lines = text.split("\n")

    # Tìm dòng tiêu đề (thường là dòng uppercase sau loại VB)
    found_type = False
    title_lines = []

    for line in lines[:50]:
        line_stripped = line.strip()
        if not line_stripped:
            if found_type and title_lines:
                break
            continue

        # Detect doc type header
        upper = line_stripped.upper()
        for pattern, _ in DOC_TYPE_PATTERNS:
            if re.search(pattern, upper) and len(line_stripped) < 30:
                found_type = True
                break

        if found_type:
            # Tiêu đề = các dòng uppercase sau doc type header
            if line_stripped.isupper() or (len(line_stripped) > 5 and
                    sum(1 for c in line_stripped if c.isupper()) > len(line_stripped) * 0.5):
                title_lines.append(line_stripped)
            elif title_lines:
                break

    if title_lines:
        title = " ".join(title_lines)
        # Clean up
        title = re.sub(r"\s+", " ", title).strip()
        # Title case
        if title.isupper():
            title = title.title()
        return title

    # Fallback: dùng doc_type + doc_number
    if doc_type and doc_number:
        return f"{doc_type} {doc_number}"

    return ""


def extract_effective_date(text: str) -> str:
    """Extract ngày hiệu lực."""
    patterns = [
        r"có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:từ|kể\s+từ)\s+" + DATE_PATTERN,
    ]
    for p in patterns:
        match = re.search(p, text[-5000:], re.IGNORECASE)
        if match:
            day, month, year = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
    return ""


def detect_sectors(text: str, url: str) -> list[str]:
    """Detect lĩnh vực từ nội dung và URL."""
    sectors = set()
    text_lower = text.lower()
    url_lower = url.lower()

    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower or kw.replace(" ", "-") in url_lower:
                sectors.add(sector)
                break

    return sorted(sectors) if sectors else ["Nội vụ (chung)"]


def parse_txt_file(filepath: Path) -> dict | None:
    """Parse 1 file .txt thành document dict."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="utf-8-sig")
        except Exception as e:
            logger.error(f"Không đọc được {filepath.name}: {e}")
            return None

    lines = text.split("\n")
    if len(lines) < 5:
        logger.warning(f"File quá ngắn: {filepath.name} ({len(lines)} lines)")
        return None

    # Extract URL
    url, content_start = extract_url(lines)

    # Content (bỏ URL line)
    content = "\n".join(lines[content_start:]).strip()

    # Bỏ phần tiếng Anh ở cuối (nếu có)
    en_marker = re.search(r"\n\s*(?:THE\s+NATIONAL\s+ASSEMBLY|GOVERNMENT|SOCIALIST\s+REPUBLIC)", content)
    if en_marker and en_marker.start() > len(content) * 0.7:
        content = content[:en_marker.start()].strip()

    # Extract metadata
    doc_type = extract_doc_type(content, filepath.stem)
    doc_number = extract_doc_number(content)
    issuer = extract_issuer(content)
    issue_date = extract_date(content)
    title = extract_title(content, doc_type, doc_number)
    effective_date = extract_effective_date(content)
    sectors = detect_sectors(content, url)

    # Generate doc_id từ filename
    doc_id = filepath.stem

    return {
        "doc_id": doc_id,
        "title": title,
        "document_number": doc_number,
        "doc_type": doc_type,
        "issuer": issuer,
        "issue_date": issue_date,
        "effective_date": effective_date,
        "source_url": url,
        "sectors": sectors,
        "source_file": filepath.name,
        "content": content,
        "content_length": len(content),
    }


def main():
    parser = argparse.ArgumentParser(description="Import raw-data → documents.jsonl")
    parser.add_argument("--input", type=str, default=None,
                        help="Thư mục chứa file .txt")
    parser.add_argument("--output", type=str, default=None,
                        help="File output .jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ xem trước, không ghi file")
    args = parser.parse_args()

    # Paths
    input_dir = Path(args.input) if args.input else _ROOT / "data" / "raw-data"
    output_path = Path(args.output) if args.output else _ROOT / "processed" / "documents.jsonl"

    logger.info("=" * 60)
    logger.info("IMPORT RAW-DATA → documents.jsonl")
    logger.info(f"  Input:  {input_dir}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Mode:   {'DRY RUN' if args.dry_run else 'WRITE'}")
    logger.info("=" * 60)

    if not input_dir.exists():
        logger.error(f"Thư mục không tồn tại: {input_dir}")
        sys.exit(1)

    # Find all .txt files (exclude README)
    txt_files = sorted([
        f for f in input_dir.glob("*.txt")
        if f.stem.upper() != "README"
    ])

    logger.info(f"Tìm thấy {len(txt_files)} file .txt")

    # Parse
    documents = []
    errors = []
    type_counter = Counter()
    sector_counter = Counter()

    for filepath in txt_files:
        doc = parse_txt_file(filepath)
        if doc:
            documents.append(doc)
            type_counter[doc["doc_type"]] += 1
            for s in doc["sectors"]:
                sector_counter[s] += 1
            logger.info(
                f"  ✅ {filepath.name} → "
                f"{doc['doc_type']} {doc['document_number'] or '(no number)'} "
                f"| {doc['title'][:50]}..."
            )
        else:
            errors.append(filepath.name)
            logger.error(f"  ❌ {filepath.name} → FAILED")

    # Stats
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 THỐNG KÊ:")
    logger.info(f"  Tổng files: {len(txt_files)}")
    logger.info(f"  Thành công: {len(documents)}")
    logger.info(f"  Lỗi: {len(errors)}")
    logger.info("")
    logger.info("  📋 Phân bố theo loại VB:")
    for dtype, count in type_counter.most_common():
        logger.info(f"    {dtype}: {count}")
    logger.info("")
    logger.info("  🏷️ Phân bố theo lĩnh vực:")
    for sector, count in sector_counter.most_common():
        logger.info(f"    {sector}: {count}")
    logger.info("")

    total_chars = sum(d["content_length"] for d in documents)
    logger.info(f"  📏 Tổng nội dung: {total_chars:,} chars ({total_chars/1024/1024:.1f} MB)")
    logger.info("=" * 60)

    # Write output
    if args.dry_run:
        logger.info("🔍 DRY RUN — không ghi file")
        # Show sample
        if documents:
            sample = documents[0].copy()
            sample["content"] = sample["content"][:200] + "..."
            logger.info(f"\nSample record:\n{json.dumps(sample, ensure_ascii=False, indent=2)}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"✅ Đã ghi {len(documents)} documents → {output_path}")
        logger.info(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    if errors:
        logger.warning(f"\n⚠️ Files lỗi: {errors}")


if __name__ == "__main__":
    main()
