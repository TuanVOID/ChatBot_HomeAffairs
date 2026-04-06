"""
Legal Structure Parser — Parse cấu trúc văn bản pháp luật Việt Nam.

Hỗ trợ nhận diện:
  - Phần (PHẦN THỨ NHẤT, PHẦN I, ...)
  - Chương (Chương I, Chương 1, ...)
  - Mục (Mục 1, Mục I, ...)
  - Điều (Điều 1, Điều 1a, ...)
  - Khoản (1. , 2. , ...)
  - Điểm (a) , b) , đ) , ...)
"""

import re
from dataclasses import dataclass, field


# ── Regex patterns cho cấu trúc pháp luật ──

# Phần: "PHẦN THỨ NHẤT", "PHẦN I", "Phần 1"
RE_PHAN = re.compile(
    r'^(?:PHẦN\s+(?:THỨ\s+)?(?:[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]+|\d+))',
    re.MULTILINE
)

# Chương: "Chương I", "Chương 1", "CHƯƠNG I"
RE_CHUONG = re.compile(
    r'^(?:Chương|CHƯƠNG)\s+([IVXLCDM]+|\d+)[\.\s:]?(.*)',
    re.MULTILINE
)

# Mục: "Mục 1", "MỤC 1"  
RE_MUC = re.compile(
    r'^(?:Mục|MỤC)\s+(\d+|[IVXLCDM]+)[\.\s:]?(.*)',
    re.MULTILINE
)

# Điều: "Điều 1.", "Điều 1a.", "ĐIỀU 1"
RE_DIEU = re.compile(
    r'^(?:Điều|ĐIỀU)\s+(\d+[a-zđ]?)[\.\s:]?(.*)',
    re.MULTILINE
)

# Khoản: "1. ", "2. ", "10. " (đầu dòng, theo sau bởi text)
RE_KHOAN = re.compile(
    r'^(\d+)\.\s+(.+)',
    re.MULTILINE
)

# Điểm: "a) ", "b) ", "đ) ", "dd) "
RE_DIEM = re.compile(
    r'^([a-zđ]{1,2})\)\s+(.+)',
    re.MULTILINE
)


@dataclass
class LegalNode:
    """Một node trong cây cấu trúc pháp luật."""
    level: str       # 'phan', 'chuong', 'muc', 'dieu', 'khoan', 'diem'
    number: str      # 'I', '1', 'a', etc.
    title: str       # Tiêu đề (nếu có)
    text: str        # Nội dung text
    start_pos: int   # Vị trí bắt đầu trong content gốc
    end_pos: int     # Vị trí kết thúc


@dataclass
class ParsedArticle:
    """Một Điều đã được parse, chứa các Khoản/Điểm bên trong."""
    article_number: str       # "1", "2a", etc.
    article_title: str        # "Phạm vi điều chỉnh"
    full_text: str            # Toàn bộ text của Điều
    chapter: str = ""         # "Chương I"
    section: str = ""         # "Mục 1" (nếu có)
    clauses: list = field(default_factory=list)  # List of (number, text)


def find_header_end(content: str) -> int:
    """
    Tìm vị trí kết thúc header (phần trước nội dung chính).
    Header thường kết thúc ở "QUYẾT ĐỊNH:", "QUYẾT NGHỊ:", "Điều 1", etc.
    """
    # Tìm pattern kết thúc preamble
    patterns = [
        r'QUYẾT ĐỊNH:',
        r'QUYẾT NGHỊ:',
        r'NGHỊ QUYẾT:',
        r'^Điều\s+1[\.\s]',
    ]
    
    min_pos = len(content)
    for pat in patterns:
        m = re.search(pat, content, re.MULTILINE)
        if m and m.start() < min_pos:
            min_pos = m.start()
    
    # Nếu không tìm thấy pattern nào, lấy từ đầu
    if min_pos == len(content):
        return 0
    
    return min_pos


def split_into_articles(content: str) -> list[ParsedArticle]:
    """
    Tách content thành danh sách các Điều (article).
    Mỗi Điều bao gồm toàn bộ text từ "Điều X" đến trước "Điều X+1".
    """
    articles = []
    
    # Tìm tất cả vị trí "Điều X"
    matches = list(RE_DIEU.finditer(content))
    
    if not matches:
        return articles
    
    # Track chapter/section context
    current_chapter = ""
    current_section = ""
    
    # Scan text trước mỗi Điều để tìm Chương/Mục
    full_text_before = content[:matches[0].start()] if matches else ""
    
    for i, m in enumerate(matches):
        article_num = m.group(1)
        article_title = m.group(2).strip().rstrip('.')
        
        # Xác định end position
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # Lấy text giữa vị trí match trước và match hiện tại để tìm Chương/Mục
        if i > 0:
            between_text = content[matches[i-1].end():m.start()]
        else:
            between_text = content[:m.start()]
        
        # Check for Chương
        chuong_matches = list(RE_CHUONG.finditer(between_text))
        if chuong_matches:
            last = chuong_matches[-1]
            current_chapter = f"Chương {last.group(1)}"
            chapter_title = last.group(2).strip()
            if chapter_title:
                current_chapter += f". {chapter_title}"
        
        # Check for Mục
        muc_matches = list(RE_MUC.finditer(between_text))
        if muc_matches:
            last = muc_matches[-1]
            current_section = f"Mục {last.group(1)}"
            section_title = last.group(2).strip()
            if section_title:
                current_section += f". {section_title}"
        
        # Extract full text of this article
        full_text = content[m.start():end_pos].strip()
        
        # Parse clauses (Khoản) within article
        clauses = parse_clauses(full_text)
        
        articles.append(ParsedArticle(
            article_number=article_num,
            article_title=article_title,
            full_text=full_text,
            chapter=current_chapter,
            section=current_section,
            clauses=clauses,
        ))
    
    return articles


def parse_clauses(article_text: str) -> list[dict]:
    """
    Parse các Khoản trong một Điều.
    Returns list of {'number': '1', 'text': '...', 'points': [{'letter': 'a', 'text': '...'}]}
    """
    clauses = []
    
    # Tìm tất cả Khoản
    # Bỏ qua dòng đầu (tiêu đề Điều)
    lines = article_text.split('\n')
    
    # Tìm vị trí bắt đầu nội dung (sau tiêu đề Điều)
    clause_matches = list(RE_KHOAN.finditer(article_text))
    
    if not clause_matches:
        # Không có Khoản → cả Điều là 1 chunk
        return []
    
    for i, m in enumerate(clause_matches):
        clause_num = m.group(1)
        
        # End position
        if i + 1 < len(clause_matches):
            end_pos = clause_matches[i + 1].start()
        else:
            end_pos = len(article_text)
        
        clause_text = article_text[m.start():end_pos].strip()
        
        # Parse điểm (points) within clause
        points = parse_points(clause_text)
        
        clauses.append({
            'number': clause_num,
            'text': clause_text,
            'points': points,
        })
    
    return clauses


def parse_points(clause_text: str) -> list[dict]:
    """Parse các Điểm (a), b), ...) trong một Khoản."""
    points = []
    point_matches = list(RE_DIEM.finditer(clause_text))
    
    for i, m in enumerate(point_matches):
        letter = m.group(1)
        if i + 1 < len(point_matches):
            end_pos = point_matches[i + 1].start()
        else:
            end_pos = len(clause_text)
        
        point_text = clause_text[m.start():end_pos].strip()
        points.append({'letter': letter, 'text': point_text})
    
    return points


def build_breadcrumb(doc_title: str, chapter: str, section: str,
                     article_num: str, clause_num: str = "",
                     point_letter: str = "") -> str:
    """
    Xây dựng breadcrumb path cho chunk.
    Ví dụ: "Luật Đất đai 2024 > Chương I > Điều 1 > Khoản 2 > Điểm a"
    """
    parts = []
    
    # Rút gọn title (bỏ phần quá dài)
    short_title = doc_title
    if len(short_title) > 60:
        short_title = short_title[:57] + "..."
    parts.append(short_title)
    
    if chapter:
        parts.append(chapter.split('.')[0].strip())  # Chỉ lấy "Chương I", bỏ tiêu đề
    if section:
        parts.append(section.split('.')[0].strip())
    
    parts.append(f"Điều {article_num}")
    
    if clause_num:
        parts.append(f"Khoản {clause_num}")
    if point_letter:
        parts.append(f"Điểm {point_letter}")
    
    return " > ".join(parts)


def detect_footer_start(content: str) -> int:
    """
    Phát hiện phần footer (Nơi nhận, chữ ký, ...).
    Trả về vị trí bắt đầu footer hoặc len(content) nếu không có.
    """
    patterns = [
        r'^Nơi nhận:',
        r'^\|\s*TM\.\s',
        r'^\|\s*KT\.\s',
        r'^\|\s*CHỦ TỊCH',
        r'^\|\s*BỘ TRƯỞNG',
        r'^\|\s*TỔNG GIÁM ĐỐC',
    ]
    
    min_pos = len(content)
    for pat in patterns:
        m = re.search(pat, content, re.MULTILINE)
        if m and m.start() < min_pos:
            # Chỉ coi là footer nếu ở 30% cuối document
            if m.start() > len(content) * 0.5:
                min_pos = m.start()
    
    return min_pos
