"""
Query Preprocessor — Làm giàu câu hỏi trước khi retrieval.

Chức năng:
  1. Phát hiện query không dấu → gọi LLM thêm dấu tiếng Việt
  2. Phát hiện tiếng Anh → trả message yêu cầu viết tiếng Việt
  3. Chuẩn hóa: strip, lowercase keywords, etc.
"""

import re
import httpx
import json
from loguru import logger


# ── Vietnamese diacritics detection ──
# Ký tự có dấu đặc trưng tiếng Việt
_VIET_DIACRITICS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
                       "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")

# Các từ tiếng Việt phổ biến không dấu (để detect query VN không dấu)
_COMMON_VN_WORDS = {
    "la", "cua", "va", "trong", "cho", "voi", "co", "khong", "nhu", "thi",
    "duoc", "nhung", "cac", "mot", "nay", "day", "tai", "ve", "theo",
    "luat", "dieu", "khoan", "nghi", "dinh", "quyet", "thong", "tu",
    "cong", "chuc", "vien", "bo", "noi", "vu", "hanh", "chinh",
    "nha", "nuoc", "can", "tien", "luong", "bao", "hiem", "xa", "hoi",
    "thi", "dua", "khen", "thuong", "giao", "duc", "dao", "tao",
    "quyen", "nghia", "trach", "nhiem", "quy", "trinh", "thu", "tuc",
    "don", "vi", "to", "chuc", "nhan", "su", "tuyen", "dung",
    "de", "bat", "bo", "nhiem", "ky", "luat", "sa", "thai",
    "huong", "dan", "qd", "nd", "tt", "nq", "bhxh",
    "chinh", "quyen", "dia", "phuong", "ton", "giao", "tin", "nguong",
    "thanh", "nien", "dan", "chu", "co", "so", "luu", "tru",
    "cai", "cach", "thanh", "tra", "nguoi", "cong",
}


class QueryPreprocessor:
    """
    Tiền xử lý câu hỏi trước khi đưa vào retrieval pipeline.
    
    Pipeline:
      1. Detect ngôn ngữ (Việt/Anh/Việt không dấu)
      2. Nếu Việt không dấu → LLM thêm dấu
      3. Nếu tiếng Anh → trả thông báo yêu cầu viết tiếng Việt
      4. Chuẩn hóa query
    """
    
    def __init__(self, ollama_url: str, chat_model: str):
        self._ollama_url = ollama_url
        self._chat_model = chat_model
        logger.info(f"QueryPreprocessor initialized (model={chat_model})")
    
    def process(self, query: str) -> dict:
        """
        Xử lý query, trả về dict:
        {
            "original": str,          # Query gốc
            "processed": str,         # Query đã xử lý (có dấu, chuẩn hóa)
            "lang": str,              # "vi", "vi_no_accent", "en", "mixed"
            "rejected": bool,         # True nếu từ chối xử lý (ví dụ: tiếng Anh)
            "reject_message": str,    # Thông báo cho user nếu rejected
            "enriched": bool,         # True nếu query đã được thêm dấu/cải thiện
        }
        """
        query = query.strip()
        if not query:
            return {
                "original": query,
                "processed": query,
                "lang": "unknown",
                "rejected": True,
                "reject_message": "Vui lòng nhập câu hỏi.",
                "enriched": False,
            }
        
        # Detect language
        lang = self._detect_language(query)
        
        # Case 1: Tiếng Anh → nhắc viết tiếng Việt
        if lang == "en":
            return {
                "original": query,
                "processed": query,
                "lang": lang,
                "rejected": True,
                "reject_message": (
                    "⚠️ Hệ thống chuyên trả lời pháp luật **tiếng Việt**.\n\n"
                    "Vui lòng đặt câu hỏi bằng **tiếng Việt** để được hỗ trợ chính xác nhất.\n\n"
                    "Ví dụ: *\"Công chức có những nghĩa vụ gì?\"*"
                ),
                "enriched": False,
            }
        
        # Case 2: Tiếng Việt không dấu → thêm dấu
        if lang == "vi_no_accent":
            enriched_query = self._add_diacritics(query)
            if enriched_query and enriched_query != query:
                logger.info(f"Query enriched: '{query}' → '{enriched_query}'")
                return {
                    "original": query,
                    "processed": enriched_query,
                    "lang": lang,
                    "rejected": False,
                    "reject_message": "",
                    "enriched": True,
                }
        
        # Case 3: Tiếng Việt có dấu → dùng nguyên
        return {
            "original": query,
            "processed": query,
            "lang": lang,
            "rejected": False,
            "reject_message": "",
            "enriched": False,
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Phát hiện ngôn ngữ:
        - "vi": tiếng Việt có dấu
        - "vi_no_accent": tiếng Việt không dấu
        - "en": tiếng Anh
        - "mixed": hỗn hợp
        """
        # Đếm ký tự có dấu Việt
        vn_chars = sum(1 for c in text if c in _VIET_DIACRITICS)
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        if alpha_chars == 0:
            return "mixed"
        
        vn_ratio = vn_chars / alpha_chars
        
        # Nếu có nhiều ký tự dấu → tiếng Việt
        if vn_ratio > 0.05:
            return "vi"
        
        # Không có dấu → kiểm tra xem có phải VN không dấu hay English
        words = re.findall(r'[a-zA-Z]+', text.lower())
        if not words:
            return "mixed"
        
        vn_word_count = sum(1 for w in words if w in _COMMON_VN_WORDS)
        vn_word_ratio = vn_word_count / len(words)
        
        # Nếu >40% từ là VN phổ biến → VN không dấu
        if vn_word_ratio > 0.35:
            return "vi_no_accent"
        
        # Kiểm tra thêm: nếu có keyword pháp luật VN → VN không dấu
        legal_keywords = {"luat", "dieu", "khoan", "nghi", "dinh", "quyet",
                         "thong", "tu", "cong", "chuc", "vien", "bhxh",
                         "nd", "tt", "qd", "nq"}
        if any(w in legal_keywords for w in words):
            return "vi_no_accent"
        
        # Mặc định: tiếng Anh
        return "en"
    
    def _add_diacritics(self, text: str) -> str:
        """Dùng LLM để thêm dấu tiếng Việt cho text không dấu."""
        prompt = f"""Hãy thêm dấu tiếng Việt cho câu sau. Quy tắc:
- CHỈ thêm dấu, KHÔNG thay đổi từ ngữ, KHÔNG thêm/bớt từ
- Giữ nguyên cấu trúc câu và số lượng từ
- Trả về DUY NHẤT câu đã thêm dấu, không giải thích

Ví dụ:
Input: cong chuc co nhung nghia vu gi?
Output: Công chức có những nghĩa vụ gì?

Input: luat can bo cong chuc quy dinh ve tuyen dung nhu the nao
Output: Luật cán bộ công chức quy định về tuyển dụng như thế nào

Input: {text}
Output:"""
        
        try:
            r = httpx.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._chat_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                    },
                },
                timeout=15,
            )
            r.raise_for_status()
            result = r.json().get("response", "").strip()
            
            # Cleanup: lấy dòng đầu tiên, bỏ dấu ngoặc/prefix
            result = result.split("\n")[0].strip()
            result = re.sub(r'^["\']|["\']$', '', result)  # Bỏ quotes
            
            # Validate: kết quả phải có dấu và không quá dài
            if result and len(result) < len(text) * 3:
                vn_chars = sum(1 for c in result if c in _VIET_DIACRITICS)
                if vn_chars > 0:  # Phải có ít nhất 1 ký tự dấu
                    return result
            
            logger.warning(f"Diacritics enrichment returned invalid: '{result}'")
            return text
            
        except Exception as e:
            logger.warning(f"Diacritics enrichment failed: {e}")
            return text
