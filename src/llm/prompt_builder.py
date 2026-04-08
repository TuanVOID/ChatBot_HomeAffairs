"""
Prompt Builder — Tạo prompt chuyên ngành Nội vụ cho LLM.
"""


SYSTEM_PROMPT = """Bạn là **Trợ lý Pháp luật Nội vụ**, hệ thống tra cứu chuyên ngành các văn bản quy phạm pháp luật thuộc lĩnh vực Bộ Nội vụ Việt Nam.

📋 **Phạm vi chuyên môn:**
- Quản lý tổ chức bộ máy và biên chế
- Quản lý cán bộ, công chức, viên chức (tuyển dụng, bổ nhiệm, kỷ luật, đào tạo)
- Xây dựng chính quyền địa phương, đơn vị hành chính
- Cải cách hành chính
- Quản lý lao động, tiền lương, phụ cấp
- Thi đua, khen thưởng
- Quản lý nhà nước về tôn giáo, tín ngưỡng
- Văn thư - lưu trữ
- Công tác thanh niên
- Bảo hiểm xã hội, người có công

📝 **Quy tắc BẮT BUỘC:**
1. CHỈ trả lời DỰA TRÊN các trích đoạn văn bản pháp luật được cung cấp bên dưới.
2. Luôn TRÍCH DẪN NGUỒN cụ thể: tên văn bản, số hiệu, Điều, Khoản, Điểm.
3. Nếu thông tin KHÔNG ĐỦ → nói rõ: "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
4. KHÔNG bịa thông tin. KHÔNG suy luận ngoài context.
5. CHỈ trả lời bằng tiếng Việt.
6. Trả lời có cấu trúc, dùng markdown (bullet points, **bold**, heading) cho dễ đọc.
7. Nếu câu hỏi ngoài phạm vi Nội vụ → hướng dẫn tra cứu đúng lĩnh vực.
8. Khi trả lời xong → DỪNG NGAY. Không thêm nội dung thừa.

⚠️ Đây là hệ thống hỗ trợ tra cứu, KHÔNG thay thế tư vấn pháp lý chuyên nghiệp."""


GREETING_SUGGESTIONS = [
    "Điều kiện tuyển dụng công chức theo Luật Cán bộ, công chức?",
    "Quy định về kỷ luật viên chức hiện hành?",
    "Mức lương cơ sở mới nhất áp dụng cho cán bộ, công chức?",
    "Thủ tục sáp nhập đơn vị hành chính cấp xã?",
    "Quy trình xét thi đua, khen thưởng Huân chương Lao động?",
    "Điều kiện thành lập tổ chức tôn giáo theo Luật Tín ngưỡng?",
]


def build_rag_prompt(query: str, contexts: list[dict],
                     history: list[dict] = None) -> list[dict]:
    """
    Xây dựng messages cho Ollama chat API.

    Args:
        query: Câu hỏi của user
        contexts: List of retrieval results
        history: Chat history [{role, content}, ...]

    Returns:
        List of messages cho Ollama chat API
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history (giới hạn 10 turns gần nhất)
    if history:
        for h in history[-10:]:
            messages.append({"role": h["role"], "content": h["content"]})

    # Build context string
    if contexts:
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            path = ctx.get("path", "")
            title = ctx.get("title", "")
            doc_num = ctx.get("document_number", "")
            doc_type = ctx.get("doc_type", "")
            issuer = ctx.get("issuer", "")
            text = ctx.get("text", "")

            header = f"[{i}]"
            if doc_type:
                header += f" {doc_type}"
            if doc_num:
                header += f" {doc_num}"
            if title:
                header += f" — {title[:80]}"
            if path and path != title:
                header += f"\n    📍 {path}"
            if issuer:
                header += f"\n    🏛️ {issuer}"

            context_parts.append(f"{header}\n{text}")

        context_str = "\n\n---\n\n".join(context_parts)

        user_content = f"""Dưới đây là các trích đoạn văn bản pháp luật liên quan:

{context_str}

---

Câu hỏi: {query}

Hãy trả lời dựa trên các trích đoạn trên. Trích dẫn nguồn cụ thể (số hiệu, Điều, Khoản)."""
    else:
        user_content = f"""Không tìm thấy văn bản pháp luật liên quan trong cơ sở dữ liệu.

Câu hỏi: {query}

Hãy thông báo rằng không có thông tin phù hợp và gợi ý cách đặt câu hỏi khác."""

    messages.append({"role": "user", "content": user_content})
    return messages


def build_search_summary(contexts: list[dict]) -> list[dict]:
    """Tạo danh sách sources cho UI hiển thị citations."""
    sources = []
    seen = set()
    for ctx in contexts:
        doc_id = ctx.get("doc_id", "")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        sources.append({
            "title": ctx.get("title", ""),
            "document_number": ctx.get("document_number", ""),
            "doc_type": ctx.get("doc_type", ""),
            "issuer": ctx.get("issuer", ""),
            "issue_date": ctx.get("issue_date", ""),
            "article": ctx.get("article", ""),
            "path": ctx.get("path", ""),
        })
    return sources
