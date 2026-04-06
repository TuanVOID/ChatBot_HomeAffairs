"""
Prompt Builder — Tạo prompt cho LLM từ context + query.
"""


SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật Việt Nam. Quy tắc BẮT BUỘC:

1. CHỈ trả lời DỰA TRÊN các trích đoạn văn bản pháp luật được cung cấp bên dưới.
2. Luôn TRÍCH DẪN nguồn: tên văn bản, số hiệu, Điều, Khoản, Điểm cụ thể.
3. Nếu thông tin trong context KHÔNG ĐỦ → nói rõ "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu hiện có."
4. KHÔNG bịa thông tin. KHÔNG suy luận ngoài context.
5. CHỈ trả lời bằng tiếng Việt. TUYỆT ĐỐI KHÔNG viết tiếng Trung, tiếng Anh hoặc ngôn ngữ khác.
6. Trả lời ngắn gọn, đúng trọng tâm. Dùng markdown (bullet points, **bold**) cho dễ đọc.
7. Khi đã trả lời xong câu hỏi → DỪNG NGAY. Không thêm nội dung không liên quan.

⚠️ Đây là hệ thống hỗ trợ tra cứu, KHÔNG thay thế tư vấn pháp lý chuyên nghiệp."""


def build_rag_prompt(query: str, contexts: list[dict],
                     history: list[dict] = None) -> list[dict]:
    """
    Xây dựng messages cho Ollama chat API.

    Args:
        query: Câu hỏi của user
        contexts: List of retrieval results (mỗi item có 'text', 'path', 'title', ...)
        history: Chat history [{role, content}, ...]

    Returns:
        List of messages cho Ollama chat API
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history (nếu có)
    if history:
        for h in history[-10:]:  # Giới hạn 10 turns
            messages.append({"role": h["role"], "content": h["content"]})

    # Build context string
    if contexts:
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            path = ctx.get("path", "")
            title = ctx.get("title", "")
            doc_num = ctx.get("document_number", "")
            text = ctx.get("text", "")

            header = f"[{i}] {path}"
            if doc_num:
                header += f" ({doc_num})"
            context_parts.append(f"{header}\n{text}")

        context_str = "\n\n---\n\n".join(context_parts)

        user_content = f"""Dưới đây là các trích đoạn văn bản pháp luật liên quan:

{context_str}

---

Câu hỏi: {query}

Hãy trả lời dựa trên các trích đoạn trên. Trích dẫn nguồn cụ thể."""
    else:
        user_content = f"""Không tìm thấy văn bản pháp luật liên quan đến câu hỏi này.

Câu hỏi: {query}

Hãy trả lời rằng không có thông tin trong cơ sở dữ liệu hiện tại."""

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
