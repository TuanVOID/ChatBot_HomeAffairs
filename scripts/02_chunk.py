"""
Phase 2 — Parse Structure & Chunk
====================================
Đọc documents.jsonl, parse cấu trúc pháp luật (Chương/Điều/Khoản),
tạo chunks với metadata breadcrumb, xuất chunks.jsonl.

Usage:
    python scripts/02_chunk.py                    # Chunk tất cả documents
    python scripts/02_chunk.py --limit 100        # Test với 100 docs

Output:
    processed/chunks.jsonl    — mỗi dòng = 1 chunk JSON
    processed/manifest.json   — cập nhật thông tin chunking
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Thêm project root vào sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg
from scripts.utils.text_utils import clean_legal_text, estimate_tokens
from scripts.utils.legal_parser import (
    split_into_articles,
    build_breadcrumb,
    find_header_end,
    detect_footer_start,
)

# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "02_chunk.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")


def chunk_document(doc: dict) -> list[dict]:
    """
    Tách 1 document thành nhiều chunks dựa trên cấu trúc pháp luật.
    
    Strategy:
    1. Parse Điều → nếu Điều có Khoản → chunk theo Khoản
    2. Nếu Điều không có Khoản → cả Điều = 1 chunk
    3. Nếu Điều quá dài (>MAX_CHUNK_TOKENS) → split thêm
    4. Nếu Điều/Khoản quá ngắn (<MIN_CHUNK_TOKENS) → merge với context
    5. Docs không có cấu trúc Điều → sliding window fallback
    """
    content = doc.get("content", "")
    doc_id = doc["doc_id"]
    title = doc.get("title", "")
    
    if not content or len(content) < cfg.MIN_CONTENT_LENGTH:
        return []
    
    # Cắt bỏ footer (Nơi nhận, chữ ký)
    footer_pos = detect_footer_start(content)
    body = content[:footer_pos].strip()
    
    # Parse articles
    articles = split_into_articles(body)
    
    if not articles:
        # Fallback: sliding window cho docs không có cấu trúc Điều
        return _sliding_window_chunks(doc, body)
    
    chunks = []
    
    for article in articles:
        article_header = f"Điều {article.article_number}"
        if article.article_title:
            article_header += f". {article.article_title}"
        
        if article.clauses:
            # Có Khoản → chunk theo Khoản
            for clause in article.clauses:
                clause_num = clause['number']
                clause_text = clause['text']
                
                # Thêm context: tiêu đề Điều phía trước
                chunk_text = f"{article_header}\n\n{clause_text}"
                
                tokens = estimate_tokens(chunk_text)
                
                # Nếu quá dài, split thêm
                if tokens > cfg.MAX_CHUNK_TOKENS:
                    sub_chunks = _split_long_text(
                        chunk_text, cfg.MAX_CHUNK_TOKENS
                    )
                    for si, sub_text in enumerate(sub_chunks):
                        chunk_id = f"{doc_id}_d{article.article_number}_k{clause_num}_s{si}"
                        chunks.append(_make_chunk(
                            doc, chunk_id, sub_text,
                            article, clause_num, "",
                            f"(phần {si+1})"
                        ))
                elif tokens >= cfg.MIN_CHUNK_TOKENS:
                    chunk_id = f"{doc_id}_d{article.article_number}_k{clause_num}"
                    chunks.append(_make_chunk(
                        doc, chunk_id, chunk_text,
                        article, clause_num, ""
                    ))
                else:
                    # Quá ngắn → gộp vào chunk trước hoặc tạo riêng
                    chunk_id = f"{doc_id}_d{article.article_number}_k{clause_num}"
                    chunks.append(_make_chunk(
                        doc, chunk_id, chunk_text,
                        article, clause_num, ""
                    ))
        else:
            # Không có Khoản → cả Điều = 1 chunk
            chunk_text = article.full_text
            tokens = estimate_tokens(chunk_text)
            
            if tokens > cfg.MAX_CHUNK_TOKENS:
                sub_chunks = _split_long_text(
                    chunk_text, cfg.MAX_CHUNK_TOKENS
                )
                for si, sub_text in enumerate(sub_chunks):
                    chunk_id = f"{doc_id}_d{article.article_number}_s{si}"
                    chunks.append(_make_chunk(
                        doc, chunk_id, sub_text,
                        article, "", "",
                        f"(phần {si+1})"
                    ))
            else:
                chunk_id = f"{doc_id}_d{article.article_number}"
                chunks.append(_make_chunk(
                    doc, chunk_id, chunk_text,
                    article, "", ""
                ))
    
    # Nếu parse ra 0 chunks dù có articles, fallback
    if not chunks:
        return _sliding_window_chunks(doc, body)
    
    return chunks


def _make_chunk(doc: dict, chunk_id: str, text: str,
                article, clause_num: str, point_letter: str,
                suffix: str = "") -> dict:
    """Tạo chunk dict với đầy đủ metadata."""
    title = doc.get("title", "")
    
    path = build_breadcrumb(
        title,
        article.chapter,
        article.section,
        article.article_number,
        clause_num,
        point_letter,
    )
    if suffix:
        path += f" {suffix}"
    
    # Tạo text_for_keyword: title + path + text (cho BM25)
    text_for_keyword = f"{title} {doc.get('document_number', '')} {path} {text}"
    
    return {
        "chunk_id": chunk_id,
        "doc_id": doc["doc_id"],
        "title": title,
        "document_number": doc.get("document_number", ""),
        "doc_type": doc.get("doc_type", ""),
        "issuer": doc.get("issuer", ""),
        "issue_date": doc.get("issue_date", ""),
        "chapter": article.chapter if article else "",
        "section": article.section if article else "",
        "article": f"Điều {article.article_number}" if article else "",
        "clause": f"Khoản {clause_num}" if clause_num else "",
        "point": f"Điểm {point_letter}" if point_letter else "",
        "path": path,
        "text": text.strip(),
        "text_for_keyword": clean_legal_text(text_for_keyword),
        "token_count": estimate_tokens(text),
    }


def _sliding_window_chunks(doc: dict, text: str) -> list[dict]:
    """
    Fallback chunking cho docs không có cấu trúc Điều.
    Dùng sliding window với overlap.
    """
    chunks = []
    words = text.split()
    
    # Ước tính window size theo tokens
    window_words = int(cfg.MAX_CHUNK_TOKENS / 1.5)  # ~tokens/1.5 = words
    overlap_words = int(cfg.OVERLAP_TOKENS / 1.5)
    
    if len(words) <= window_words:
        # Document ngắn → 1 chunk
        chunk_id = f"{doc['doc_id']}_full"
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc["doc_id"],
            "title": doc.get("title", ""),
            "document_number": doc.get("document_number", ""),
            "doc_type": doc.get("doc_type", ""),
            "issuer": doc.get("issuer", ""),
            "issue_date": doc.get("issue_date", ""),
            "chapter": "",
            "section": "",
            "article": "",
            "clause": "",
            "point": "",
            "path": doc.get("title", "")[:60],
            "text": text.strip(),
            "text_for_keyword": clean_legal_text(
                f"{doc.get('title', '')} {doc.get('document_number', '')} {text}"
            ),
            "token_count": estimate_tokens(text),
        }
        return [chunk]
    
    start = 0
    chunk_idx = 0
    
    while start < len(words):
        end = min(start + window_words, len(words))
        chunk_text = " ".join(words[start:end])
        
        chunk_id = f"{doc['doc_id']}_w{chunk_idx}"
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc["doc_id"],
            "title": doc.get("title", ""),
            "document_number": doc.get("document_number", ""),
            "doc_type": doc.get("doc_type", ""),
            "issuer": doc.get("issuer", ""),
            "issue_date": doc.get("issue_date", ""),
            "chapter": "",
            "section": "",
            "article": "",
            "clause": "",
            "point": "",
            "path": f"{doc.get('title', '')[:50]} (phần {chunk_idx+1})",
            "text": chunk_text.strip(),
            "text_for_keyword": clean_legal_text(
                f"{doc.get('title', '')} {doc.get('document_number', '')} {chunk_text}"
            ),
            "token_count": estimate_tokens(chunk_text),
        }
        chunks.append(chunk)
        
        if end >= len(words):
            break
        start = end - overlap_words
        chunk_idx += 1
    
    return chunks


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split text dài thành nhiều phần, cố gắng cắt ở paragraph boundary."""
    paragraphs = text.split('\n\n')
    
    result = []
    current = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        
        if current_tokens + para_tokens > max_tokens and current:
            result.append('\n\n'.join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens
    
    if current:
        result.append('\n\n'.join(current))
    
    return result if result else [text]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Parse Structure & Chunk")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số documents xử lý")
    parser.add_argument("--input", type=str, default=None,
                        help="Path tới documents.jsonl (mặc định: processed/documents.jsonl)")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t_start = time.time()
    
    input_path = Path(args.input) if args.input else cfg.PROCESSED_DIR / "documents.jsonl"
    output_path = cfg.PROCESSED_DIR / "chunks.jsonl"
    
    logger.info("=" * 60)
    logger.info("PHASE 2: PARSE STRUCTURE & CHUNK")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Max chunk tokens: {cfg.MAX_CHUNK_TOKENS}")
    logger.info(f"  Min chunk tokens: {cfg.MIN_CHUNK_TOKENS}")
    logger.info(f"  Limit: {args.limit or 'ALL'}")
    logger.info("=" * 60)
    
    if not input_path.exists():
        logger.error(f"Input file không tồn tại: {input_path}")
        logger.error("Hãy chạy Phase 1 trước: python scripts/01_ingest.py")
        sys.exit(1)
    
    # ── Step 1: Đọc documents ──
    logger.info("Step 1: Loading documents...")
    documents = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            documents.append(json.loads(line.strip()))
    logger.info(f"  Loaded {len(documents):,} documents")
    
    # ── Step 2: Chunk từng document ──
    logger.info("Step 2: Chunking documents...")
    all_chunks = []
    docs_with_articles = 0
    docs_fallback = 0
    
    for i, doc in enumerate(documents):
        chunks = chunk_document(doc)
        
        # Track article-based vs fallback
        has_articles = any(c.get("article", "") for c in chunks)
        if has_articles:
            docs_with_articles += 1
        elif chunks:
            docs_fallback += 1
        
        all_chunks.extend(chunks)
        
        if (i + 1) % cfg.BATCH_LOG_INTERVAL == 0 or (i + 1) == len(documents):
            logger.info(
                f"  Processed {i+1:,}/{len(documents):,} docs "
                f"→ {len(all_chunks):,} chunks"
            )
    
    if not all_chunks:
        logger.error("Không tạo được chunk nào!")
        sys.exit(1)
    
    # ── Step 3: Stats ──
    token_counts = [c["token_count"] for c in all_chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    
    # Token distribution
    buckets = {
        "< 50": sum(1 for t in token_counts if t < 50),
        "50-200": sum(1 for t in token_counts if 50 <= t < 200),
        "200-500": sum(1 for t in token_counts if 200 <= t < 500),
        "500-1024": sum(1 for t in token_counts if 500 <= t < 1024),
        "> 1024": sum(1 for t in token_counts if t >= 1024),
    }
    
    logger.info(f"\n  📊 Chunk Statistics:")
    logger.info(f"    Total chunks: {len(all_chunks):,}")
    logger.info(f"    Avg tokens/chunk: {avg_tokens:.0f}")
    logger.info(f"    Min tokens: {min(token_counts)}")
    logger.info(f"    Max tokens: {max(token_counts)}")
    logger.info(f"    Docs with articles: {docs_with_articles}")
    logger.info(f"    Docs fallback (sliding window): {docs_fallback}")
    logger.info(f"    Token distribution: {buckets}")
    
    # ── Step 4: Save ──
    logger.info("Step 3: Saving chunks.jsonl...")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(all_chunks):,} chunks → {output_path}")
    
    # Update manifest
    elapsed = time.time() - t_start
    manifest = {
        "phase": "02_chunk",
        "input": str(input_path),
        "total_documents": len(documents),
        "total_chunks": len(all_chunks),
        "avg_tokens_per_chunk": round(avg_tokens),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "docs_with_articles": docs_with_articles,
        "docs_fallback": docs_fallback,
        "token_distribution": buckets,
        "max_chunk_tokens": cfg.MAX_CHUNK_TOKENS,
        "min_chunk_tokens": cfg.MIN_CHUNK_TOKENS,
        "elapsed_seconds": round(elapsed, 1),
    }
    manifest_path = cfg.PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"✅ PHASE 2 COMPLETE — {len(all_chunks):,} chunks in {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
