"""
Phase 3a — Build BM25 Keyword Index (Whoosh)
===============================================
Đọc chunks.jsonl, tokenize tiếng Việt, build Whoosh BM25 index.

Usage:
    python scripts/03_index_bm25.py
    python scripts/03_index_bm25.py --input processed/chunks.jsonl

Output:
    indexes/bm25/   — Whoosh index directory 
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg

# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "03_index_bm25.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")


def create_vietnamese_analyzer():
    """
    Tạo analyzer cho tiếng Việt.
    Thử dùng underthesea word_tokenize, fallback về simple tokenizer.
    """
    try:
        from underthesea import word_tokenize as vn_tokenize
        logger.info("Dùng underthesea Vietnamese tokenizer")
        
        from whoosh.analysis import Analyzer, Token
        
        class VietnameseAnalyzer(Analyzer):
            """Custom Whoosh analyzer dùng underthesea."""
            def __call__(self, value, positions=False, chars=False,
                         keeporiginal=False, removestops=True,
                         start_pos=0, start_char=0, tokenize=True,
                         mode='', **kwargs):
                t = Token(positions, chars, removestops=removestops,
                          mode=mode)
                
                if not value:
                    return
                
                # Tokenize với underthesea
                try:
                    words = vn_tokenize(str(value)).split()
                except Exception:
                    words = str(value).lower().split()
                
                pos = start_pos
                for word in words:
                    word = word.lower().strip()
                    if not word:
                        continue
                    t.text = word
                    t.boost = 1.0
                    if positions:
                        t.pos = pos
                    pos += 1
                    yield t
        
        return VietnameseAnalyzer()
    
    except ImportError:
        logger.warning("underthesea not found, dùng simple tokenizer")
        from whoosh.analysis import SimpleAnalyzer
        return SimpleAnalyzer()


def build_index(chunks_path: Path, index_dir: Path):
    """Build Whoosh BM25 index từ chunks.jsonl."""
    import whoosh.index as windex
    from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
    
    # Tạo analyzer
    analyzer = create_vietnamese_analyzer()
    
    # Define schema
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        doc_id=ID(stored=True),
        title=TEXT(stored=True, analyzer=analyzer),
        doc_type=KEYWORD(stored=True, lowercase=True),
        issuer=TEXT(stored=True, analyzer=analyzer),
        article=STORED,
        clause=STORED,
        path=STORED,
        # Main search field — text_for_keyword đã bao gồm title + path + text
        content=TEXT(analyzer=analyzer),
    )
    
    # Tạo hoặc clear index directory
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Xóa index cũ nếu có
    import shutil
    if index_dir.exists() and any(index_dir.iterdir()):
        logger.info(f"Xóa index cũ tại {index_dir}")
        shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
    
    ix = windex.create_in(str(index_dir), schema)
    
    # Load chunks
    logger.info(f"Loading chunks từ {chunks_path}...")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    logger.info(f"  Loaded {len(chunks):,} chunks")
    
    # Index chunks
    logger.info("Building BM25 index...")
    writer = ix.writer(procs=1, limitmb=256)
    
    indexed = 0
    errors = 0
    
    for i, chunk in enumerate(chunks):
        try:
            writer.add_document(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                title=chunk.get("title", ""),
                doc_type=chunk.get("doc_type", ""),
                issuer=chunk.get("issuer", ""),
                article=chunk.get("article", ""),
                clause=chunk.get("clause", ""),
                path=chunk.get("path", ""),
                content=chunk.get("text_for_keyword", chunk.get("text", "")),
            )
            indexed += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Error indexing {chunk['chunk_id']}: {e}")
        
        if (i + 1) % 5000 == 0:
            logger.info(f"  Indexed {i+1:,}/{len(chunks):,}...")
    
    logger.info("Committing index (this may take a moment)...")
    writer.commit()
    
    logger.info(f"  Indexed: {indexed:,}, Errors: {errors}")
    
    return ix, indexed


def test_index(index_dir: Path, test_queries: list[str] = None):
    """Chạy vài query test để verify index."""
    import whoosh.index as windex
    from whoosh.qparser import MultifieldParser
    
    if test_queries is None:
        test_queries = [
            "quyền sử dụng đất",
            "thuế thu nhập cá nhân",
            "xử phạt vi phạm hành chính",
            "bảo hiểm xã hội",
            "hội đồng nhân dân",
        ]
    
    ix = windex.open_dir(str(index_dir))
    
    parser = MultifieldParser(["content", "title"], ix.schema)
    
    logger.info("\n  🔍 Test queries:")
    with ix.searcher() as searcher:
        for query_str in test_queries:
            try:
                query = parser.parse(query_str)
                results = searcher.search(query, limit=3)
                
                logger.info(f"\n  Query: '{query_str}' → {len(results)} results")
                for r in results[:3]:
                    logger.info(
                        f"    [{r.score:.2f}] {r['chunk_id']} — "
                        f"{r.get('title', '')[:60]}"
                    )
            except Exception as e:
                logger.warning(f"  Query '{query_str}' failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3a: Build BM25 Index")
    parser.add_argument("--input", type=str, default=None,
                        help="Path tới chunks.jsonl")
    parser.add_argument("--skip-test", action="store_true",
                        help="Bỏ qua test queries")
    args = parser.parse_args()
    
    cfg.ensure_dirs()
    t_start = time.time()
    
    chunks_path = Path(args.input) if args.input else cfg.PROCESSED_DIR / "chunks.jsonl"
    index_dir = cfg.BM25_INDEX_DIR

    logger.info("=" * 60)
    logger.info("PHASE 3a: BUILD BM25 INDEX (Whoosh)")
    logger.info(f"  Input: {chunks_path}")
    logger.info(f"  Index dir: {index_dir}")
    logger.info("=" * 60)
    
    if not chunks_path.exists():
        logger.error(f"Input file không tồn tại: {chunks_path}")
        logger.error("Hãy chạy Phase 2 trước: python scripts/02_chunk.py")
        sys.exit(1)
    
    # Build
    ix, indexed = build_index(chunks_path, index_dir)
    
    # Test  
    if not args.skip_test:
        test_index(index_dir)
    
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"✅ PHASE 3a COMPLETE — {indexed:,} chunks indexed in {elapsed:.1f}s")
    logger.info(f"  Index at: {index_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
