"""
Phase 3a — Build BM25 Keyword Index (Whoosh)
=============================================
Hỗ trợ 2 mode:
- full rebuild (mặc định): xóa index cũ và build lại toàn bộ
- incremental (--incremental): chỉ add chunk mới chưa có trong index

Usage:
    python scripts/03_index_bm25.py
    python scripts/03_index_bm25.py --incremental
    python scripts/03_index_bm25.py --input processed/chunks.jsonl
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg
from src.retrieval.text_tokenizer import tokenize_for_index

# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "03_index_bm25.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")

from whoosh.analysis import RegexTokenizer, LowercaseFilter


def _tokenize_for_index(text: str) -> str:
    """
    Pre-tokenize tiếng Việt trước khi index.
    Output dạng "token_1 token_2 ..." để analyzer whitespace xử lý ổn định.
    """
    return tokenize_for_index(text)


def _load_existing_chunk_ids(ix) -> set[str]:
    """Đọc toàn bộ chunk_id hiện có trong index (dùng cho incremental mode)."""
    ids = set()
    with ix.searcher() as s:
        for fields in s.all_stored_fields():
            cid = fields.get("chunk_id")
            if cid:
                ids.add(cid)
    return ids


def build_index(chunks_path: Path, index_dir: Path, incremental: bool = False):
    """Build/Update Whoosh BM25 index từ chunks.jsonl (streaming mode)."""
    import whoosh.index as windex
    from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
    
    logger.info("Using internal Unicode tokenizer for BM25 indexing")

    # Dùng analyzer built-in để schema pickle-safe.
    analyzer = RegexTokenizer(r"\S+") | LowercaseFilter()
    
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
    
    index_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    existing_ids: set[str] = set()
    skipped_existing = 0
    mode = "full_rebuild"

    if incremental and index_dir.exists() and any(index_dir.iterdir()):
        logger.info(f"Incremental mode: mở index hiện có tại {index_dir}")
        ix = windex.open_dir(str(index_dir))
        existing_ids = _load_existing_chunk_ids(ix)
        logger.info(f"  Existing chunk_ids: {len(existing_ids):,}")
        mode = "incremental"
    else:
        if index_dir.exists() and any(index_dir.iterdir()):
            logger.info(f"Xóa index cũ tại {index_dir}")
            shutil.rmtree(index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
        ix = windex.create_in(str(index_dir), schema)
    
    # Streaming: đọc từng dòng, index ngay
    logger.info(f"Streaming chunks từ {chunks_path}...")
    writer = ix.writer(procs=1, limitmb=1024)
    
    indexed = 0
    errors = 0
    total_lines = 0
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                chunk = json.loads(line.strip())
                cid = chunk["chunk_id"]

                if cid in existing_ids:
                    skipped_existing += 1
                    continue

                writer.add_document(
                    chunk_id=cid,
                    doc_id=chunk["doc_id"],
                    title=_tokenize_for_index(chunk.get("title", "")),
                    doc_type=chunk.get("doc_type", ""),
                    issuer=_tokenize_for_index(chunk.get("issuer", "")),
                    article=chunk.get("article", ""),
                    clause=chunk.get("clause", ""),
                    path=chunk.get("path", ""),
                    content=_tokenize_for_index(
                        chunk.get("text_for_keyword", chunk.get("text", ""))
                    ),
                )
                indexed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"  Error indexing: {e}")
            
            if indexed % 50000 == 0 and indexed > 0:
                logger.info(f"  Indexed {indexed:,}...")
    
    logger.info(f"Committing index ({indexed:,} docs, this may take a moment)...")
    writer.commit()
    
    logger.info(
        f"  Mode={mode} | Added={indexed:,} | Skipped(existing)={skipped_existing:,} "
        f"| Total lines={total_lines:,} | Errors={errors}"
    )
    
    return ix, indexed, skipped_existing, total_lines


def write_bm25_config(
    index_dir: Path,
    *,
    input_path: Path,
    indexed_added: int,
    skipped_existing: int,
    total_lines: int,
    incremental: bool,
    elapsed_seconds: float,
):
    config_path = index_dir / "config.json"
    previous = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = {}

    created_at = previous.get("created_at") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    try:
        import whoosh.index as windex

        ix = windex.open_dir(str(index_dir))
        total_docs = ix.doc_count()
    except Exception:
        total_docs = None

    cfg_data = {
        "index_type": "whoosh_bm25",
        "build_mode": "incremental" if incremental else "full_rebuild",
        "input_chunks_path": str(input_path),
        "indexed_added": indexed_added,
        "skipped_existing": skipped_existing,
        "total_lines_scanned": total_lines,
        "total_docs_in_index": total_docs,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_version": f"bm25-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved BM25 config -> {config_path}")


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
    parser.add_argument("--incremental", action="store_true",
                        help="Chỉ add chunk mới chưa có trong index")
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
    logger.info(f"  Mode: {'INCREMENTAL' if args.incremental else 'FULL_REBUILD'}")
    logger.info("=" * 60)
    
    if not chunks_path.exists():
        logger.error(f"Input file không tồn tại: {chunks_path}")
        logger.error("Hãy chạy Phase 2 trước: python scripts/02_chunk.py")
        sys.exit(1)
    
    # Build
    ix, indexed, skipped_existing, total_lines = build_index(
        chunks_path, index_dir, incremental=args.incremental
    )
    
    # Test  
    if not args.skip_test:
        test_index(index_dir)
    
    elapsed = time.time() - t_start
    write_bm25_config(
        index_dir,
        input_path=chunks_path,
        indexed_added=indexed,
        skipped_existing=skipped_existing,
        total_lines=total_lines,
        incremental=args.incremental,
        elapsed_seconds=elapsed,
    )
    logger.info("=" * 60)
    logger.info(
        f"✅ PHASE 3a COMPLETE — added {indexed:,} chunks "
        f"(skipped {skipped_existing:,}/{total_lines:,}) in {elapsed:.1f}s"
    )
    logger.info(f"  Index at: {index_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
