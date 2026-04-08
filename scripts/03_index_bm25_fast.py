"""
Phase 3a — Build BM25 Index (FAST — Pre-tokenized)
====================================================
Strategy: Pre-tokenize text bằng underthesea (multiprocessing)
→ Whoosh chỉ split trên spaces → nhanh gấp 50x.

Usage:
    python scripts/03_index_bm25_fast.py
    python scripts/03_index_bm25_fast.py --workers 8

Output:
    G:/ChatBot_indexes/bm25/
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg

# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

# Index output trên ổ G:
INDEX_DIR = Path("G:/ChatBot_indexes/bm25")


# ══════════════════════════════════════════════════════════
# STEP 1: Pre-tokenize bằng underthesea (multiprocessing)
# ══════════════════════════════════════════════════════════

def _init_tokenizer():
    """Initialize tokenizer trong worker process."""
    global _tokenize_fn
    try:
        from underthesea import word_tokenize
        word_tokenize("test")
        _tokenize_fn = word_tokenize
    except Exception:
        try:
            from pyvi import ViTokenizer
            ViTokenizer.tokenize("test")
            _tokenize_fn = ViTokenizer.tokenize
        except Exception:
            _tokenize_fn = lambda x: x


def _tokenize_batch(batch: list[dict]) -> list[dict]:
    """Tokenize 1 batch chunks trong worker process."""
    global _tokenize_fn
    results = []
    for chunk in batch:
        text = chunk.get("text_for_keyword", chunk.get("text", ""))
        try:
            tokenized = _tokenize_fn(str(text))
        except Exception:
            tokenized = text
        chunk["text_tokenized"] = tokenized
        results.append(chunk)
    return results


def pretokenize_chunks(chunks_path: Path, output_path: Path,
                       workers: int = 4, batch_size: int = 500) -> int:
    """Pre-tokenize chunks bằng multiprocessing (streaming, low RAM)."""
    logger.info(f"STEP 1: Pre-tokenize chunks ({workers} workers)")
    logger.info(f"  Input: {chunks_path}")
    logger.info(f"  Output: {output_path}")

    max_inflight = workers * 2  # Giới hạn batches đang xử lý
    written = 0
    submitted = 0
    batch_num = 0

    with open(chunks_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         ProcessPoolExecutor(max_workers=workers,
                              initializer=_init_tokenizer) as pool:

        futures = {}
        current_batch = []

        def _collect_done():
            """Thu kết quả từ futures đã hoàn tất."""
            nonlocal written
            done_keys = [k for k, v in futures.items() if v.done()]
            for key in done_keys:
                future = futures.pop(key)
                try:
                    results = future.result()
                    for chunk in results:
                        fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        written += 1
                except Exception as e:
                    logger.warning(f"  Batch error: {e}")

                if written % 100000 == 0 and written > 0:
                    logger.info(f"  Tokenized: {written:,}")

        for line in fin:
            try:
                chunk = json.loads(line.strip())
                current_batch.append(chunk)
            except json.JSONDecodeError:
                continue

            if len(current_batch) >= batch_size:
                # Chờ nếu quá nhiều batches đang chạy
                while len(futures) >= max_inflight:
                    _collect_done()
                    if len(futures) >= max_inflight:
                        import time as _t
                        _t.sleep(0.1)

                batch_num += 1
                futures[batch_num] = pool.submit(_tokenize_batch, current_batch)
                submitted += 1
                current_batch = []

        # Submit batch cuối
        if current_batch:
            batch_num += 1
            futures[batch_num] = pool.submit(_tokenize_batch, current_batch)
            submitted += 1

        logger.info(f"  Submitted {submitted} batches, collecting results...")

        # Collect tất cả remaining
        from concurrent.futures import wait
        wait(list(futures.values()))
        _collect_done()

    logger.info(f"  Done: {written:,} chunks tokenized")
    return written


def _prepare_content(chunk: dict) -> str:
    """
    Chuẩn bị content string cho Whoosh indexing.
    Nếu text_tokenized là list (underthesea output), join bằng space,
    nhưng mỗi token multi-word → nối bằng underscore để RegexTokenizer(\S+) giữ nguyên.
    Ví dụ: ["công chức", "viên chức"] → "công_chức viên_chức"
    """
    tokenized = chunk.get("text_tokenized")
    if tokenized and isinstance(tokenized, list):
        return " ".join(t.replace(" ", "_") for t in tokenized)
    # Fallback: dùng text_for_keyword hoặc text thô
    return chunk.get("text_for_keyword", chunk.get("text", ""))


def build_index(chunks_path: Path, index_dir: Path) -> int:
    """Build Whoosh BM25 index trên pre-tokenized text."""
    import shutil
    import whoosh.index as windex
    from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
    from whoosh.analysis import RegexTokenizer, LowercaseFilter

    logger.info(f"STEP 2: Build BM25 index")
    logger.info(f"  Index dir: {index_dir}")

    # RegexTokenizer(r'\S+') → split by whitespace only, giữ nguyên underscore
    # công_chức → 1 token "công_chức" (KHÔNG tách thành "công" + "chức")
    analyzer = RegexTokenizer(r'\S+') | LowercaseFilter()

    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        doc_id=ID(stored=True),
        title=TEXT(stored=True, analyzer=analyzer),
        doc_type=KEYWORD(stored=True, lowercase=True),
        issuer=TEXT(stored=True, analyzer=analyzer),
        article=STORED,
        clause=STORED,
        path=STORED,
        content=TEXT(analyzer=analyzer),
    )

    # Clear old index
    if index_dir.exists() and any(index_dir.iterdir()):
        logger.info(f"  Xóa index cũ...")
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    ix = windex.create_in(str(index_dir), schema)
    writer = ix.writer(procs=1, limitmb=1024)

    indexed = 0
    errors = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunk = json.loads(line.strip())
                writer.add_document(
                    chunk_id=chunk["chunk_id"],
                    doc_id=chunk["doc_id"],
                    title=chunk.get("title", ""),
                    doc_type=chunk.get("doc_type", ""),
                    issuer=chunk.get("issuer", ""),
                    article=chunk.get("article", ""),
                    clause=chunk.get("clause", ""),
                    path=chunk.get("path", ""),
                    # Dùng text đã pre-tokenized
                    content=_prepare_content(chunk),
                )
                indexed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"  Error: {e}")

            if indexed % 100000 == 0 and indexed > 0:
                logger.info(f"  Indexed {indexed:,}...")

    logger.info(f"  Committing ({indexed:,} docs)...")
    writer.commit()
    logger.info(f"  Committed! Indexed: {indexed:,}, Errors: {errors}")

    return indexed


def test_index(index_dir: Path):
    """Quick test search."""
    import whoosh.index as windex
    from whoosh.qparser import MultifieldParser

    test_queries = [
        "công_chức viên_chức",
        "bảo_hiểm_xã_hội",
        "thi_đua khen_thưởng",
        "chính_quyền địa_phương",
        "tiền_lương",
        "nghị_định",
    ]

    ix = windex.open_dir(str(index_dir))
    parser = MultifieldParser(["content", "title"], ix.schema)

    logger.info("\n  🔍 Test queries:")
    with ix.searcher() as searcher:
        for q in test_queries:
            try:
                query = parser.parse(q)
                results = searcher.search(query, limit=3)
                logger.info(f"\n  '{q}' → {len(results)} results")
                for r in results[:2]:
                    logger.info(f"    [{r.score:.1f}] {r.get('title', '')[:60]}")
            except Exception as e:
                logger.warning(f"  '{q}' failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="BM25 Index (Fast)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Số CPU workers cho pre-tokenize (default: 6)")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Bỏ qua pre-tokenize (dùng file đã tokenize)")
    parser.add_argument("--skip-test", action="store_true",
                        help="Bỏ qua test queries")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("BM25 INDEX BUILD (FAST MODE)")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Index dir: {INDEX_DIR}")
    logger.info("=" * 60)

    chunks_path = cfg.PROCESSED_DIR / "chunks.jsonl"
    tokenized_path = cfg.PROCESSED_DIR / "chunks_tokenized.jsonl"

    if not chunks_path.exists():
        logger.error(f"chunks.jsonl not found: {chunks_path}")
        sys.exit(1)

    # Step 1: Pre-tokenize
    if not args.skip_tokenize:
        t1 = time.time()
        pretokenize_chunks(chunks_path, tokenized_path,
                           workers=args.workers)
        logger.info(f"  Pre-tokenize time: {time.time()-t1:.0f}s")
    else:
        logger.info("Skipping pre-tokenize (using existing file)")

    # Step 2: Build index
    input_file = tokenized_path if tokenized_path.exists() else chunks_path
    t2 = time.time()
    indexed = build_index(input_file, INDEX_DIR)
    logger.info(f"  Index build time: {time.time()-t2:.0f}s")

    # Step 3: Test
    if not args.skip_test:
        test_index(INDEX_DIR)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"✅ DONE — {indexed:,} chunks indexed in {elapsed/60:.1f} min")
    logger.info(f"  Index at: {INDEX_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
