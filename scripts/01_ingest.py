"""
Phase 1 — Ingestion & Normalize
=================================
Đọc parquet files từ vietnamese-legal-documents dataset,
normalize text, extract metadata, xuất documents.jsonl.

Usage:
    python scripts/01_ingest.py                    # Full run (518K docs)
    python scripts/01_ingest.py --limit 1000       # Test với 1000 docs
    python scripts/01_ingest.py --limit 1000 --use-metadata-content  # Dùng content từ metadata

Output:
    processed/documents.jsonl   — mỗi dòng = 1 document JSON
    processed/manifest.json     — thông tin processing run 
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Thêm project root vào sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from loguru import logger

from config.settings import cfg
from scripts.utils.text_utils import clean_legal_text
from scripts.utils.parquet_reader import load_metadata, iter_content_batches


# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "01_ingest.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")


def parse_legal_sectors(sectors_raw) -> list[str]:
    """Parse cột legal_sectors (có thể là string, list, hoặc None)."""
    if sectors_raw is None or (isinstance(sectors_raw, float) and pd.isna(sectors_raw)):
        return []
    if isinstance(sectors_raw, list):
        return [str(s).strip() for s in sectors_raw if s]
    if isinstance(sectors_raw, str):
        # Có thể là "Đất đai, Bất động sản" hoặc "['Đất đai', 'Bất động sản']"
        s = sectors_raw.strip()
        if s.startswith("["):
            try:
                return json.loads(s.replace("'", '"'))
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def parse_signers(signers_raw) -> list[str]:
    """Parse cột signers tương tự legal_sectors."""
    return parse_legal_sectors(signers_raw)


def safe_str(val) -> str:
    """Convert giá trị sang string, xử lý NaN/None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def build_document(row: pd.Series, content_text: str | None = None) -> dict | None:
    """
    Tạo 1 document dict từ metadata row + content text.
    Returns None nếu document không hợp lệ (quá ngắn, thiếu content).
    """
    # Ưu tiên content_text (từ content config), fallback về metadata content
    raw_content = content_text if content_text else safe_str(row.get("content", ""))

    # Normalize
    cleaned = clean_legal_text(raw_content)

    # Filter: bỏ docs quá ngắn
    if len(cleaned) < cfg.MIN_CONTENT_LENGTH:
        return None

    doc_id = safe_str(row.get("id", ""))
    if not doc_id:
        return None

    return {
        "doc_id": f"doc_{doc_id}",
        "title": clean_legal_text(safe_str(row.get("title", ""))),
        "document_number": safe_str(row.get("document_number", "")),
        "doc_type": safe_str(row.get("legal_type", "")),
        "issuer": safe_str(row.get("issuing_authority", "")),
        "issue_date": safe_str(row.get("issuance_date", "")),
        "legal_sectors": parse_legal_sectors(row.get("legal_sectors")),
        "signers": parse_signers(row.get("signers")),
        "url": safe_str(row.get("url", "")),
        "content": cleaned,
        "content_length": len(cleaned),
    }


def run_with_metadata_content(metadata_df: pd.DataFrame, limit: int | None) -> list[dict]:
    """
    Strategy 1: Dùng content từ metadata config (nhanh, 1 file, ~82MB).
    Metadata config chứa sẵn cả cột 'content' (markdown).
    """
    logger.info("Strategy: Dùng content từ metadata config")
    documents = []
    total = len(metadata_df) if limit is None else min(limit, len(metadata_df))
    skipped = 0

    for i, (_, row) in enumerate(metadata_df.head(total).iterrows()):
        doc = build_document(row, content_text=None)  # dùng row["content"]
        if doc:
            documents.append(doc)
        else:
            skipped += 1

        if (i + 1) % cfg.BATCH_LOG_INTERVAL == 0:
            logger.info(f"  Processed {i+1:,}/{total:,} — kept {len(documents):,}, skipped {skipped:,}")

    logger.info(f"Hoàn tất: {len(documents):,} documents, skipped {skipped:,}")
    return documents


def run_with_content_files(metadata_df: pd.DataFrame, limit: int | None) -> list[dict]:
    """
    Strategy 2: Join metadata với content config riêng (11 parquet files).
    Content config có text chất lượng cao hơn (full content, không bị truncate).
    """
    logger.info("Strategy: Join metadata + content files (streaming)")
    content_dir = cfg.LEGAL_DOCS_DIR / "content"

    # Index metadata theo id để lookup nhanh
    meta_lookup = metadata_df.set_index("id").to_dict("index")
    logger.info(f"Metadata lookup: {len(meta_lookup):,} entries")

    documents = []
    skipped = 0
    processed = 0

    for batch_df in iter_content_batches(content_dir, batch_size=5_000):
        for _, row in batch_df.iterrows():
            doc_id = safe_str(row.get("id", ""))
            content_text = safe_str(row.get("content", ""))

            # Lookup metadata
            meta = meta_lookup.get(doc_id if not doc_id.startswith("doc_") else doc_id, {})
            if not meta and doc_id:
                # Thử với numeric id
                try:
                    meta = meta_lookup.get(int(doc_id), {})
                except (ValueError, TypeError):
                    pass

            # Build combined row
            combined = pd.Series({**meta, "id": doc_id})
            doc = build_document(combined, content_text=content_text)

            if doc:
                documents.append(doc)
            else:
                skipped += 1

            processed += 1
            if processed % cfg.BATCH_LOG_INTERVAL == 0:
                logger.info(f"  Processed {processed:,} — kept {len(documents):,}, skipped {skipped:,}")

            if limit and processed >= limit:
                logger.info(f"Đạt limit {limit:,}, dừng.")
                return documents

    logger.info(f"Hoàn tất: {len(documents):,} documents, skipped {skipped:,}")
    return documents


def save_documents(documents: list[dict], output_dir: Path) -> Path:
    """Lưu documents ra JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "documents.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(documents):,} documents → {output_path}")
    return output_path


def save_manifest(output_dir: Path, stats: dict):
    """Lưu manifest.json với thông tin processing run."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved manifest → {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Ingestion & Normalize")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số documents xử lý (mặc định: tất cả)")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 1: INGESTION & NORMALIZE")
    logger.info(f"  Data source: {cfg.LEGAL_DOCS_DIR}")
    logger.info(f"  Output: {cfg.PROCESSED_DIR}")
    logger.info(f"  Limit: {args.limit or 'ALL'}")
    logger.info("=" * 60)

    # ── Step 1: Load metadata ──
    logger.info("Step 1: Loading metadata...")
    metadata_dir = cfg.LEGAL_DOCS_DIR / "metadata"
    metadata_df = load_metadata(metadata_dir)

    # Log column info
    logger.info(f"  Columns: {list(metadata_df.columns)}")
    logger.info(f"  Total rows: {len(metadata_df):,}")

    # ── Step 2: Process documents (join metadata + content files) ──
    documents = run_with_content_files(metadata_df, args.limit)

    if not documents:
        logger.error("Không có document nào được tạo! Kiểm tra lại data path.")
        sys.exit(1)

    # ── Step 3: Stats ──
    content_lengths = [d["content_length"] for d in documents]
    avg_len = sum(content_lengths) / len(content_lengths) if content_lengths else 0

    # Loại bỏ content_length field trước khi save (nó chỉ dùng cho stats)
    for d in documents:
        del d["content_length"]

    # ── Step 4: Save ──
    logger.info("Step 3: Saving documents.jsonl...")
    save_documents(documents, cfg.PROCESSED_DIR)

    elapsed = time.time() - t_start
    stats = {
        "phase": "01_ingest",
        "source": str(cfg.LEGAL_DOCS_DIR),
        "strategy": "metadata_content" if args.use_metadata_content else "content_files",
        "total_documents": len(documents),
        "limit_applied": args.limit,
        "avg_content_length": round(avg_len),
        "min_content_length": min(content_lengths) if content_lengths else 0,
        "max_content_length": max(content_lengths) if content_lengths else 0,
        "elapsed_seconds": round(elapsed, 1),
        "doc_types": _count_field(documents, "doc_type"),
        "top_issuers": _count_field(documents, "issuer", top_n=10),
    }
    save_manifest(cfg.PROCESSED_DIR, stats)

    logger.info("=" * 60)
    logger.info(f"✅ PHASE 1 COMPLETE — {len(documents):,} documents in {elapsed:.1f}s")
    logger.info(f"  Avg content length: {avg_len:,.0f} chars")
    logger.info("=" * 60)


def _count_field(docs: list[dict], field: str, top_n: int | None = None) -> dict:
    """Đếm phân bố giá trị của 1 field."""
    from collections import Counter
    counter = Counter(d.get(field, "unknown") for d in docs)
    items = counter.most_common(top_n) if top_n else counter.most_common()
    return dict(items)


if __name__ == "__main__":
    main()
