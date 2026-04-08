"""
Phase 01 — Lọc văn bản Nội vụ từ dataset 518K (v2 - Optimized)
================================================================
2-pass approach:
  Pass 1: Scan metadata → lọc IDs + lưu metadata JSONL (vectorized, ~10s)
  Pass 2: Stream content parquets → join chỉ docs matched → output JSONL

Usage:
    python scripts/00_filter_noiwu.py                  # Full filter
    python scripts/00_filter_noiwu.py --dry-run        # Chỉ đếm, không xuất
    python scripts/00_filter_noiwu.py --limit 5000     # Giới hạn

Output:
    processed/filtered_noiwu_docs.jsonl
    processed/filter_stats.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pyarrow.parquet as pq
import pandas as pd
from loguru import logger

from config.settings import cfg

# ── Logger ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

# ══════════════════════════════════════════════════════════
# BỘ LỌC NỘI VỤ (tất cả dùng simple string matching, không regex phức tạp)
# ══════════════════════════════════════════════════════════

# Sectors liên quan Nội vụ
SECTOR_KEYWORDS = [
    "bộ máy hành chính", "lao động", "tiền lương",
    "cán bộ", "công chức", "viên chức",
    "bảo hiểm", "việc làm",
    "thi đua", "khen thưởng",
    "tôn giáo", "tín ngưỡng",
    "lưu trữ", "văn thư",
    "thanh niên", "dân chủ",
    "người có công", "biên chế",
    "chính quyền địa phương", "đơn vị hành chính", "địa giới",
]

# Cơ quan ban hành
ISSUER_KEYWORDS = [
    "quốc hội", "chính phủ", "thủ tướng",
    "bộ nội vụ", "bộ lao động", "bộ tài chính",
    "bảo hiểm xã hội",
]

# Title keywords (fallback)
TITLE_KEYWORDS = [
    "công chức", "viên chức", "cán bộ", "biên chế",
    "tiền lương", "lương cơ sở", "phụ cấp",
    "thi đua", "khen thưởng", "huân chương",
    "tôn giáo", "tín ngưỡng",
    "lưu trữ", "văn thư",
    "thanh niên",
    "bảo hiểm xã hội", "hưu trí",
    "bộ nội vụ",
    "chính quyền địa phương",
    "đơn vị hành chính",
    "dân chủ cơ sở", "người có công",
    "tai nạn lao động", "bệnh nghề nghiệp",
    "cải cách hành chính", "vị trí việc làm",
    "tuyển dụng công chức", "kỷ luật cán bộ",
    "đào tạo bồi dưỡng",
]

# Loại VB ưu tiên
PRIORITY_DOC_TYPES = {
    "Hiến pháp", "Luật", "Bộ luật",
    "Nghị quyết", "Nghị định", "Quyết định",
    "Thông tư", "Thông tư liên tịch",
    "Chỉ thị", "Lệnh", "Pháp lệnh",
}


def build_pattern(keywords: list[str]) -> str:
    """Build regex pattern từ keyword list (escape special chars, join với |)."""
    import re
    escaped = [re.escape(k) for k in keywords]
    return "|".join(escaped)


SECTOR_PAT = build_pattern(SECTOR_KEYWORDS)
ISSUER_PAT = build_pattern(ISSUER_KEYWORDS)
TITLE_PAT = build_pattern(TITLE_KEYWORDS)


def safe_str(val) -> str:
    if val is None:
        return ""
    s = str(val)
    if s in ("None", "nan", "NaN"):
        return ""
    return s.strip()


# ══════════════════════════════════════════════════════════
# PASS 1: Scan metadata → collect matched IDs + metadata
# ══════════════════════════════════════════════════════════

def pass1_filter_metadata(metadata_path: Path, limit: int | None = None):
    """
    Vectorized metadata scan. Trả về DataFrame chỉ chứa matched rows.
    """
    logger.info(f"PASS 1: Scan metadata ({metadata_path.name})")
    pf = pq.ParquetFile(str(metadata_path))
    total_rows = pf.metadata.num_rows
    logger.info(f"  Total rows: {total_rows:,}")

    # Chỉ đọc columns cần cho filter (KHÔNG đọc content)
    filter_cols = ["id", "document_number", "title", "legal_type",
                   "legal_sectors", "issuing_authority", "issuance_date",
                   "signers", "url"]

    all_matched = []
    stats = {
        "total_scanned": 0,
        "filtered_by_doctype": 0,
        "matched_by_sector": 0,
        "matched_by_issuer": 0,
        "matched_by_title": 0,
    }

    batch_num = 0
    for batch in pf.iter_batches(batch_size=10000, columns=filter_cols):
        batch_num += 1
        df = batch.to_pandas()
        stats["total_scanned"] += len(df)

        # Vectorized string conversion
        s_sectors = df["legal_sectors"].astype(str).fillna("")
        s_issuer = df["issuing_authority"].astype(str).fillna("")
        s_title = df["title"].astype(str).fillna("")
        s_type = df["legal_type"].astype(str).fillna("")

        # Filter 1: doc_type
        type_ok = s_type.isin(PRIORITY_DOC_TYPES) | s_type.isin(["", "None", "nan"])
        n_filtered = int((~type_ok).sum())
        stats["filtered_by_doctype"] += n_filtered

        # Filter 2: sector OR issuer OR title (vectorized)
        m_sector = s_sectors.str.contains(SECTOR_PAT, case=False, na=False)
        m_issuer = s_issuer.str.contains(ISSUER_PAT, case=False, na=False)
        m_title = s_title.str.contains(TITLE_PAT, case=False, na=False)

        # Combine: type OK AND (sector OR issuer OR title)
        mask = type_ok & (m_sector | m_issuer | m_title)

        stats["matched_by_sector"] += int((type_ok & m_sector).sum())
        stats["matched_by_issuer"] += int((type_ok & m_issuer).sum())
        stats["matched_by_title"] += int((type_ok & m_title).sum())

        matched_df = df[mask]
        if len(matched_df) > 0:
            all_matched.append(matched_df)

        if batch_num % 10 == 0:
            n = sum(len(m) for m in all_matched)
            logger.info(f"  Batch {batch_num}: scanned {stats['total_scanned']:,} | matched {n:,}")

        # Check limit
        if limit:
            n = sum(len(m) for m in all_matched)
            if n >= limit:
                break

    # Concat all matched
    if all_matched:
        result = pd.concat(all_matched, ignore_index=True)
        if limit and len(result) > limit:
            result = result.head(limit)
    else:
        result = pd.DataFrame()

    stats["final_matched"] = len(result)
    logger.info(f"  PASS 1 DONE: {stats['final_matched']:,} matched in "
                f"{stats['total_scanned']:,} scanned")
    return result, stats


# ══════════════════════════════════════════════════════════
# PASS 2: Join matched IDs với content → output JSONL
# ══════════════════════════════════════════════════════════

def pass2_join_content(matched_df: pd.DataFrame, content_dir: Path,
                       output_path: Path) -> int:
    """Stream content parquets, join với matched IDs, xuất JSONL."""
    logger.info(f"PASS 2: Join content ({content_dir})")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ID set + metadata lookup
    matched_ids = set(matched_df["id"].astype(str))
    logger.info(f"  Matched IDs: {len(matched_ids):,}")

    # Pre-build metadata dict từ DataFrame (vectorized, nhanh)
    meta_lookup = {}
    for col in matched_df.columns:
        matched_df[col] = matched_df[col].astype(str).replace(
            {"None": "", "nan": "", "NaN": ""}, regex=False
        )

    for _, row in matched_df.iterrows():
        doc_id = row["id"]
        meta_lookup[doc_id] = {
            "doc_id": f"doc_{doc_id}",
            "title": row.get("title", ""),
            "document_number": row.get("document_number", ""),
            "doc_type": row.get("legal_type", ""),
            "issuer": row.get("issuing_authority", ""),
            "issue_date": row.get("issuance_date", ""),
            "url": row.get("url", ""),
        }

    logger.info(f"  Built metadata lookup: {len(meta_lookup):,} entries")

    # Find content files
    content_files = sorted(content_dir.glob("data-*.parquet"))
    if not content_files:
        content_files = sorted(content_dir.glob("train-*.parquet"))
    logger.info(f"  Content files: {len(content_files)}")

    written = 0
    ids_done = set()

    with open(output_path, "w", encoding="utf-8") as fout:
        for fpath in content_files:
            logger.info(f"  Reading: {fpath.name}")
            pf = pq.ParquetFile(str(fpath))

            for batch in pf.iter_batches(batch_size=5000, columns=["id", "content"]):
                df = batch.to_pandas()
                df["id"] = df["id"].astype(str)

                # Vectorized filter: chỉ giữ rows có id trong matched_ids
                mask = df["id"].isin(matched_ids) & ~df["id"].isin(ids_done)
                hits = df[mask]

                for _, row in hits.iterrows():
                    doc_id = row["id"]
                    content = str(row.get("content", ""))
                    if len(content) < 50:
                        continue

                    meta = meta_lookup.get(doc_id, {"doc_id": f"doc_{doc_id}"})
                    doc = {**meta, "content": content}
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    ids_done.add(doc_id)
                    written += 1

                if written > 0 and written % 5000 == 0:
                    logger.info(f"  Written: {written:,}/{len(matched_ids):,}")

            # Early exit nếu đã tìm hết
            if len(ids_done) >= len(matched_ids):
                logger.info(f"  Đã tìm đủ {len(ids_done):,} docs, dừng sớm")
                break

    remaining = matched_ids - ids_done
    if remaining:
        logger.info(f"  {len(remaining):,} docs không có content riêng")

    return written


def pass2_join_from_metadata(matched_df: pd.DataFrame, metadata_path: Path,
                              output_path: Path) -> int:
    """Fallback: lấy content từ metadata parquet."""
    logger.info(f"PASS 2 (fallback): Join content from metadata")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_ids = set(matched_df["id"].astype(str))

    # Build meta lookup
    meta_lookup = {}
    for _, row in matched_df.iterrows():
        doc_id = str(row["id"])
        meta_lookup[doc_id] = {
            "doc_id": f"doc_{doc_id}",
            "title": safe_str(row.get("title", "")),
            "document_number": safe_str(row.get("document_number", "")),
            "doc_type": safe_str(row.get("legal_type", "")),
            "issuer": safe_str(row.get("issuing_authority", "")),
            "issue_date": safe_str(row.get("issuance_date", "")),
            "url": safe_str(row.get("url", "")),
        }

    pf = pq.ParquetFile(str(metadata_path))
    written = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=5000, columns=["id", "content"]):
            df = batch.to_pandas()
            df["id"] = df["id"].astype(str)
            mask = df["id"].isin(matched_ids)
            hits = df[mask]

            for _, row in hits.iterrows():
                doc_id = row["id"]
                content = str(row.get("content", ""))
                if len(content) < 50:
                    continue
                meta = meta_lookup.get(doc_id, {"doc_id": f"doc_{doc_id}"})
                doc = {**meta, "content": content}
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

            if written > 0 and written % 5000 == 0:
                logger.info(f"  Written: {written:,}")

    return written


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 01: Lọc VB Nội vụ")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ đếm, không xuất file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số docs output")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 01: LỌC VĂN BẢN NỘI VỤ (v2)")
    logger.info(f"  Data source: {cfg.LEGAL_DOCS_DIR}")
    logger.info(f"  Mode: {'DRY RUN' if args.dry_run else 'FULL'}")
    logger.info(f"  Limit: {args.limit or 'ALL'}")
    logger.info("=" * 60)

    # ── Find metadata file ──
    metadata_dir = cfg.LEGAL_DOCS_DIR / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.parquet"))
    if not metadata_files:
        logger.error(f"Không tìm thấy metadata parquet tại {metadata_dir}")
        sys.exit(1)

    # ── PASS 1: Filter metadata ──
    matched_df, stats = pass1_filter_metadata(metadata_files[0], args.limit)

    logger.info("=" * 40)
    logger.info("KẾT QUẢ LỌC:")
    logger.info(f"  Total scanned: {stats['total_scanned']:,}")
    logger.info(f"  Filtered by doc_type: {stats['filtered_by_doctype']:,}")
    logger.info(f"  Matched by sector: {stats['matched_by_sector']:,}")
    logger.info(f"  Matched by issuer: {stats['matched_by_issuer']:,}")
    logger.info(f"  Matched by title: {stats['matched_by_title']:,}")
    logger.info(f"  FINAL MATCHED: {stats['final_matched']:,}")
    logger.info("=" * 40)

    if args.dry_run:
        elapsed = time.time() - t_start
        logger.info(f"DRY RUN hoàn tất trong {elapsed:.1f}s")
        return

    if len(matched_df) == 0:
        logger.error("Không có docs nào matched!")
        sys.exit(1)

    # ── PASS 2: Join content ──
    output_path = cfg.PROCESSED_DIR / "filtered_noiwu_docs.jsonl"
    content_dir = cfg.LEGAL_DOCS_DIR / "content"

    if content_dir.exists() and any(content_dir.glob("*.parquet")):
        written = pass2_join_content(matched_df, content_dir, output_path)
    else:
        logger.warning(f"Content dir không tồn tại hoặc rỗng: {content_dir}")
        written = pass2_join_from_metadata(
            matched_df, metadata_files[0], output_path
        )

    # ── Save stats ──
    elapsed = time.time() - t_start
    stats["written_docs"] = written
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["output_file"] = str(output_path)

    stats_path = cfg.PROCESSED_DIR / "filter_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 60)
    logger.info(f"PHASE 01 COMPLETE")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Docs written: {written:,}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Stats: {stats_path}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
