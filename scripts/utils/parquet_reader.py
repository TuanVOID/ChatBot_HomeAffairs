"""
Đọc parquet files từ dataset vietnamese-legal-documents.
Hỗ trợ cả metadata (1 file) và content (11 files).
"""

import glob
from pathlib import Path
from typing import Generator

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


def iter_parquet_files(directory: Path, pattern: str = "data-*.parquet"):
    """Liệt kê tất cả parquet files trong thư mục, sắp xếp theo tên."""
    files = sorted(glob.glob(str(directory / pattern)))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet files tại {directory}")
    logger.info(f"Tìm thấy {len(files)} parquet files tại {directory}")
    return files


def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    """
    Load toàn bộ metadata (1 parquet file, ~82MB, ~518K rows).
    Columns: id, document_number, title, url, legal_type, legal_sectors,
             issuing_authority, issuance_date, signers, content (markdown)
    
    Lưu ý: metadata config có cả cột 'content' nhưng nó chứa
    markdown-formatted text. Ta sẽ ưu tiên dùng content từ content config.
    """
    files = iter_parquet_files(metadata_dir)
    dfs = []
    for f in files:
        logger.info(f"Đọc metadata: {Path(f).name}")
        df = pd.read_parquet(f)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Tổng metadata: {len(result):,} rows, columns: {list(result.columns)}")
    return result


def iter_content_batches(
    content_dir: Path, batch_size: int = 10_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream load content parquet files theo batch.
    11 files × ~350MB = ~3.6GB tổng. Không load hết vào RAM.
    
    Content config columns: id, content
    
    Yields:
        pd.DataFrame  với columns [id, content] — mỗi batch ~batch_size rows
    """
    files = iter_parquet_files(content_dir)
    total_yielded = 0

    for fpath in files:
        logger.info(f"Đọc content file: {Path(fpath).name}")
        parquet_file = pq.ParquetFile(fpath)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            total_yielded += len(df)
            yield df

    logger.info(f"Tổng content rows đã đọc: {total_yielded:,}")


def load_content_full(content_dir: Path) -> pd.DataFrame:
    """
    Load TOÀN BỘ content vào RAM. Cần ~4-6GB RAM.
    Chỉ dùng khi máy có đủ 32GB RAM.
    """
    files = iter_parquet_files(content_dir)
    dfs = []
    for f in files:
        logger.info(f"Loading full content: {Path(f).name}")
        dfs.append(pd.read_parquet(f))
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Tổng content: {len(result):,} rows")
    return result
