"""
Phase 3b - Build/Update Vector Index (FAISS via Ollama Embeddings)
===================================================================
Supports:
- Full rebuild (default): rebuild FAISS + metadata from chunks.jsonl
- Incremental update (--incremental): only embed/add chunks not in metadata.jsonl

Also supports parallel embedding requests to Ollama via --workers.
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg

# Logger setup
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
log_file = cfg.LOG_DIR / "04_index_vector.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")

_HTTP_CLIENT = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        import httpx

        _HTTP_CLIENT = httpx.Client(
            timeout=120,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _HTTP_CLIENT


def check_ollama():
    """Check if Ollama server is running."""
    import httpx

    try:
        r = httpx.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama OK - {len(models)} models available")
            return True, models
        return False, []
    except Exception as e:
        logger.error(f"Ollama not responding: {e}")
        return False, []


def embed_batch(texts: list[str], model: str) -> np.ndarray:
    """
    Call Ollama embedding API for a batch of texts.
    Returns ndarray (batch_size, dim)
    """
    client = _get_http_client()
    url = f"{cfg.OLLAMA_BASE_URL}/api/embed"
    payload = {"model": model, "input": texts}

    r = client.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings", [])
    if not embs:
        raise ValueError(f"Ollama returned empty embeddings: {data}")
    return np.array(embs, dtype=np.float32)


def _prepare_embedding_text(chunk: dict) -> str:
    text = chunk.get("text", "")
    if not text.strip():
        text = chunk.get("title", "empty")
    if len(text) > 4000:
        text = text[:4000]
    return text


def _iter_chunks(path: Path, limit: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_existing_chunk_ids(meta_path: Path) -> set[str]:
    ids = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["chunk_id"])
            except Exception:
                continue
    return ids


def _count_pending_chunks(
    chunks_path: Path,
    existing_ids: set[str],
    limit: int | None,
) -> tuple[int, int]:
    total_seen = 0
    pending = 0
    for chunk in _iter_chunks(chunks_path, limit):
        total_seen += 1
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        if cid in existing_ids:
            continue
        pending += 1
    return total_seen, pending


def _embed_with_fallback(texts: list[str], model_name: str, dim: int) -> tuple[np.ndarray, int]:
    """
    Embed a batch; if fails, fallback to single-item embeds.
    Returns (embeddings, errors_added)
    """
    errors = 0
    try:
        embs = embed_batch(texts, model_name)
        if embs.shape[1] != dim:
            raise ValueError(f"Embedding dim mismatch: expected {dim}, got {embs.shape[1]}")
        return embs, errors
    except Exception as e:
        logger.warning(f"  Batch failed, fallback single embedding: {e}")
        rows = []
        for t in texts:
            try:
                one = embed_batch([t], model_name)
                if one.shape[1] != dim:
                    raise ValueError(
                        f"Single embedding dim mismatch: expected {dim}, got {one.shape[1]}"
                    )
                rows.append(one[0])
            except Exception as e2:
                errors += 1
                logger.warning(f"  Single embed failed, use zero vector: {e2}")
                rows.append(np.zeros((dim,), dtype=np.float32))
        return np.vstack(rows).astype(np.float32), errors


def _fit_embedding_rows(embs: np.ndarray, expected_rows: int, dim: int) -> np.ndarray:
    """Pad/trim rows if API returns unexpected batch size."""
    rows = embs.shape[0]
    if rows == expected_rows:
        return embs
    if rows > expected_rows:
        return embs[:expected_rows]
    pad = np.zeros((expected_rows - rows, dim), dtype=np.float32)
    return np.vstack([embs, pad]).astype(np.float32)


def _write_config(
    config_path: Path,
    *,
    model_name: str,
    dim: int,
    n_vectors: int,
    build_mode: str,
    source_path: Path,
    batch_size: int,
    workers: int,
    added_vectors: int,
    skipped_existing: int,
    previous_n_vectors: int,
    previous_config: dict | None,
):
    created_at = _now_utc_iso()
    if previous_config and previous_config.get("created_at"):
        created_at = previous_config["created_at"]

    config = {
        "model": model_name,
        "dim": dim,
        "n_vectors": int(n_vectors),
        "index_type": "Flat (IP + L2 normalize)",
        "metric": "cosine",
        "build_mode": build_mode,
        "source_chunks_path": str(source_path),
        "batch_size": batch_size,
        "workers": workers,
        "added_vectors": int(added_vectors),
        "skipped_existing": int(skipped_existing),
        "previous_n_vectors": int(previous_n_vectors),
        "created_at": created_at,
        "updated_at": _now_utc_iso(),
        "index_version": f"vec-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved config -> {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3b: Build/Update Vector Index")
    parser.add_argument("--input", type=str, default=None, help="Path to chunks.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding API batch size")
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel embedding workers (batch-level)",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help="Max in-flight embedding batches (default: workers*3)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only append new chunks not present in vector metadata",
    )
    # Keep for backward compatibility with old command lines.
    parser.add_argument(
        "--use-ivf",
        action="store_true",
        help="(ignored) Streaming mode currently uses Flat index only",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip post-build test query")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t_start = time.time()

    chunks_path = Path(args.input) if args.input else cfg.PROCESSED_DIR / "chunks.jsonl"
    index_dir = cfg.VECTOR_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.jsonl"
    config_path = index_dir / "config.json"
    staging_meta_path = index_dir / "metadata.build.tmp.jsonl"

    # Clean stale staging file from old interrupted runs.
    if staging_meta_path.exists():
        try:
            staging_meta_path.unlink()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("PHASE 3b: BUILD/UPDATE VECTOR INDEX (FAISS)")
    logger.info(f"  Input: {chunks_path}")
    logger.info(f"  Index dir: {index_dir}")
    logger.info(f"  Embedding model: {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Mode: {'INCREMENTAL' if args.incremental else 'FULL_REBUILD'}")
    logger.info(f"  Limit: {args.limit or 'ALL'}")
    logger.info("=" * 60)

    if args.use_ivf:
        logger.warning("`--use-ivf` is currently ignored in streaming mode.")

    if args.workers < 1:
        logger.error("--workers must be >= 1")
        sys.exit(1)

    if not chunks_path.exists():
        logger.error(f"Input file not found: {chunks_path}")
        logger.error("Run chunking first: python scripts/02_chunk.py")
        sys.exit(1)

    logger.info("Checking Ollama...")
    ok, models = check_ollama()
    if not ok:
        logger.error("Ollama server is not ready. Run: ollama serve")
        sys.exit(1)

    model_name = cfg.EMBEDDING_MODEL
    if not any(model_name in m for m in models):
        logger.warning(f"Model '{model_name}' not present. Trying automatic pull...")
        import httpx

        try:
            r = httpx.post(
                f"{cfg.OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=600,
            )
            logger.info(f"  Pull result: {r.status_code}")
        except Exception as e:
            logger.error(f"Cannot pull model: {e}")
            sys.exit(1)

    import faiss

    # Probe embedding dim
    probe = embed_batch(["probe"], model_name)
    dim = int(probe.shape[1])
    logger.info(f"Embedding dim = {dim}")

    previous_config = None
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                previous_config = json.load(f)
        except Exception:
            previous_config = None

    existing_ids: set[str] = set()
    previous_n_vectors = 0
    build_mode = "full_rebuild"

    if args.incremental and index_path.exists() and meta_path.exists():
        logger.info("Loading existing FAISS + metadata for incremental update...")
        index = faiss.read_index(str(index_path))
        previous_n_vectors = int(index.ntotal)
        if index.d != dim:
            logger.error(
                f"Existing index dim mismatch: index.d={index.d}, embedding dim={dim}. "
                "Need full rebuild with current model."
            )
            sys.exit(1)
        existing_ids = _load_existing_chunk_ids(meta_path)
        logger.info(f"  Existing vectors: {previous_n_vectors:,}")
        logger.info(f"  Existing metadata IDs: {len(existing_ids):,}")
        build_mode = "incremental"
        # In incremental mode, write additions to staging first.
        meta_open_mode = "w"
    else:
        if args.incremental:
            logger.warning(
                "Incremental requested but existing index/metadata not found. "
                "Fallback to full rebuild."
            )
        index = faiss.IndexFlatIP(dim)
        # In full rebuild mode, write full metadata to staging first.
        meta_open_mode = "w"

    total_seen, pending = _count_pending_chunks(chunks_path, existing_ids, args.limit)
    logger.info(f"  Total chunks scanned: {total_seen:,}")
    logger.info(f"  Pending chunks to embed: {pending:,}")

    if pending == 0:
        logger.info("No new chunks to process. Vector index unchanged.")
        _write_config(
            config_path,
            model_name=model_name,
            dim=dim,
            n_vectors=int(index.ntotal),
            build_mode=build_mode,
            source_path=chunks_path,
            batch_size=args.batch_size,
            workers=args.workers,
            added_vectors=0,
            skipped_existing=total_seen,
            previous_n_vectors=previous_n_vectors,
            previous_config=previous_config,
        )
        return

    max_pending = args.max_pending or max(1, args.workers * 3)
    if max_pending < args.workers:
        max_pending = args.workers

    logger.info(
        f"Embedding {pending:,} chunks (batch_size={args.batch_size}, "
        f"workers={args.workers}, max_pending={max_pending})..."
    )

    processed = 0
    skipped_existing = 0
    errors = 0
    next_log_at = 16_000

    batch_chunks: list[dict] = []
    batch_texts: list[str] = []

    def finalize_batch(chunks_batch: list[dict], embs: np.ndarray):
        nonlocal processed, next_log_at

        expected = len(chunks_batch)
        embs = _fit_embedding_rows(embs, expected, dim)
        faiss.normalize_L2(embs)
        index.add(embs)

        for c in chunks_batch:
            meta = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c.get("title", ""),
                "path": c.get("path", ""),
                "article": c.get("article", ""),
                "clause": c.get("clause", ""),
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            existing_ids.add(c["chunk_id"])

        processed += expected
        if processed >= next_log_at or processed == pending:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (pending - processed) / rate if rate > 0 else 0
            logger.info(
                f"  Embedded {processed:,}/{pending:,} ({rate:.0f} chunks/s, ETA {eta:.0f}s)"
            )
            while next_log_at <= processed:
                next_log_at += 16_000

    with open(staging_meta_path, meta_open_mode, encoding="utf-8") as meta_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_chunks: dict = {}

            def submit_current_batch():
                nonlocal batch_chunks, batch_texts
                if not batch_texts:
                    return
                chunks_copy = batch_chunks
                texts_copy = batch_texts
                future = executor.submit(_embed_with_fallback, texts_copy, model_name, dim)
                future_to_chunks[future] = chunks_copy
                batch_chunks = []
                batch_texts = []

            def drain_some(block_until_one: bool):
                nonlocal errors
                if not future_to_chunks:
                    return
                if block_until_one:
                    done, _ = wait(
                        list(future_to_chunks.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    done = list(future_to_chunks.keys())

                for fut in done:
                    chunks_batch = future_to_chunks.pop(fut)
                    try:
                        embs, err_inc = fut.result()
                        errors += err_inc
                    except Exception as e:
                        logger.warning(f"  Worker future failed, fill zeros: {e}")
                        errors += len(chunks_batch)
                        embs = np.zeros((len(chunks_batch), dim), dtype=np.float32)
                    finalize_batch(chunks_batch, embs)

            for chunk in _iter_chunks(chunks_path, args.limit):
                cid = chunk.get("chunk_id")
                if not cid:
                    continue
                if cid in existing_ids:
                    skipped_existing += 1
                    continue

                batch_chunks.append(chunk)
                batch_texts.append(_prepare_embedding_text(chunk))

                if len(batch_texts) >= args.batch_size:
                    submit_current_batch()
                    if len(future_to_chunks) >= max_pending:
                        drain_some(block_until_one=True)

            # Last partial batch
            submit_current_batch()

            # Drain remaining tasks
            while future_to_chunks:
                drain_some(block_until_one=True)

    logger.info(f"Saving FAISS index ({index.ntotal:,} vectors)...")
    faiss.write_index(index, str(index_path))
    logger.info(f"  Saved FAISS index -> {index_path}")

    # Commit metadata only after FAISS write succeeds.
    if build_mode == "incremental":
        with open(meta_path, "a", encoding="utf-8") as dst, open(
            staging_meta_path, "r", encoding="utf-8"
        ) as src:
            shutil.copyfileobj(src, dst)
        try:
            staging_meta_path.unlink()
        except Exception:
            pass
    else:
        staging_meta_path.replace(meta_path)

    _write_config(
        config_path,
        model_name=model_name,
        dim=dim,
        n_vectors=int(index.ntotal),
        build_mode=build_mode,
        source_path=chunks_path,
        batch_size=args.batch_size,
        workers=args.workers,
        added_vectors=processed,
        skipped_existing=skipped_existing,
        previous_n_vectors=previous_n_vectors,
        previous_config=previous_config,
    )

    # Optional quick test
    if not args.skip_test and index.ntotal > 0:
        logger.info("\n  Quick vector test:")
        for q in ["cong chuc vien chuc", "thi dua khen thuong", "chinh quyen dia phuong"]:
            try:
                q_emb = embed_batch([q], model_name)
                faiss.normalize_L2(q_emb)
                scores, indices = index.search(q_emb, 3)
                logger.info(f"  Query '{q}'")
                for score, idx in zip(scores[0], indices[0]):
                    logger.info(f"    [{score:.4f}] vec#{idx}")
            except Exception as e:
                logger.warning(f"  Test query failed '{q}': {e}")

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(
        f"✅ PHASE 3b COMPLETE - added {processed:,} vectors "
        f"(skipped existing {skipped_existing:,}) in {elapsed:.1f}s"
    )
    logger.info(f"  Total vectors now: {index.ntotal:,}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Index at: {index_dir}")
    logger.info("=" * 60)

    # Close shared HTTP client
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        try:
            _HTTP_CLIENT.close()
        except Exception:
            pass
        _HTTP_CLIENT = None


if __name__ == "__main__":
    main()
