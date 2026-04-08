"""
Phase 3b — Build Vector Index (FAISS via Ollama Embeddings)
=============================================================
Đọc chunks.jsonl, embed qua Ollama (Qwen3-Embedding-0.6B),
build FAISS index.

Usage:
    python scripts/04_index_vector.py
    python scripts/04_index_vector.py --limit 1000       # Test với 1000 chunks 
    python scripts/04_index_vector.py --batch-size 32     # Smaller batches

Yêu cầu:
    - Ollama phải đang chạy: ollama serve
    - Model embedding đã pull: ollama pull qwen3-embedding:0.6b  

Output:
    indexes/vector/faiss.index     — FAISS index file
    indexes/vector/metadata.jsonl  — chunk_id mapping (cùng thứ tự)
    indexes/vector/config.json     — index config (dim, model, etc.)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from loguru import logger

from config.settings import cfg

# ── Logger setup ──
logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
log_file = cfg.LOG_DIR / "04_index_vector.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(str(log_file), level="DEBUG", rotation="10 MB")


def check_ollama():
    """Kiểm tra Ollama server đang chạy."""
    import httpx
    try:
        r = httpx.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama OK — {len(models)} models available")
            return True, models
        return False, []
    except Exception as e:
        logger.error(f"Ollama không phản hồi: {e}")
        return False, []


def embed_batch(texts: list[str], model: str = None) -> np.ndarray:
    """
    Gọi Ollama embedding API cho batch texts.
    Returns: numpy array shape (len(texts), dim)
    """
    import httpx
    
    model = model or cfg.EMBEDDING_MODEL
    url = f"{cfg.OLLAMA_BASE_URL}/api/embed"
    
    payload = {
        "model": model,
        "input": texts,
    }
    
    r = httpx.post(url, json=payload, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    embeddings = data.get("embeddings", [])
    
    if not embeddings:
        raise ValueError(f"Ollama trả về empty embeddings: {data}")
    
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, use_ivf: bool = False,
                      nlist: int = 256):
    """
    Build FAISS index.
    - Nếu < 50K vectors: dùng Flat (exact search, nhanh cho dataset nhỏ)
    - Nếu >= 50K: dùng IVF (approximate, cần training)
    """
    import faiss
    
    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    
    if use_ivf and n >= nlist * 40:
        logger.info(f"Building IVF index (nlist={nlist}, dim={dim}, n={n:,})")
        quantizer = faiss.IndexFlatIP(dim)  # Inner product
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train
        logger.info("  Training IVF...")
        index.train(embeddings)
        
        # Add
        logger.info("  Adding vectors...")
        index.add(embeddings)
        
        # Set nprobe for search quality
        index.nprobe = min(32, nlist // 4)
    else:
        logger.info(f"Building Flat index (dim={dim}, n={n:,})")
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalize)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    
    logger.info(f"  Index built: {index.ntotal:,} vectors")
    return index


def main():
    parser = argparse.ArgumentParser(description="Phase 3b: Build Vector Index")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số chunks embed")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size cho embedding API")
    parser.add_argument("--use-ivf", action="store_true",
                        help="Dùng IVF index (cho > 50K chunks)")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    
    cfg.ensure_dirs()
    t_start = time.time()
    
    chunks_path = Path(args.input) if args.input else cfg.PROCESSED_DIR / "chunks.jsonl"
    index_dir = cfg.VECTOR_INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PHASE 3b: BUILD VECTOR INDEX (FAISS)")
    logger.info(f"  Input: {chunks_path}")
    logger.info(f"  Index dir: {index_dir}")
    logger.info(f"  Embedding model: {cfg.EMBEDDING_MODEL}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Limit: {args.limit or 'ALL'}")
    logger.info("=" * 60)
    
    # Check Ollama
    logger.info("Checking Ollama...")
    ok, models = check_ollama()
    if not ok:
        logger.error("Ollama server chưa chạy! Hãy chạy: ollama serve")
        sys.exit(1)
    
    # Check nếu embedding model đã pull
    model_name = cfg.EMBEDDING_MODEL
    if not any(model_name in m for m in models):
        logger.warning(f"Model '{model_name}' chưa pull. Đang thử pull...")
        logger.warning(f"  Chạy: ollama pull {model_name}")
        # Thử pull tự động
        import httpx
        try:
            r = httpx.post(
                f"{cfg.OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=600
            )
            logger.info(f"  Pull result: {r.status_code}")
        except Exception as e:
            logger.error(f"  Không thể pull model: {e}")
            sys.exit(1)
    
    # Load chunks
    if not chunks_path.exists():
        logger.error(f"File không tồn tại: {chunks_path}")
        logger.error("Hãy chạy Phase 2 trước: python scripts/02_chunk.py")
        sys.exit(1)
    
    logger.info("Loading chunks...")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            chunks.append(json.loads(line.strip()))
    logger.info(f"  Loaded {len(chunks):,} chunks")
    
    # Embed in batches
    logger.info(f"Embedding {len(chunks):,} chunks (batch_size={args.batch_size})...")
    
    all_embeddings = []
    batch_texts = []
    errors = 0
    
    for i, chunk in enumerate(chunks):
        # Dùng text field cho embedding (không dùng text_for_keyword vì quá dài)
        text = chunk.get("text", "")
        if not text.strip():
            text = chunk.get("title", "empty")
        
        # Truncate nếu quá dài (embedding model có max context)
        if len(text) > 4000:
            text = text[:4000]
        
        batch_texts.append(text)
        
        # Process batch
        if len(batch_texts) >= args.batch_size or i == len(chunks) - 1:
            try:
                embeddings = embed_batch(batch_texts, model_name)
                all_embeddings.append(embeddings)
            except Exception as e:
                errors += 1
                logger.warning(f"  Batch {i//args.batch_size} failed: {e}")
                # Fallback: embed từng text
                for t in batch_texts:
                    try:
                        emb = embed_batch([t], model_name)
                        all_embeddings.append(emb)
                    except Exception as e2:
                        errors += 1
                        logger.warning(f"  Single embed failed: {e2}")
                        # Tạo zero vector
                        if all_embeddings:
                            dim = all_embeddings[0].shape[1]
                        else:
                            dim = 1024  # Default dim
                        all_embeddings.append(np.zeros((1, dim), dtype=np.float32))
            
            batch_texts = []
            
            done = i + 1
            if done % 500 == 0 or done == len(chunks):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - done) / rate if rate > 0 else 0
                logger.info(
                    f"  Embedded {done:,}/{len(chunks):,} "
                    f"({rate:.0f} chunks/s, ETA {eta:.0f}s)"
                )
    
    if not all_embeddings:
        logger.error("Không có embedding nào! Kiểm tra Ollama + model.")
        sys.exit(1)
    
    # Concat all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    logger.info(f"  Embeddings shape: {embeddings_matrix.shape}")
    
    if embeddings_matrix.shape[0] != len(chunks):
        logger.warning(
            f"  Mismatch: {embeddings_matrix.shape[0]} embeddings "
            f"vs {len(chunks)} chunks"
        )
        # Trim to min
        min_n = min(embeddings_matrix.shape[0], len(chunks))
        embeddings_matrix = embeddings_matrix[:min_n]
        chunks = chunks[:min_n]
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    import faiss
    
    use_ivf = args.use_ivf or len(chunks) > 50_000
    index = build_faiss_index(embeddings_matrix, use_ivf=use_ivf)
    
    # Save index
    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"  Saved FAISS index → {index_path}")
    
    # Save metadata mapping (chunk_ids in order)
    meta_path = index_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
                "path": chunk.get("path", ""),
                "article": chunk.get("article", ""),
                "clause": chunk.get("clause", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(f"  Saved metadata → {meta_path}")
    
    # Save config
    config = {
        "model": model_name,
        "dim": int(embeddings_matrix.shape[1]),
        "n_vectors": int(embeddings_matrix.shape[0]),
        "index_type": "IVF" if use_ivf else "Flat",
        "metric": "cosine (IP after L2 normalize)",
    }
    config_path = index_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved config → {config_path}")
    
    # Test search
    if not args.skip_test and chunks:
        logger.info("\n  🔍 Test vector search:")
        test_queries = [
            "công chức viên chức",
            "thi đua khen thưởng",
            "chính quyền địa phương",
        ]
        for q in test_queries:
            try:
                q_emb = embed_batch([q], model_name)
                faiss.normalize_L2(q_emb)
                scores, indices = index.search(q_emb, 3)
                
                logger.info(f"\n  Query: '{q}'")
                for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx >= 0 and idx < len(chunks):
                        c = chunks[idx]
                        logger.info(
                            f"    [{score:.4f}] {c['chunk_id']} — "
                            f"{c.get('path', '')[:60]}"
                        )
            except Exception as e:
                logger.warning(f"  Test query '{q}' failed: {e}")
    
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(
        f"✅ PHASE 3b COMPLETE — {len(chunks):,} vectors, "
        f"dim={embeddings_matrix.shape[1]}, {elapsed:.1f}s"
    )
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Index at: {index_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
