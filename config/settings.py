"""
Centralized configuration — đọc từ .env file.
Import: from config.settings import cfg
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---- Load .env ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class _Settings:
    """Singleton chứa tất cả config. Dùng `cfg` instance bên dưới."""

    # -- Project paths --
    PROJECT_ROOT: Path = _PROJECT_ROOT

    # Data nguồn (từ workspace ChatBot/data)
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "f:/SpeechToText-indti/ChatBot/data"))
    LEGAL_DOCS_DIR: Path = DATA_DIR / "vietnamese-legal-documents"
    LEGAL_QA_DIR: Path = DATA_DIR / "vietnamese-legal-qa"

    # Output paths
    PROCESSED_DIR: Path = _PROJECT_ROOT / os.getenv("PROCESSED_DIR", "processed")
    INDEX_DIR: Path = _PROJECT_ROOT / os.getenv("INDEX_DIR", "indexes")
    BM25_INDEX_DIR: Path = INDEX_DIR / "bm25"
    VECTOR_INDEX_DIR: Path = INDEX_DIR / "vector"
    LOG_DIR: Path = _PROJECT_ROOT / "logs"

    # -- Ollama --
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen2.5:7b-instruct")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

    # -- Ingestion --
    MIN_CONTENT_LENGTH: int = 50          # Bỏ qua docs ngắn hơn 50 chars
    BATCH_LOG_INTERVAL: int = 10_000      # Log mỗi 10K docs

    # -- Chunking --
    MAX_CHUNK_TOKENS: int = 1024          # Max tokens per chunk
    MIN_CHUNK_TOKENS: int = 30            # Bỏ chunk quá ngắn
    OVERLAP_TOKENS: int = 64              # Context overlap

    # -- Retrieval --
    BM25_TOP_K: int = 20
    VECTOR_TOP_K: int = 20
    HYBRID_TOP_K: int = 10
    RRF_K: int = 60                       # RRF constant

    # -- API --
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MAX_HISTORY_TURNS: int = 5

    def ensure_dirs(self):
        """Tạo tất cả output directories nếu chưa có."""
        for d in [
            self.PROCESSED_DIR,
            self.BM25_INDEX_DIR,
            self.VECTOR_INDEX_DIR,
            self.LOG_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


cfg = _Settings()
