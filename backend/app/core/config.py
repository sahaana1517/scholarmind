"""
Centralized configuration for ScholarMind.

Loads environment variables from .env and exposes them as a typed
settings object. Use `from backend.app.core.config import settings`
anywhere in the codebase.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory (resolved from this file's location)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration in one place."""

    # ─── LLM Providers ─────────────────────────────────────────
    GROQ_API_KEY: str
    HUGGINGFACE_API_KEY: str = ""  # optional

    # ─── Vector & Graph Databases ──────────────────────────────
    QDRANT_URL: str
    QDRANT_API_KEY: str
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # ─── Other Storage ─────────────────────────────────────────
    DATABASE_URL: str
    REDIS_URL: str
    REDIS_TOKEN: str

    # ─── Observability ─────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ─── Local Model Names ─────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ─── Project Constants ─────────────────────────────────────
    QDRANT_COLLECTION_NAME: str = "scholarmind_papers"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # ─── Paths ─────────────────────────────────────────────────
    PROJECT_ROOT: Path = PROJECT_ROOT
    PAPERS_RAW_DIR: Path = PROJECT_ROOT / "data" / "papers_raw"
    PAPERS_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "papers_processed"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance — import this everywhere
settings = Settings()