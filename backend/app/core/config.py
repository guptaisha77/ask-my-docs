"""Application configuration.

Single source of truth for every secret and tunable in the system.
Values resolve in priority order (highest wins): OS environment variables,
then the .env file, then the defaults below. An invalid value or combination
raises at import time, so the app refuses to start in a broken state.
"""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application metadata ─────────────────────────────────────────
    app_name: str = "ask-my-docs"
    environment: str = Field(default="development")

    # ── LLM provider (Groq) ──────────────────────────────────────────
    # Groq exposes an OpenAI-compatible API, so we use the OpenAI client
    # library but point its base_url at Groq with a Groq key.
    groq_api_key: str = Field(default="")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    groq_model: str = Field(default="llama-3.1-8b-instant")

    # ── Reranker (Cohere) ────────────────────────────────────────────
    cohere_api_key: str = Field(default="")
    cohere_rerank_model: str = Field(default="rerank-english-v3.0")

    # ── Vector store (ChromaDB) ──────────────────────────────────────
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection: str = Field(default="ask_my_docs")

    # ── Embeddings (runs locally, no API key) ────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # ── Chunking ─────────────────────────────────────────────────────
    # 700 tokens ≈ one dense page; 100-token overlap bridges chunk boundaries.
    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # ── BM25 keyword index ───────────────────────────────────────────
    bm25_index_path: str = Field(default="./bm25_index.pkl")

    # ── Retrieval ────────────────────────────────────────────────────
    # Cast a wide net (10 + 10 candidates), then the reranker trims to 5.
    bm25_top_k: int = Field(default=10, gt=0)
    vector_top_k: int = Field(default=10, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)

    # ── CI eval gates ────────────────────────────────────────────────
    # CI fails a PR if any RAGAS score drops below these. Scores are in [0, 1].
    eval_faithfulness_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    eval_answer_relevancy_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_context_precision_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # ── CORS ─────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # ── Validators ───────────────────────────────────────────────────

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        normalised = value.lower().strip()
        if normalised not in allowed:
            raise ValueError(
                f"environment must be one of {sorted(allowed)}, got '{value}'"
            )
        return normalised

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        # Overlap must be strictly smaller than chunk size, or chunks never advance.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        # Reranker can't return more results than were retrieved in total.
        max_candidates = self.bm25_top_k + self.vector_top_k
        if self.rerank_top_k > max_candidates:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) cannot exceed total "
                f"retrieved candidates ({max_candidates})"
            )
        # Fail fast: a missing key is fine in dev but fatal in production.
        if self.is_production and not self.has_groq_key:
            raise ValueError("GROQ_API_KEY is required when environment=production")
        return self

    # ── Convenience properties ───────────────────────────────────────

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def has_cohere_key(self) -> bool:
        return bool(self.cohere_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    @lru_cache builds Settings once and returns the same instance thereafter,
    so .env is read a single time. Tests can reset it with get_settings.cache_clear().
    """
    return Settings()


settings = get_settings()
