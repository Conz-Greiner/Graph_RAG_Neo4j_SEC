"""
Centralized configuration for the Agentic Graph-RAG system.

Uses Pydantic v2 BaseSettings to load, validate, and document every
runtime parameter from environment variables / .env files.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
    ConfigDict,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested configuration models
# ---------------------------------------------------------------------------
class LLMSettings(BaseModel):
    """Settings for the generative LLM served via vLLM."""
    model_config = ConfigDict(frozen=True)  # Safe and highly performant in 2.14+

    provider: Literal["openai_compatible", "anthropic", "azure"] = Field(
        default="openai_compatible",
        description="Backend provider. vLLM exposes an OpenAI-compatible API.",
    )
    base_url: str = Field(
        default="http://vllm:8000/v1",
        description="Base URL of the vLLM OpenAI-compatible endpoint.",
    )
    api_key: SecretStr = Field(
        default=SecretStr("EMPTY"),
        description="API key. vLLM accepts any non-empty string; use 'EMPTY'.",
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-14B-Instruct",
        description="Model identifier as registered in vLLM.",
    )

    enable_auto_tool_choice: bool = Field(
        default=True,
        description="Enables vLLM server-side automatic tool choice parsing."
    )
    tool_call_parser: str = Field(
        default="qwen3_coder",
        description="vLLM parser for Qwen tool calling format."
    )
    reasoning_parser: str = Field(
        default="qwen3",
        description="vLLM parser for Qwen reasoning/thinking tokens."
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0, le=16384)
    timeout_s: int = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)


class EmbeddingSettings(BaseModel):
    """Settings for the dense embedding model (BGE / E5 / GTE family)."""

    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace model id for dense embeddings.",
    )
    batch_size: int = Field(default=64, gt=0, le=512)
    dimensions: int = Field(default=1024, gt=0)
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    normalize: bool = Field(default=True)


class RerankerSettings(BaseModel):
    """Settings for the cross-encoder reranker."""

    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    top_k: int = Field(default=5, gt=0, le=50)
    batch_size: int = Field(default=32, gt=0, le=256)
    score_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)


class QdrantSettings(BaseModel):
    """Qdrant vector store connection parameters."""

    host: str = Field(default="qdrant")
    port: int = Field(default=6333, gt=0, le=65535)
    grpc_port: int = Field(default=6334, gt=0, le=65535)
    api_key: SecretStr | None = Field(default=None)
    prefer_grpc: bool = Field(default=True)
    https: bool = Field(default=False)
    collection_chunks: str = Field(default="doc_chunks")
    collection_communities: str = Field(default="community_summaries")
    vector_size: int = Field(default=1024, gt=0)
    distance: Literal["Cosine", "Euclid", "Dot"] = Field(default="Cosine")
    on_disk: bool = Field(default=True)
    replication_factor: int = Field(default=1, ge=1)


class Neo4jSettings(BaseModel):
    """Neo4j graph database connection parameters."""

    uri: str = Field(default="bolt://neo4j:7687")
    user: str = Field(default="neo4j")
    password: SecretStr = Field(default=SecretStr("changeme"))
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50, gt=0)
    connection_acquisition_timeout_s: int = Field(default=60, gt=0)
    max_transaction_retry_time_s: int = Field(default=30, gt=0)


class ParsingSettings(BaseModel):
    """Document parsing and chunking parameters."""

    chunk_size: int = Field(default=1024, gt=64, le=8192)
    chunk_overlap: int = Field(default=128, ge=0)
    min_chunk_size: int = Field(default=50, gt=0)
    pdf_parser: Literal["marker", "pypdf", "pdfplumber"] = Field(default="marker")
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    ocr_languages: list[str] = Field(default=["en"])


class AgentSettings(BaseModel):
    """LangGraph agent behavior parameters."""

    max_retries: int = Field(default=3, ge=0, le=10)
    router_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    graphrag_enabled: bool = Field(default=True)
    graphrag_community_level: int = Field(default=2, ge=0, le=5)
    max_hops: int = Field(default=4, ge=1, le=10)
    citation_required: bool = Field(default=True)
    min_citations: int = Field(default=1, ge=0)
    self_correction_enabled: bool = Field(default=True)


class IngestionSettings(BaseModel):
    """Entity extraction and graph ingestion parameters."""

    entity_types: list[str] = Field(
        default=["ORG", "PERSON", "GPE", "MONEY", "DATE", "PRODUCT", "CONCEPT"]
    )
    relation_types: list[str] = Field(
        default=["INVESTS_IN", "ACQUIRES", "OWES", "REPORTS_TO", "LOCATED_IN", "PARTNERS_WITH"]
    )
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, gt=0)


class APISettings(BaseModel):
    """FastAPI server parameters."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, gt=0, le=65535)
    workers: int = Field(default=2, ge=1, le=16)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    cors_origins: list[str] = Field(default=["http://localhost:8501"])
    rate_limit_per_minute: int = Field(default=60, gt=0)


class LangSmithSettings(BaseModel):
    """Optional LangSmith tracing parameters."""

    enabled: bool = Field(default=False)
    api_key: SecretStr | None = Field(default=None)
    project: str = Field(default="agentic-graph-rag")
    endpoint: str = Field(default="https://api.smith.langchain.com")


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Root application settings.

    All values are sourced from environment variables (with the prefix
    ``AGR_``) or from the ``.env`` file located at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AGR_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application identity ---
    app_name: str = Field(default="agentic-graph-rag")
    environment: Literal["development", "staging", "production"] = Field(
        default="production"
    )
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "console"] = Field(default="json")
    project_root: Path = Field(default=Path("/app"))

    # --- Nested sections ---
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    api: APISettings = Field(default_factory=APISettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    # --- Cross-field validation ---
    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        if self.parsing.chunk_overlap >= self.parsing.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if self.embedding.dimensions != self.qdrant.vector_size:
            raise ValueError(
                "embedding.dimensions must equal qdrant.vector_size "
                f"(got {self.embedding.dimensions} vs {self.qdrant.vector_size})"
            )
        if self.environment == "production" and self.debug:
            raise ValueError("debug=True is not allowed in production environment")
        return self

    @field_validator("project_root", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------
def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog + stdlib logging based on settings."""
    settings = settings or get_settings()

    log_level = getattr(logging, settings.log_level.upper())

    # Stdlib root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    logging.basicConfig(handlers=[handler], level=log_level, force=True)

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3", "neo4j", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Structlog processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
settings: Settings = get_settings()
configure_logging(settings)
logger = structlog.get_logger("agentic_graph_rag.config")