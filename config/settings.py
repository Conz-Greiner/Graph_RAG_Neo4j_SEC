from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMIngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_INGEST_", frozen=True)
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.5-9B-FP8-dynamic"
    timeout: float = 120.0
    max_concurrency: int = 32
    max_model_len: int = 16384

class LLMAnswerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_ANSWER_", frozen=True)
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen3.5-4B-FP8-dynamic"
    timeout: float = 60.0
    max_concurrency: int = 64
    streaming_enabled: bool = True

class OCRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", frozen=True)
    enabled: bool = False
    base_url: str = "http://localhost:8002/v1"
    timeout: float = 120.0
    fallback_threshold: float = 0.8

class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", frozen=True)
    url: str = "http://localhost:6333"
    api_key: SecretStr = SecretStr("")
    prefer_grpc: bool = True

class Neo4jSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", frozen=True)
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("password")

class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", frozen=True)
    model: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", frozen=True)
    max_iterations: int = 10
    router_threshold: float = 0.7

class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGSMITH_", frozen=True)
    api_key: SecretStr = SecretStr("")
    project: str = "graph-rag-sec"
    tracing: bool = True

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_ingest: LLMIngestSettings = Field(default_factory=LLMIngestSettings)
    llm_answer: LLMAnswerSettings = Field(default_factory=LLMAnswerSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

def get_settings() -> Settings:
    return Settings()