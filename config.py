from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pipeline API (edge function)
    pipeline_api_url: str  # e.g. https://.../functions/v1/leaf-pipeline
    pipeline_api_key: str  # anon key or dedicated API key

    # Legal DB
    legal_db_base: str = "http://host.docker.internal:3000"

    # LLM & Embeddings (OpenAI)
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Pipeline
    workers: int = 20
    batch_delay_ms: int = 500
    stagger_ms: int = 150

    # Worker API
    worker_api_token: str | None = None
    worker_api_log_limit: int = Field(default=500, ge=50, le=5000)