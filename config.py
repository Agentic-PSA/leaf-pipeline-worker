from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_role_key: str

    # Legal DB
    legal_db_base: str = "https://legal-documents-database.agentic.pl"
    legal_db_api_key: str

    # LLM (Google Gemini API - direct)
    gemini_api_key: str
    llm_model: str = "gemini-2.5-flash-lite"

    # Embeddings (OpenAI)
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Pipeline
    workers: int = 10
    batch_delay_ms: int = 500
    stagger_ms: int = 150

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
