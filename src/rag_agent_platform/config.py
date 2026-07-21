"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings with safe development defaults."""

    app_env: str = "dev"
    app_mode: str = "real"
    app_name: str = "RAG Agent Platform"

    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""

    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    document_repository_provider: str = "file"
    metadata_path: str = "data/metadata/documents.json"
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "rag_child_chunks"
    graph_persist_directory: str = "data/graph"
    upload_directory: str = "data/uploads"
    retrieval_top_k: int = 5
    agent_max_retries: int = 2
    hash_embedding_dimensions: int = 32

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = "rag_agent"
    mysql_charset: str = "utf8mb4"
    mysql_connect_timeout: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
