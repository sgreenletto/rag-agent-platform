"""Factory for shared embedding services."""

from rag_agent_platform.config import Settings
from rag_agent_platform.embeddings.base import EmbeddingModel
from rag_agent_platform.embeddings.hash import HashEmbeddingModel
from rag_agent_platform.embeddings.openai_compatible import OpenAICompatibleEmbeddingModel


def build_embedding_model(settings: Settings) -> EmbeddingModel:
    """Build the configured embedding model used by ingestion and retrieval."""
    provider = settings.embedding_provider.strip().lower()
    if provider in {"", "hash", "local"}:
        return HashEmbeddingModel(settings.hash_embedding_dimensions)
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleEmbeddingModel(
            provider="openai_compatible",
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or "https://api.openai.com/v1",
        )
    if provider == "siliconflow":
        return OpenAICompatibleEmbeddingModel(
            provider="siliconflow",
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or "https://api.siliconflow.cn/v1",
        )
    raise ValueError(
        f"unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'; "
        "use hash, siliconflow, or openai-compatible"
    )
