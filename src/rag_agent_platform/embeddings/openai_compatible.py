"""OpenAI-compatible embedding client used by providers such as DashScope."""

import json
from typing import Any
from urllib import error, request

from rag_agent_platform.embeddings.base import EmbeddingIdentity


class OpenAICompatibleEmbeddingModel:
    """Call an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        provider: str = "openai_compatible",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model.strip():
            raise ValueError("EMBEDDING_MODEL is required for openai-compatible embeddings")
        if not api_key.strip():
            raise ValueError("EMBEDDING_API_KEY is required for openai-compatible embeddings")
        if not base_url.strip():
            raise ValueError("EMBEDDING_BASE_URL is required for openai-compatible embeddings")
        self._model = model.strip()
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._provider = provider.strip().lower() or "openai_compatible"
        self._timeout_seconds = timeout_seconds
        self._dimensions: int | None = None

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return provider and model identity for Chroma compatibility checks."""
        return EmbeddingIdentity(
            provider=self._provider,
            model=self._model,
            base_url=self._base_url,
            dimensions=self._dimensions,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        if not texts:
            return []
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace").strip()
            detail = f": {message}" if message else ""
            raise RuntimeError(f"Embedding request failed: HTTP {exc.code}{detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        vectors = self._parse_vectors(body, expected_count=len(texts))
        self._remember_dimensions(vectors)
        return vectors

    def _parse_vectors(self, body: dict[str, Any], *, expected_count: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response did not contain a data list")
        ordered_items = sorted(
            data,
            key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
        )
        vectors = [self._parse_vector(item) for item in ordered_items]
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Embedding response count mismatch: expected {expected_count}, got {len(vectors)}"
            )
        return vectors

    @staticmethod
    def _parse_vector(item: Any) -> list[float]:
        if not isinstance(item, dict):
            raise RuntimeError("Embedding response item must be an object")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Embedding response item did not contain a vector")
        vector: list[float] = []
        for value in embedding:
            if not isinstance(value, int | float):
                raise RuntimeError("Embedding vector must contain only numbers")
            vector.append(float(value))
        return vector

    def _remember_dimensions(self, vectors: list[list[float]]) -> None:
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("Embedding response vectors must have the same dimensions")
        if self._dimensions is None:
            self._dimensions = dimensions
            return
        if self._dimensions != dimensions:
            raise RuntimeError(
                f"Embedding dimensions changed from {self._dimensions} to {dimensions}"
            )
