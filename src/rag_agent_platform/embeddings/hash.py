"""Deterministic lightweight embeddings for tests and local smoke checks."""

from hashlib import sha256
from math import sqrt


class HashEmbeddingModel:
    """Generate stable normalized vectors without downloading a model."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")
        self._dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors for the provided texts."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for index, char in enumerate(text):
            digest = sha256(f"{index}:{char}".encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
