"""Dense retriever and vector-store adapter boundary."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Protocol, runtime_checkable

from rag_agent_platform.models import ChildChunk, RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.validation import (
    finalize_results,
    min_max_normalize,
    validate_retrieval_request,
)


@dataclass(slots=True)
class DenseSearchHit:
    """Storage-neutral dense search hit with an unnormalized backend score."""

    chunk: ChildChunk
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.source.strip():
            raise ValueError("source must not be empty")


@runtime_checkable
class DenseSearchBackend(Protocol):
    """Adapter implemented by Chroma or another vector search provider."""

    def search(
        self,
        query: str,
        document_ids: list[str] | None,
        limit: int,
    ) -> list[DenseSearchHit]:
        """Return up to limit raw vector hits for the requested documents."""
        ...


class DenseRetriever(BaseRetriever):
    """Normalize backend vector scores into the shared retriever contract."""

    def __init__(
        self,
        backend: DenseSearchBackend,
        score_kind: Literal["similarity", "distance"] = "similarity",
        score_threshold: float = 0.0,
        candidate_multiplier: int = 2,
        candidate_k: int | None = None,
    ) -> None:
        if score_kind not in {"similarity", "distance"}:
            raise ValueError("score_kind must be 'similarity' or 'distance'")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        self._backend = backend
        self._score_kind = score_kind
        self._score_threshold = score_threshold
        self._candidate_multiplier = candidate_multiplier
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        candidate_limit = self._candidate_k or top_k * self._candidate_multiplier
        if candidate_limit < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        hits = self._backend.search(
            normalized_query,
            document_ids,
            limit=candidate_limit,
        )[:candidate_limit]
        ranking_scores = [
            hit.score if self._score_kind == "similarity" else -hit.score for hit in hits
        ]
        normalized_scores = min_max_normalize(ranking_scores)
        results = [
            self._to_result(hit, normalized_score)
            for hit, normalized_score in zip(hits, normalized_scores, strict=True)
            if normalized_score >= self._score_threshold
        ]
        return finalize_results(results, document_ids, top_k)

    def _to_result(self, hit: DenseSearchHit, normalized_score: float) -> RetrievedChunk:
        metadata = dict(hit.chunk.metadata)
        metadata.update(hit.metadata)
        metadata["dense_score"] = hit.score
        metadata["dense_score_kind"] = self._score_kind
        metadata["raw_score"] = hit.score
        metadata["score_type"] = self._score_kind
        metadata["normalized_score"] = normalized_score
        return RetrievedChunk(
            chunk_id=hit.chunk.chunk_id,
            content=hit.chunk.content,
            normalized_score=normalized_score,
            source=hit.source,
            document_id=hit.chunk.document_id,
            parent_id=hit.chunk.parent_id,
            page=hit.chunk.page,
            retrieval_method="dense",
            metadata=metadata,
        )
