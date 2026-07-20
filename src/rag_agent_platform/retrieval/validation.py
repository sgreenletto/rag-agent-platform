"""Shared input, score and result helpers for retrieval implementations."""

from collections.abc import Iterable
from math import isfinite

from rag_agent_platform.models import RetrievedChunk


def validate_retrieval_request(query: str, top_k: int) -> str:
    """Validate common request fields and return the stripped query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    return normalized_query


def min_max_normalize(scores: Iterable[float]) -> list[float]:
    """Normalize finite scores to [0, 1], preserving equal positive scores."""
    values = list(scores)
    if any(not isfinite(value) for value in values):
        raise ValueError("scores must contain only finite values")
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [1.0 for _ in values]
    scale = maximum - minimum
    return [(value - minimum) / scale for value in values]


def finalize_results(
    chunks: Iterable[RetrievedChunk],
    document_ids: list[str] | None,
    top_k: int,
) -> list[RetrievedChunk]:
    """Apply document filtering, deterministic score ordering and top-k."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    allowed_ids = None if document_ids is None else set(document_ids)
    filtered = [
        chunk for chunk in chunks if allowed_ids is None or chunk.document_id in allowed_ids
    ]
    ranked = sorted(filtered, key=lambda chunk: (-chunk.normalized_score, chunk.chunk_id))
    deduplicated: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in ranked:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduplicated.append(chunk)
    return deduplicated[:top_k]


def validate_transformed_results(
    candidates: Iterable[RetrievedChunk],
    transformed: Iterable[RetrievedChunk],
    component: str,
    *,
    allow_content_change: bool = False,
) -> list[RetrievedChunk]:
    """Reject evidence injected or identity-mutated by a post-processing component."""
    provenance = {
        chunk.chunk_id: (
            chunk.content,
            chunk.source,
            chunk.document_id,
            chunk.parent_id,
            chunk.page,
        )
        for chunk in candidates
    }
    results = list(transformed)
    for chunk in results:
        expected = provenance.get(chunk.chunk_id)
        if expected is None:
            raise ValueError(f"{component} returned unknown chunk_id: {chunk.chunk_id}")
        content, source, document_id, parent_id, page = expected
        actual = (chunk.source, chunk.document_id, chunk.parent_id, chunk.page)
        if actual != (source, document_id, parent_id, page) or (
            not allow_content_change and chunk.content != content
        ):
            raise ValueError(f"{component} changed chunk provenance: {chunk.chunk_id}")
    return results
