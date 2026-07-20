"""Rank fusion utilities for combining heterogeneous retriever results."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

from rag_agent_platform.models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_results: Mapping[str, Sequence[RetrievedChunk]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse named ranked lists by chunk ID using weighted RRF."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")
    configured_weights = dict(weights or {})
    unknown_names = configured_weights.keys() - ranked_results.keys()
    if unknown_names:
        raise ValueError(f"weights contain unknown retrievers: {sorted(unknown_names)}")

    fused_scores: dict[str, float] = {}
    canonical_chunks: dict[str, RetrievedChunk] = {}
    details: dict[str, dict[str, dict[str, float | int]]] = {}

    for retriever_name, chunks in ranked_results.items():
        weight = configured_weights.get(retriever_name, 1.0)
        if not isfinite(weight) or weight <= 0:
            raise ValueError("retriever weights must be finite and greater than 0")
        seen: set[str] = set()
        for rank, chunk in enumerate(chunks, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            canonical_chunks.setdefault(chunk.chunk_id, chunk)
            contribution = weight / (rrf_k + rank)
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + contribution
            details.setdefault(chunk.chunk_id, {})[retriever_name] = {
                "rank": rank,
                "normalized_score": chunk.normalized_score,
                "rrf_contribution": contribution,
            }

    if not fused_scores:
        return []

    maximum = max(fused_scores.values())
    fused: list[RetrievedChunk] = []
    for chunk_id, raw_score in fused_scores.items():
        chunk = canonical_chunks[chunk_id]
        metadata: dict[str, Any] = dict(chunk.metadata)
        previous_details = metadata.get("retriever_results")
        if previous_details is not None:
            retrieval_stages = dict(metadata.get("retrieval_stages", {}))
            retrieval_stages[chunk.retrieval_method] = previous_details
            metadata["retrieval_stages"] = retrieval_stages
        metadata["rrf_score"] = raw_score
        metadata["retriever_results"] = details[chunk_id]
        fused.append(
            replace(
                chunk,
                normalized_score=raw_score / maximum,
                retrieval_method="hybrid_rrf",
                metadata=metadata,
            )
        )
    return sorted(fused, key=lambda chunk: (-chunk.normalized_score, chunk.chunk_id))
