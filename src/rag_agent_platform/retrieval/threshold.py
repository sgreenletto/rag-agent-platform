"""Configurable no-answer decisions for normalized retrieval results."""

from dataclasses import dataclass

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.validation import finalize_results, validate_retrieval_request


@dataclass(frozen=True, slots=True)
class RelevanceThreshold:
    """Filter weak evidence and return no results when confidence is insufficient."""

    chunk_score_threshold: float = 0.0
    min_top_score: float = 0.0
    min_results: int = 1
    min_score_gap: float = 0.0

    def __post_init__(self) -> None:
        for name in ("chunk_score_threshold", "min_top_score", "min_score_gap"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_results <= 0:
            raise ValueError("min_results must be greater than 0")

    def apply(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return qualifying chunks, or an empty list for a no-answer decision."""
        ranked = sorted(chunks, key=lambda chunk: (-chunk.normalized_score, chunk.chunk_id))
        eligible = [
            chunk for chunk in ranked if chunk.normalized_score >= self.chunk_score_threshold
        ]
        if len(eligible) < self.min_results:
            return []
        if eligible[0].normalized_score < self.min_top_score:
            return []
        if len(eligible) > 1:
            score_gap = eligible[0].normalized_score - eligible[1].normalized_score
            if score_gap < self.min_score_gap:
                return []
        return eligible


class ThresholdRetriever(BaseRetriever):
    """Wrap a Retriever with an explicit normalized-score no-answer policy."""

    def __init__(
        self,
        retriever: BaseRetriever,
        policy: RelevanceThreshold,
        *,
        candidate_multiplier: int = 2,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")
        self._retriever = retriever
        self._policy = policy
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        candidate_k = max(top_k * self._candidate_multiplier, self._policy.min_results)
        candidates = self._retriever.retrieve(normalized_query, document_ids, candidate_k)
        candidates = finalize_results(candidates, document_ids, candidate_k)
        accepted = self._policy.apply(candidates)
        return finalize_results(accepted, document_ids, top_k)
