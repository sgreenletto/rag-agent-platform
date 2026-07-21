"""Validated retrieval parameters sourced from the application's shared settings."""

from dataclasses import dataclass
from math import isfinite

from rag_agent_platform.config import Settings


@dataclass(frozen=True, slots=True)
class RetrievalParameters:
    """Member-two retrieval knobs without introducing another settings system."""

    top_k: int = 5
    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    rerank_top_k: int = 5
    score_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("retrieval top_k must be greater than 0")
        if self.dense_candidate_k < self.top_k:
            raise ValueError("dense_candidate_k must be greater than or equal to top_k")
        if self.bm25_candidate_k < self.top_k:
            raise ValueError("bm25_candidate_k must be greater than or equal to top_k")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")
        if not isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalParameters":
        """Read defaults from the existing application Settings instance."""
        return cls(
            top_k=settings.retrieval_top_k,
            dense_candidate_k=settings.dense_candidate_k,
            bm25_candidate_k=settings.bm25_candidate_k,
            rerank_top_k=settings.rerank_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
