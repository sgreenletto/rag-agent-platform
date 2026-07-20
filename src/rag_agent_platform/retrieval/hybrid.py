"""Dense and sparse hybrid retrieval coordinated through RRF."""

from dataclasses import replace
from typing import Literal

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.fusion import reciprocal_rank_fusion
from rag_agent_platform.retrieval.validation import finalize_results, validate_retrieval_request


class HybridRetriever(BaseRetriever):
    """Combine dense and sparse candidate lists with weighted RRF."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        *,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        failure_mode: Literal["fallback", "raise"] = "fallback",
    ) -> None:
        if dense_weight <= 0 or sparse_weight <= 0:
            raise ValueError("retriever weights must be greater than 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")
        if failure_mode not in {"fallback", "raise"}:
            raise ValueError("failure_mode must be 'fallback' or 'raise'")
        self._retrievers = {"dense": dense_retriever, "sparse": sparse_retriever}
        self._weights = {"dense": dense_weight, "sparse": sparse_weight}
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier
        self._failure_mode = failure_mode

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        candidate_k = top_k * self._candidate_multiplier
        ranked_results: dict[str, list[RetrievedChunk]] = {}
        errors: dict[str, str] = {}

        for name, retriever in self._retrievers.items():
            try:
                ranked_results[name] = retriever.retrieve(
                    normalized_query,
                    document_ids=document_ids,
                    top_k=candidate_k,
                )
            except Exception as error:
                if self._failure_mode == "raise":
                    raise
                errors[name] = f"{type(error).__name__}: {error}"

        if not ranked_results:
            detail = "; ".join(f"{name}={error}" for name, error in errors.items())
            raise RuntimeError(f"all hybrid retrievers failed: {detail}")

        fused = reciprocal_rank_fusion(
            ranked_results,
            weights={name: self._weights[name] for name in ranked_results},
            rrf_k=self._rrf_k,
        )
        if errors:
            fused = [
                replace(chunk, metadata={**chunk.metadata, "retrieval_warnings": dict(errors)})
                for chunk in fused
            ]
        return finalize_results(fused, document_ids, top_k)
