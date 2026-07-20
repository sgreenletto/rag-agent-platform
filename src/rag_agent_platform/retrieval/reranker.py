"""Deterministic reranking and retriever composition interfaces."""

from abc import ABC, abstractmethod
from dataclasses import replace

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.tokenizer import tokenize
from rag_agent_platform.retrieval.validation import (
    finalize_results,
    validate_retrieval_request,
    validate_transformed_results,
)


class BaseReranker(ABC):
    """Reorder normalized retrieval candidates without changing their identity."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return at most top_k reranked chunks."""
        raise NotImplementedError


class TokenOverlapReranker(BaseReranker):
    """Dependency-light reranker combining token overlap and retrieval score."""

    def __init__(self, overlap_weight: float = 0.7) -> None:
        if not 0.0 <= overlap_weight <= 1.0:
            raise ValueError("overlap_weight must be between 0.0 and 1.0")
        self._overlap_weight = overlap_weight

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        query_tokens = set(tokenize(normalized_query))
        reranked: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_tokens = set(tokenize(chunk.content))
            overlap_score = (
                len(query_tokens.intersection(chunk_tokens)) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            rerank_score = (
                self._overlap_weight * overlap_score
                + (1.0 - self._overlap_weight) * chunk.normalized_score
            )
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "pre_rerank_score": chunk.normalized_score,
                    "token_overlap_score": overlap_score,
                    "rerank_score": rerank_score,
                }
            )
            reranked.append(
                replace(
                    chunk,
                    normalized_score=rerank_score,
                    retrieval_method=f"{chunk.retrieval_method}+token_rerank",
                    metadata=metadata,
                )
            )
        return finalize_results(reranked, None, top_k)


class RerankingRetriever(BaseRetriever):
    """Expand recall candidates and apply a pluggable reranker."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        *,
        candidate_multiplier: int = 4,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        candidate_k = top_k * self._candidate_multiplier
        candidates = self._retriever.retrieve(
            normalized_query,
            document_ids=document_ids,
            top_k=candidate_k,
        )
        candidates = finalize_results(candidates, document_ids, candidate_k)
        reranked = self._reranker.rerank(normalized_query, candidates, top_k)
        reranked = validate_transformed_results(candidates, reranked, "reranker")
        return finalize_results(reranked, document_ids, top_k)
