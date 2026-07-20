"""Query transformation and multi-query retrieval composition."""

from dataclasses import replace
from typing import Protocol, runtime_checkable

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.fusion import reciprocal_rank_fusion
from rag_agent_platform.retrieval.validation import finalize_results, validate_retrieval_request


@runtime_checkable
class QueryTransformer(Protocol):
    """Generate alternate search queries without prescribing an LLM provider."""

    def transform(self, query: str) -> list[str]:
        """Return zero or more query rewrites."""
        ...


class IdentityQueryTransformer:
    """Safe fallback used when no rewrite model is configured."""

    def transform(self, query: str) -> list[str]:
        return [query]


class MultiQueryRetriever(BaseRetriever):
    """Retrieve with the original query and deduplicated rewrites, then fuse."""

    def __init__(
        self,
        retriever: BaseRetriever,
        transformer: QueryTransformer | None = None,
        *,
        max_queries: int = 4,
        candidate_multiplier: int = 2,
        rrf_k: int = 60,
        fallback_on_transform_error: bool = True,
    ) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than 0")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        self._retriever = retriever
        self._transformer = transformer or IdentityQueryTransformer()
        self._max_queries = max_queries
        self._candidate_multiplier = candidate_multiplier
        self._rrf_k = rrf_k
        self._fallback_on_transform_error = fallback_on_transform_error

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        transform_warning: str | None = None
        try:
            rewrites = self._transformer.transform(normalized_query)
            queries = self._prepare_queries(normalized_query, rewrites)
        except Exception as error:
            if not self._fallback_on_transform_error:
                raise
            queries = [normalized_query]
            transform_warning = f"{type(error).__name__}: {error}"

        candidate_k = top_k * self._candidate_multiplier
        ranked_results = {
            f"query_{index}": self._retriever.retrieve(
                variant,
                document_ids=document_ids,
                top_k=candidate_k,
            )
            for index, variant in enumerate(queries)
        }
        fused = reciprocal_rank_fusion(ranked_results, rrf_k=self._rrf_k)
        query_map = {f"query_{index}": variant for index, variant in enumerate(queries)}
        annotated: list[RetrievedChunk] = []
        for chunk in fused:
            metadata = {**chunk.metadata, "query_variants": query_map}
            if transform_warning is not None:
                metadata["query_transform_warning"] = transform_warning
            annotated.append(replace(chunk, metadata=metadata))
        return finalize_results(annotated, document_ids, top_k)

    def _prepare_queries(self, original: str, rewrites: object) -> list[str]:
        if not isinstance(rewrites, list) or any(not isinstance(item, str) for item in rewrites):
            raise TypeError("query transformer must return list[str]")
        unique: list[str] = []
        seen: set[str] = set()
        for candidate in [original, *rewrites]:
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
            if len(unique) == self._max_queries:
                break
        return unique
