"""BM25 sparse retriever backed by a storage-agnostic chunk corpus."""

from threading import RLock
from typing import Any

from rank_bm25 import BM25Okapi

from rag_agent_platform.models import ChildChunk, RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.corpus import ChunkCorpus
from rag_agent_platform.retrieval.tokenizer import tokenize
from rag_agent_platform.retrieval.validation import (
    finalize_results,
    min_max_normalize,
    validate_retrieval_request,
)


class BM25Retriever(BaseRetriever):
    """Retrieve keyword-relevant chunks with a refreshable in-memory index."""

    def __init__(self, corpus: ChunkCorpus, score_threshold: float = 0.0) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        self._corpus = corpus
        self._score_threshold = score_threshold
        self._lock = RLock()
        self._chunks: list[ChildChunk] = []
        self._tokenized_chunks: list[list[str]] = []
        self._index: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the index from the corpus's current chunk snapshot."""
        chunks_by_id: dict[str, ChildChunk] = {}
        for chunk in self._corpus.list_chunks():
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
        chunks = list(chunks_by_id.values())
        tokenized_chunks = [tokenize(chunk.content) for chunk in chunks]
        index = BM25Okapi(tokenized_chunks) if tokenized_chunks else None
        with self._lock:
            self._chunks = chunks
            self._tokenized_chunks = tokenized_chunks
            self._index = index

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        query_tokens = tokenize(normalized_query)
        with self._lock:
            if self._index is None or not query_tokens:
                return []
            all_scores = [float(score) for score in self._index.get_scores(query_tokens)]
            query_token_set = set(query_tokens)
            allowed_ids = None if document_ids is None else set(document_ids)
            candidates = [
                (chunk, raw_score)
                for chunk, chunk_tokens, raw_score in zip(
                    self._chunks, self._tokenized_chunks, all_scores, strict=True
                )
                if query_token_set.intersection(chunk_tokens)
                and (allowed_ids is None or chunk.document_id in allowed_ids)
            ]
        normalized_scores = min_max_normalize(score for _, score in candidates)
        results = [
            self._to_result(chunk, raw_score, normalized_score)
            for (chunk, raw_score), normalized_score in zip(
                candidates, normalized_scores, strict=True
            )
            if normalized_score >= self._score_threshold
        ]
        return finalize_results(results, None, top_k)

    @staticmethod
    def _to_result(chunk: ChildChunk, raw_score: float, normalized_score: float) -> RetrievedChunk:
        metadata: dict[str, Any] = dict(chunk.metadata)
        metadata["bm25_score"] = raw_score
        source = str(
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("source_path")
            or chunk.document_id
        )
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            normalized_score=normalized_score,
            source=source,
            document_id=chunk.document_id,
            parent_id=chunk.parent_id,
            page=chunk.page,
            retrieval_method="bm25",
            metadata=metadata,
        )
