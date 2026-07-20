"""Context compression components that preserve retrieval provenance."""

import re
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

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")


class ContextCompressor(ABC):
    """Compress retrieved evidence without changing chunk identity."""

    @abstractmethod
    def compress(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_chars: int,
    ) -> list[RetrievedChunk]:
        """Return evidence whose total content length is within max_chars."""
        raise NotImplementedError


class SentenceContextCompressor(ContextCompressor):
    """Select query-relevant sentences under a global character budget."""

    def compress(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_chars: int,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, 1)
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        query_tokens = set(tokenize(normalized_query))
        remaining = max_chars
        compressed: list[RetrievedChunk] = []

        for chunk in chunks:
            if remaining <= 0:
                break
            content = chunk.content.strip()
            selected = (
                content
                if len(content) <= remaining
                else self._select_sentences(content, query_tokens, remaining)
            )
            if not selected:
                continue
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "compression_method": "sentence_overlap",
                    "original_content_length": len(content),
                    "compressed_content_length": len(selected),
                }
            )
            compressed.append(replace(chunk, content=selected, metadata=metadata))
            remaining -= len(selected)
        return compressed

    @staticmethod
    def _select_sentences(content: str, query_tokens: set[str], budget: int) -> str:
        sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(content)]
        sentences = [sentence for sentence in sentences if sentence]
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (
                -len(query_tokens.intersection(tokenize(item[1]))),
                item[0],
            ),
        )
        selected: list[tuple[int, str]] = []
        used = 0
        for index, sentence in ranked:
            available = budget - used
            if available <= 0:
                break
            if len(sentence) <= available:
                selected.append((index, sentence))
                used += len(sentence)
            elif not selected:
                return sentence[:budget].strip()
        if not selected:
            return content[:budget].strip()
        return "".join(sentence for _, sentence in sorted(selected))


class CompressionRetriever(BaseRetriever):
    """Apply a context compressor to results from another Retriever."""

    def __init__(
        self,
        retriever: BaseRetriever,
        compressor: ContextCompressor,
        *,
        max_chars: int = 4000,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        self._retriever = retriever
        self._compressor = compressor
        self._max_chars = max_chars

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        chunks = self._retriever.retrieve(normalized_query, document_ids, top_k)
        chunks = finalize_results(chunks, document_ids, top_k)
        compressed = self._compressor.compress(normalized_query, chunks, self._max_chars)
        compressed = validate_transformed_results(
            chunks, compressed, "compressor", allow_content_change=True
        )
        return finalize_results(compressed, document_ids, top_k)
