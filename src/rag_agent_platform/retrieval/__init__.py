"""Retrieval contracts and Mock implementation."""

from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.bm25 import BM25Retriever
from rag_agent_platform.retrieval.compression import (
    CompressionRetriever,
    ContextCompressor,
    SentenceContextCompressor,
)
from rag_agent_platform.retrieval.config import RetrievalParameters
from rag_agent_platform.retrieval.corpus import ChunkCorpus
from rag_agent_platform.retrieval.dense import DenseRetriever, DenseSearchBackend, DenseSearchHit
from rag_agent_platform.retrieval.fusion import reciprocal_rank_fusion
from rag_agent_platform.retrieval.hybrid import HybridRetriever
from rag_agent_platform.retrieval.mock import MockRetriever
from rag_agent_platform.retrieval.multi_query import (
    IdentityQueryTransformer,
    MultiQueryRetriever,
    QueryTransformer,
)
from rag_agent_platform.retrieval.parent import ParentContextRetriever
from rag_agent_platform.retrieval.pipelines import AdvancedRetriever, NaiveRetriever
from rag_agent_platform.retrieval.reranker import (
    BaseReranker,
    RerankingRetriever,
    TokenOverlapReranker,
)
from rag_agent_platform.retrieval.threshold import RelevanceThreshold, ThresholdRetriever

__all__ = [
    "AdvancedRetriever",
    "BM25Retriever",
    "BaseReranker",
    "BaseRetriever",
    "ChunkCorpus",
    "CompressionRetriever",
    "ContextCompressor",
    "DenseRetriever",
    "DenseSearchBackend",
    "DenseSearchHit",
    "HybridRetriever",
    "IdentityQueryTransformer",
    "MockRetriever",
    "MultiQueryRetriever",
    "NaiveRetriever",
    "ParentContextRetriever",
    "QueryTransformer",
    "RelevanceThreshold",
    "RetrievalParameters",
    "RerankingRetriever",
    "SentenceContextCompressor",
    "ThresholdRetriever",
    "TokenOverlapReranker",
    "reciprocal_rank_fusion",
]
