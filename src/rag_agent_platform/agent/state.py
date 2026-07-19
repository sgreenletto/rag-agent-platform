"""Typed state contract reserved for the later LangGraph workflow."""

from typing import TypedDict

from rag_agent_platform.models import Citation, QueryType, RetrievalStrategy, RetrievedChunk


class AgentState(TypedDict):
    """Complete state shape for the future Agentic RAG graph."""

    original_query: str
    current_query: str
    document_ids: list[str]
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    retrieved_chunks: list[RetrievedChunk]
    retrieval_sufficient: bool
    answer: str
    citations: list[Citation]
    answer_passed: bool
    evaluation_reason: str
    retry_count: int
    max_retries: int
    execution_trace: list[str]
    error: str | None
