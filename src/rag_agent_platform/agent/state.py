"""Typed state shared by the LangGraph Agent workflow."""

from typing import TypedDict

from rag_agent_platform.models import Citation, QueryType, RetrievalStrategy, RetrievedChunk


class AgentState(TypedDict):
    """Complete state shape for one bounded Agentic RAG run."""

    original_query: str
    current_query: str
    document_ids: list[str] | None
    mode: str
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    retrieved_chunks: list[RetrievedChunk]
    retrieval_sufficient: bool
    answer: str
    citations: list[Citation]
    answer_passed: bool
    evaluation_reason: str
    suggested_query: str | None
    retry_count: int
    max_retries: int
    execution_trace: list[str]
    error: str | None
