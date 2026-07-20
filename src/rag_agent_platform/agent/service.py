"""Concrete LangGraph AgentService implementation."""

from rag_agent_platform.agent.base import AgentService
from rag_agent_platform.agent.graph import build_agent_graph
from rag_agent_platform.agent.router import (
    BoundedQueryRewriter,
    QueryAnalyzer,
    QueryRewriter,
    StructuredQueryAnalyzer,
)
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation.base import AnswerEvaluator
from rag_agent_platform.generation.base import AnswerGenerator
from rag_agent_platform.generation.service import INSUFFICIENT_ANSWER
from rag_agent_platform.models import AgentResult, QueryType, RetrievalStrategy
from rag_agent_platform.retrieval.base import BaseRetriever


class LangGraphAgentService(AgentService):
    """Run manual or automatic RAG through a compiled StateGraph."""

    _VALID_MODES = {"agent", "naive", "advanced", "graph"}

    def __init__(
        self,
        *,
        naive_retriever: BaseRetriever,
        advanced_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
        generator: AnswerGenerator,
        evaluator: AnswerEvaluator,
        analyzer: QueryAnalyzer | None = None,
        rewriter: QueryRewriter | None = None,
        top_k: int = 5,
        max_retries: int = 2,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._max_retries = max_retries
        self._graph = build_agent_graph(
            analyzer=analyzer or StructuredQueryAnalyzer(),
            rewriter=rewriter or BoundedQueryRewriter(),
            generator=generator,
            evaluator=evaluator,
            naive_retriever=naive_retriever,
            advanced_retriever=advanced_retriever,
            graph_retriever=graph_retriever,
            top_k=top_k,
        )

    def invoke(
        self,
        query: str,
        document_ids: list[str] | None = None,
        mode: str = "agent",
    ) -> AgentResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if mode not in self._VALID_MODES:
            allowed = ", ".join(sorted(self._VALID_MODES))
            raise ValueError(f"unsupported mode: {mode}; allowed modes: {allowed}")
        initial: AgentState = {
            "original_query": normalized_query,
            "current_query": normalized_query,
            "document_ids": None if document_ids is None else list(document_ids),
            "mode": mode,
            "query_type": QueryType.SIMPLE,
            "retrieval_strategy": RetrievalStrategy.NONE,
            "retrieved_chunks": [],
            "retrieval_sufficient": False,
            "answer": "",
            "citations": [],
            "answer_passed": False,
            "evaluation_reason": "",
            "suggested_query": None,
            "retry_count": 0,
            "max_retries": self._max_retries,
            "execution_trace": [],
            "error": None,
        }
        try:
            final = self._graph.invoke(
                initial,
                config={"recursion_limit": 12 + self._max_retries * 6},
            )
        except Exception as exc:
            error = f"Agent 工作流执行失败：{type(exc).__name__}: {exc}"
            return AgentResult(
                answer=INSUFFICIENT_ANSWER,
                citations=[],
                strategy=RetrievalStrategy.NONE,
                query_type=QueryType.SIMPLE,
                retry_count=0,
                execution_trace=[error],
                error=error,
            )
        return AgentResult(
            answer=final["answer"] or INSUFFICIENT_ANSWER,
            citations=list(final["citations"]),
            strategy=final["retrieval_strategy"],
            query_type=final["query_type"],
            retry_count=final["retry_count"],
            execution_trace=list(final["execution_trace"]),
            retrieved_chunks=list(final["retrieved_chunks"]),
            error=final["error"],
        )
