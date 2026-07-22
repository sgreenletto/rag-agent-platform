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
from rag_agent_platform.evaluation import EvaluationDecision
from rag_agent_platform.evaluation.base import AnswerEvaluator
from rag_agent_platform.generation.base import AnswerGenerator, FallbackSynthesizer
from rag_agent_platform.generation.fallback import GroundedFallbackSynthesizer
from rag_agent_platform.llm import capture_transport_events, consume_transport_events
from rag_agent_platform.models import AgentResult, QueryType, RetrievalStrategy
from rag_agent_platform.responses import INSUFFICIENT_ANSWER
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
        max_regenerations: int = 1,
        fallback_synthesizer: FallbackSynthesizer | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if max_regenerations < 0:
            raise ValueError("max_regenerations must not be negative")
        self._max_retries = max_retries
        self._max_regenerations = max_regenerations
        self._graph = build_agent_graph(
            analyzer=analyzer or StructuredQueryAnalyzer(),
            rewriter=rewriter or BoundedQueryRewriter(),
            generator=generator,
            evaluator=evaluator,
            naive_retriever=naive_retriever,
            advanced_retriever=advanced_retriever,
            graph_retriever=graph_retriever,
            top_k=top_k,
            fallback_synthesizer=fallback_synthesizer or GroundedFallbackSynthesizer(),
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
            "evaluation_result": None,
            "evaluation_decision": EvaluationDecision.REWRITE_RETRIEVE,
            "evaluation_reason": "",
            "suggested_query": None,
            "retry_count": 0,
            "max_retries": self._max_retries,
            "regenerate_count": 0,
            "max_regenerations": self._max_regenerations,
            "query_history": [normalized_query],
            "strategy_history": [],
            "refused": False,
            "execution_trace": [],
            "error": None,
            "generation_failed": False,
        }
        with capture_transport_events():
            try:
                final = self._graph.invoke(
                    initial,
                    config={
                        "recursion_limit": (
                            12 + self._max_retries * 6 + self._max_regenerations * 3
                        )
                    },
                )
            except Exception as exc:
                error = f"Agent 工作流执行失败：{type(exc).__name__}: {exc}"
                return AgentResult(
                    answer=INSUFFICIENT_ANSWER,
                    citations=[],
                    strategy=RetrievalStrategy.NONE,
                    query_type=QueryType.SIMPLE,
                    retry_count=0,
                    regenerate_count=0,
                    refused=True,
                    evaluation_decision=EvaluationDecision.REFUSE.value,
                    query_history=[normalized_query],
                    execution_trace=[*consume_transport_events(), error],
                    error=error,
                )
            remaining_transport_events = consume_transport_events()
        if remaining_transport_events:
            final["execution_trace"] = [
                *final["execution_trace"],
                *remaining_transport_events,
            ]
        return AgentResult(
            answer=final["answer"] or INSUFFICIENT_ANSWER,
            citations=list(final["citations"]),
            strategy=final["retrieval_strategy"],
            query_type=final["query_type"],
            retry_count=final["retry_count"],
            regenerate_count=final["regenerate_count"],
            refused=final["refused"],
            evaluation_decision=final["evaluation_decision"].value,
            query_history=list(final["query_history"]),
            strategy_history=list(final["strategy_history"]),
            execution_trace=list(final["execution_trace"]),
            retrieved_chunks=list(final["retrieved_chunks"]),
            error=final["error"],
        )
