"""Rule-based Mock Agent used to prove the application wiring only."""

from rag_agent_platform.agent.base import AgentService
from rag_agent_platform.models import (
    AgentResult,
    Citation,
    QueryType,
    RetrievalStrategy,
)
from rag_agent_platform.retrieval.base import BaseRetriever


class MockAgentService(AgentService):
    """Route to Mock Retrievers; this is not a real Agentic RAG workflow."""

    _VALID_MODES = {"agent", "naive", "advanced", "graph"}
    _TRACE = [
        "analyze_query",
        "route_retriever",
        "retrieve",
        "generate",
        "evaluate",
        "finish",
    ]

    def __init__(
        self,
        naive_retriever: BaseRetriever,
        advanced_retriever: BaseRetriever,
        graph_retriever: BaseRetriever,
    ) -> None:
        self._retrievers = {
            RetrievalStrategy.NAIVE: naive_retriever,
            RetrievalStrategy.ADVANCED: advanced_retriever,
            RetrievalStrategy.GRAPH: graph_retriever,
        }

    def invoke(
        self,
        query: str,
        document_ids: list[str] | None = None,
        mode: str = "agent",
    ) -> AgentResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if mode not in self._VALID_MODES:
            raise ValueError(f"unsupported mode: {mode}")

        query_type, strategy = self._route(query, mode)
        chunks = []
        if strategy is not RetrievalStrategy.NONE:
            chunks = self._retrievers[strategy].retrieve(
                query=query,
                document_ids=document_ids,
                top_k=5,
            )

        citations = [
            Citation(
                index=index,
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        if strategy is RetrievalStrategy.NONE:
            answer = "你好！这是普通对话的 Mock 回复，当前没有调用真实大模型。"
        elif chunks:
            answer = (
                f"这是 `{strategy.value}` 模式的 Mock 回答，共获得 {len(chunks)} 条模拟证据。"
                "当前未调用真实 LLM、Embedding 或数据库。"
            )
        else:
            answer = (
                f"这是 `{strategy.value}` 模式的 Mock 回答，但所选文档没有模拟证据。"
                "当前未调用真实检索或生成服务。"
            )

        return AgentResult(
            answer=answer,
            citations=citations,
            strategy=strategy,
            query_type=query_type,
            retry_count=0,
            execution_trace=list(self._TRACE),
            retrieved_chunks=chunks,
        )

    @staticmethod
    def _route(query: str, mode: str) -> tuple[QueryType, RetrievalStrategy]:
        manual_routes = {
            "naive": (QueryType.SIMPLE, RetrievalStrategy.NAIVE),
            "advanced": (QueryType.COMPLEX, RetrievalStrategy.ADVANCED),
            "graph": (QueryType.RELATION, RetrievalStrategy.GRAPH),
        }
        if mode in manual_routes:
            return manual_routes[mode]

        normalized_query = query.strip()
        greetings = {"你好", "您好", "嗨", "hello", "hi", "你是谁", "你能做什么"}
        relation_words = ("关系", "关联", "联系", "影响", "实体", "分别负责")
        complex_words = ("结合", "综合", "比较", "对比", "根据以上", "多个文档")
        if normalized_query.lower() in greetings:
            return QueryType.CHAT, RetrievalStrategy.NONE
        if any(word in normalized_query for word in relation_words):
            return QueryType.RELATION, RetrievalStrategy.GRAPH
        if len(normalized_query) > 80 or any(word in normalized_query for word in complex_words):
            return QueryType.COMPLEX, RetrievalStrategy.ADVANCED
        return QueryType.SIMPLE, RetrievalStrategy.NAIVE
