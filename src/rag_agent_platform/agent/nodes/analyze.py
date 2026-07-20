"""Query analysis node."""

from rag_agent_platform.agent.router import QueryAnalyzer
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.models import QueryType, RetrievalStrategy

_MANUAL_ROUTES = {
    "naive": (QueryType.SIMPLE, RetrievalStrategy.NAIVE),
    "advanced": (QueryType.COMPLEX, RetrievalStrategy.ADVANCED),
    "graph": (QueryType.RELATION, RetrievalStrategy.GRAPH),
}
_AUTO_ROUTES = {
    QueryType.SIMPLE: RetrievalStrategy.NAIVE,
    QueryType.COMPLEX: RetrievalStrategy.ADVANCED,
    QueryType.RELATION: RetrievalStrategy.GRAPH,
    QueryType.CHAT: RetrievalStrategy.NONE,
}


def analyze_query(state: AgentState, analyzer: QueryAnalyzer) -> dict[str, object]:
    """Respect manual modes or classify an automatic Agent request."""
    mode = state["mode"]
    trace = list(state["execution_trace"])
    if mode in _MANUAL_ROUTES:
        query_type, strategy = _MANUAL_ROUTES[mode]
        trace.append(f"analyze_query: 手动模式固定为 {strategy.value}")
    else:
        analysis = analyzer.analyze(state["current_query"])
        query_type = analysis.query_type
        strategy = _AUTO_ROUTES[query_type] if analysis.needs_retrieval else RetrievalStrategy.NONE
        trace.append(
            f"analyze_query: query_type={query_type.value}, strategy={strategy.value}, "
            f"reason={analysis.reason}"
        )
        if analysis.warning:
            trace.append(f"analyze_query: {analysis.warning}")
    return {
        "query_type": query_type,
        "retrieval_strategy": strategy,
        "execution_trace": trace,
    }
