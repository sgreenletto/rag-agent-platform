"""Query analysis node."""

from rag_agent_platform.agent.router import QueryAnalyzer
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.llm import consume_transport_events
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
        trace.extend(consume_transport_events())
        query_type = analysis.query_type
        strategy = _AUTO_ROUTES[query_type] if analysis.needs_retrieval else RetrievalStrategy.NONE
        strategy_history = state["strategy_history"]
        if (
            strategy_history
            and strategy is RetrievalStrategy.GRAPH
            and strategy_history[-1] is not RetrievalStrategy.GRAPH
        ):
            proposed = strategy
            query_type = state["query_type"]
            strategy = strategy_history[-1]
            trace.append(
                "analyze_query: blocked_strategy_drift="
                f"{strategy.value}->{proposed.value}; preserved original intent"
            )
        trace.append(
            f"analyze_query: query_type={query_type.value}, strategy={strategy.value}, "
            f"reason={analysis.reason}"
        )
        if analysis.warning:
            trace.append(f"analyze_query: {analysis.warning}")
    history = list(state["strategy_history"])
    previous = history[-1] if history else None
    history.append(strategy)
    trace.append(
        "analyze_query: strategy_change="
        f"{previous.value if previous else 'initial'}->{strategy.value}"
    )
    return {
        "query_type": query_type,
        "retrieval_strategy": strategy,
        "strategy_history": history,
        "execution_trace": trace,
    }
