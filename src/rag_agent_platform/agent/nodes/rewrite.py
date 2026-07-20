"""Query rewrite and terminal refusal nodes."""

from rag_agent_platform.agent.router import QueryRewriter
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.generation import INSUFFICIENT_ANSWER


def rewrite_query(state: AgentState, rewriter: QueryRewriter) -> dict[str, object]:
    trace = list(state["execution_trace"])
    rewritten = rewriter.rewrite(
        original_query=state["original_query"],
        current_query=state["current_query"],
        evaluation_reason=state["evaluation_reason"],
        chunks=state["retrieved_chunks"],
        suggested_query=state["suggested_query"],
    )
    retry_count = state["retry_count"] + 1
    trace.append(
        f"rewrite_query: retry={retry_count}/{state['max_retries']}, "
        f"'{state['current_query']}' -> '{rewritten}'"
    )
    return {
        "current_query": rewritten,
        "retry_count": retry_count,
        "retrieved_chunks": [],
        "retrieval_sufficient": False,
        "answer": "",
        "citations": [],
        "answer_passed": False,
        "execution_trace": trace,
    }


def insufficient_answer(state: AgentState) -> dict[str, object]:
    trace = list(state["execution_trace"])
    trace.append(f"insufficient_answer: 达到重试上限 {state['max_retries']}，停止工作流")
    return {"answer": INSUFFICIENT_ANSWER, "citations": [], "execution_trace": trace}
