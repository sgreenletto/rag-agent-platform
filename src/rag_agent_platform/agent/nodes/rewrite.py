"""Query rewrite, clarification and terminal refusal nodes."""

from rag_agent_platform.agent.router import QueryRewriter
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation import EvaluationDecision
from rag_agent_platform.llm import consume_transport_events
from rag_agent_platform.responses import INSUFFICIENT_ANSWER


def rewrite_query(state: AgentState, rewriter: QueryRewriter) -> dict[str, object]:
    trace = list(state["execution_trace"])
    previous_query = state["current_query"]
    rewritten = rewriter.rewrite(
        original_query=state["original_query"],
        current_query=previous_query,
        evaluation_reason=state["evaluation_reason"],
        chunks=state["retrieved_chunks"],
        suggested_query=state["suggested_query"],
    )
    trace.extend(consume_transport_events())
    retry_count = state["retry_count"] + 1
    history = list(state["query_history"])
    history.append(rewritten)
    trace.append(
        f"rewrite_query: retry={retry_count}/{state['max_retries']}, "
        f"before='{previous_query}', after='{rewritten}'"
    )
    validation_note = getattr(rewriter, "last_validation_reason", "")
    if validation_note:
        trace.append(f"rewrite_query: semantic_validation={validation_note}")
    return {
        "current_query": rewritten,
        "retry_count": retry_count,
        "query_history": history,
        "retrieved_chunks": [],
        "retrieval_sufficient": False,
        "answer": "",
        "citations": [],
        "answer_passed": False,
        "evaluation_result": None,
        "evaluation_decision": EvaluationDecision.REWRITE_RETRIEVE,
        "generation_failed": False,
        "execution_trace": trace,
    }


def clarification_answer(state: AgentState) -> dict[str, object]:
    trace = list(state["execution_trace"])
    question = (state["suggested_query"] or "").strip()
    if not question:
        question = f"需要补充一个会影响制度结论的关键前提：{state['evaluation_reason']}"
    trace.append(f"clarification_answer: {state['evaluation_reason']}")
    return {
        "answer": question,
        "citations": [],
        "refused": False,
        "execution_trace": trace,
    }


def insufficient_answer(state: AgentState) -> dict[str, object]:
    trace = list(state["execution_trace"])
    decision = state["evaluation_decision"]
    if decision is EvaluationDecision.REFUSE:
        reason = f"知识库无可靠依据：{state['evaluation_reason']}"
    else:
        reason = f"达到重试上限（检索）{state['max_retries']}：{state['evaluation_reason']}"
    trace.append(f"insufficient_answer: decision=refuse, reason={reason}")
    return {
        "answer": INSUFFICIENT_ANSWER,
        "citations": [],
        "refused": True,
        "evaluation_decision": EvaluationDecision.REFUSE,
        "execution_trace": trace,
    }
