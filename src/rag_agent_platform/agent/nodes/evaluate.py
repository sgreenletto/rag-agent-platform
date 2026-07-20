"""Answer evaluation node."""

from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation.base import AnswerEvaluator, EvaluationResult


def evaluate_answer(state: AgentState, evaluator: AnswerEvaluator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    if not state["retrieved_chunks"]:
        result = EvaluationResult(False, "检索为空，回答没有知识库依据", state["original_query"])
    elif not _citations_match(state):
        result = EvaluationResult(False, "引用与检索证据不匹配", state["original_query"])
    else:
        try:
            result = evaluator.evaluate(
                state["original_query"],
                state["answer"],
                state["retrieved_chunks"],
            )
        except Exception as exc:
            result = EvaluationResult(
                False,
                f"回答评估失败：{type(exc).__name__}: {exc}",
                state["original_query"],
            )
    trace.append(f"evaluate_answer: passed={result.passed}, reason={result.reason}")
    return {
        "answer_passed": result.passed,
        "evaluation_reason": result.reason,
        "suggested_query": result.suggested_query,
        "execution_trace": trace,
    }


def _citations_match(state: AgentState) -> bool:
    chunks = state["retrieved_chunks"]
    citations = state["citations"]
    if len(citations) != len(chunks):
        return False
    return all(
        citation.index == index
        and citation.source == chunk.source
        and citation.page == chunk.page
        and citation.chunk_id == chunk.chunk_id
        for index, (citation, chunk) in enumerate(zip(citations, chunks, strict=True), start=1)
    )
