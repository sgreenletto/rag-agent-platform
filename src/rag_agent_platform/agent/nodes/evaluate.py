"""Answer evaluation node with explicit workflow decisions."""

import re

from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation.base import (
    AnswerEvaluator,
    EvaluationDecision,
    EvaluationResult,
)
from rag_agent_platform.llm import consume_transport_events


def evaluate_answer(state: AgentState, evaluator: AnswerEvaluator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    chunks = state["retrieved_chunks"]
    if state["generation_failed"]:
        result = EvaluationResult(
            False,
            "模型调用失败但检索证据仍可用，转入确定性证据降级",
            decision=(EvaluationDecision.REGENERATE if chunks else EvaluationDecision.REFUSE),
            relevance_score=1.0 if chunks else 0.0,
            groundedness_score=0.0,
            completeness_score=0.0,
            citation_quality_score=0.0,
        )
    elif not _citations_match(state):
        result = EvaluationResult(
            False,
            "回答引用编号与去重后的检索证据不匹配",
            decision=(EvaluationDecision.REGENERATE if chunks else EvaluationDecision.REFUSE),
            relevance_score=1.0 if chunks else 0.0,
        )
    else:
        try:
            result = evaluator.evaluate(state["original_query"], state["answer"], chunks)
        except Exception as exc:
            result = EvaluationResult(
                False,
                f"回答评估失败：{type(exc).__name__}: {exc}",
                decision=(EvaluationDecision.REGENERATE if chunks else EvaluationDecision.REFUSE),
                relevance_score=1.0 if chunks else 0.0,
            )
    trace.extend(consume_transport_events())
    trace.append(
        "evaluate_answer: "
        f"decision={result.decision.value}, passed={result.passed}, "
        f"scores=relevance:{result.relevance_score:.2f},"
        f"groundedness:{result.groundedness_score:.2f},"
        f"completeness:{result.completeness_score:.2f},"
        f"citation_quality:{result.citation_quality_score:.2f}, "
        f"retry={state['retry_count']}/{state['max_retries']}, "
        f"regenerate={state['regenerate_count']}/{state['max_regenerations']}, "
        f"reason={result.reason}"
    )
    return {
        "answer_passed": result.passed,
        "evaluation_result": result,
        "evaluation_decision": result.decision,
        "evaluation_reason": result.reason,
        "suggested_query": result.suggested_query,
        "execution_trace": trace,
    }


def _citations_match(state: AgentState) -> bool:
    chunks = state["retrieved_chunks"]
    citations = state["citations"]
    markers = sorted({int(value) for value in re.findall(r"\[(\d+)]", state["answer"])})
    if [citation.index for citation in citations] != markers:
        return False
    if any(index < 1 or index > len(chunks) for index in markers):
        return False
    return all(
        citation.source == chunks[citation.index - 1].source
        and citation.page == chunks[citation.index - 1].page
        and citation.chunk_id == chunks[citation.index - 1].chunk_id
        for citation in citations
    )
