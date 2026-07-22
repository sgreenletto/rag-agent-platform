"""Grounded, regenerative and direct generation nodes."""

import re
from typing import Protocol

from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation import EvaluationDecision
from rag_agent_platform.generation import AnswerGenerator, FallbackSynthesizer
from rag_agent_platform.llm import consume_transport_events
from rag_agent_platform.models import Citation, RetrievedChunk
from rag_agent_platform.responses import INSUFFICIENT_ANSWER


class ChatCapableGenerator(Protocol):
    def generate_chat(self, query: str) -> str:
        """Generate a direct conversational response."""
        ...


def direct_generate(state: AgentState, generator: AnswerGenerator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    try:
        if hasattr(generator, "generate_chat"):
            answer = generator.generate_chat(state["original_query"])
        else:
            answer = "你好！我可以基于已入库文档回答问题。"
    except Exception as exc:
        trace.extend(consume_transport_events())
        error = f"对话模型暂时不可用：{type(exc).__name__}: {exc}"
        trace.append(f"direct_generate: {error}")
        return {
            "answer": "当前模型服务暂时不可用，请稍后再试。",
            "citations": [],
            "answer_passed": False,
            "evaluation_decision": EvaluationDecision.REFUSE,
            "refused": True,
            "error": error,
            "execution_trace": trace,
        }
    trace.extend(consume_transport_events())
    trace.append("direct_generate: CHAT/NONE，未调用知识库 Retriever")
    return {
        "answer": answer,
        "citations": [],
        "answer_passed": True,
        "evaluation_decision": EvaluationDecision.PASS,
        "execution_trace": trace,
    }


def prepare_regeneration(state: AgentState) -> dict[str, object]:
    """Count answer-only repair separately from retrieval retries."""
    trace = list(state["execution_trace"])
    count = state["regenerate_count"] + 1
    trace.append(
        f"regenerate: count={count}/{state['max_regenerations']}, query_and_strategy_unchanged"
    )
    return {"regenerate_count": count, "execution_trace": trace}


def generate_answer(state: AgentState, generator: AnswerGenerator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    chunks = state["retrieved_chunks"]
    if not state["retrieval_sufficient"] or not chunks:
        trace.append("generate_answer: 证据不足，未生成确定性答案")
        return {"answer": INSUFFICIENT_ANSWER, "citations": [], "execution_trace": trace}
    previous_answer = state["answer"]
    regenerating = (
        state["regenerate_count"] > 0
        and state["evaluation_decision"] is EvaluationDecision.REGENERATE
    )
    try:
        if regenerating and hasattr(generator, "regenerate"):
            evaluation = state["evaluation_result"]
            answer, generated_citations = generator.regenerate(
                state["original_query"],
                chunks,
                previous_answer=previous_answer,
                unsupported_claims=(evaluation.unsupported_claims if evaluation else []),
            )
        else:
            answer, generated_citations = generator.generate(state["original_query"], chunks)
    except Exception as exc:
        error = f"回答生成失败：{type(exc).__name__}: {exc}"
        trace.extend(consume_transport_events())
        trace.append(f"generate_answer: {error}")
        return {
            "answer": INSUFFICIENT_ANSWER,
            "citations": [],
            "error": error,
            "generation_failed": True,
            "execution_trace": trace,
        }
    trace.extend(consume_transport_events())
    citations = _citations_for_answer(answer, chunks, generated_citations)
    trace.append(
        f"generate_answer: mode={'regenerate' if regenerating else 'initial'}, "
        f"evidence={len(chunks)}, cited={len(citations)}"
    )
    return {
        "answer": answer,
        "citations": citations,
        "error": None,
        "generation_failed": False,
        "execution_trace": trace,
    }


def conservative_answer(
    state: AgentState,
    synthesizer: FallbackSynthesizer,
) -> dict[str, object]:
    """Terminate generation failure with a short deterministic evidence synthesis."""
    trace = list(state["execution_trace"])
    chunks = state["retrieved_chunks"]
    result = synthesizer.synthesize(
        state["original_query"],
        chunks,
        retrieval_strategy=state["retrieval_strategy"],
    )
    fallback_trigger = (
        "generation_transport_failure" if state["generation_failed"] else "达到重新生成上限"
    )
    trace.append(
        f"deterministic_fallback: trigger={fallback_trigger}, "
        f"sufficient={result.sufficient}, answer_chars={len(result.answer)}, "
        f"citations={len(result.citations)}, reason={result.reason}"
    )
    if not result.sufficient:
        return {
            "answer": INSUFFICIENT_ANSWER,
            "citations": [],
            "answer_passed": False,
            "evaluation_decision": EvaluationDecision.REFUSE,
            "refused": True,
            "error": None,
            "generation_failed": False,
            "execution_trace": trace,
        }
    return {
        "answer": result.answer,
        "citations": result.citations,
        "answer_passed": True,
        "evaluation_decision": EvaluationDecision.PASS,
        "refused": False,
        "error": None,
        "generation_failed": False,
        "execution_trace": trace,
    }


def _citations_for_answer(
    answer: str,
    chunks: list[RetrievedChunk],
    generated_citations: list[Citation],
) -> list[Citation]:
    """Map answer markers to the final deduplicated chunk order."""
    del generated_citations  # Identity is rebuilt from final chunks, never trusted positionally.
    markers = sorted({int(value) for value in re.findall(r"\[(\d+)]", answer)})
    return [
        Citation(
            index,
            chunks[index - 1].source,
            chunks[index - 1].page,
            chunks[index - 1].chunk_id,
        )
        for index in markers
        if 1 <= index <= len(chunks)
    ]
