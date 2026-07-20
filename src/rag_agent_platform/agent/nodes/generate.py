"""Grounded and direct generation nodes."""

from typing import Protocol

from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.generation import INSUFFICIENT_ANSWER, AnswerGenerator
from rag_agent_platform.models import Citation


class ChatCapableGenerator(Protocol):
    def generate_chat(self, query: str) -> str:
        """Generate a direct conversational response."""
        ...


def direct_generate(state: AgentState, generator: AnswerGenerator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    if hasattr(generator, "generate_chat"):
        answer = generator.generate_chat(state["original_query"])
    else:
        answer = "你好！我可以基于已入库文档回答问题。"
    trace.append("direct_generate: CHAT/NONE，未调用知识库 Retriever")
    return {"answer": answer, "citations": [], "answer_passed": True, "execution_trace": trace}


def generate_answer(state: AgentState, generator: AnswerGenerator) -> dict[str, object]:
    trace = list(state["execution_trace"])
    chunks = state["retrieved_chunks"]
    if not state["retrieval_sufficient"] or not chunks:
        trace.append("generate_answer: 证据不足，未生成确定性答案")
        return {"answer": INSUFFICIENT_ANSWER, "citations": [], "execution_trace": trace}
    try:
        answer, _ = generator.generate(state["original_query"], chunks)
    except Exception as exc:
        error = f"回答生成失败：{type(exc).__name__}: {exc}"
        trace.append(f"generate_answer: {error}")
        return {
            "answer": INSUFFICIENT_ANSWER,
            "citations": [],
            "error": error,
            "retry_count": state["max_retries"],
            "execution_trace": trace,
        }
    citations = [
        Citation(index=index, source=chunk.source, page=chunk.page, chunk_id=chunk.chunk_id)
        for index, chunk in enumerate(chunks, start=1)
    ]
    trace.append(f"generate_answer: 使用 {len(chunks)} 条真实证据生成回答")
    return {"answer": answer, "citations": citations, "execution_trace": trace}
