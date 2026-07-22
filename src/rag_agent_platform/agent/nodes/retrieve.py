"""Retriever delegation node."""

from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.validation import finalize_results


def retrieve_chunks(
    state: AgentState,
    retriever: BaseRetriever,
    *,
    top_k: int,
) -> dict[str, object]:
    """Call one injected retriever and normalize its public output."""
    trace = list(state["execution_trace"])
    name = type(retriever).__name__
    try:
        raw_chunks = retriever.retrieve(
            state["current_query"],
            document_ids=state["document_ids"],
            top_k=top_k,
        )
        if any(not isinstance(chunk, RetrievedChunk) for chunk in raw_chunks):
            raise TypeError(f"{name} must return list[RetrievedChunk]")
        raw_count = len(raw_chunks)
        chunks = finalize_results(raw_chunks, state["document_ids"], top_k)
    except Exception as exc:
        error = f"{name} 检索失败：{type(exc).__name__}: {exc}"
        trace.append(f"retrieve: {error}")
        return {
            "retrieved_chunks": [],
            "retrieval_sufficient": False,
            "error": error,
            "execution_trace": trace,
        }
    sufficient = bool(chunks)
    highest = chunks[0].normalized_score if chunks else 0.0
    trace.append(
        f"retrieve: retriever={name}, raw_results={raw_count}, results={len(chunks)}, "
        f"deduplicated={raw_count - len(chunks)}, top_score={highest:.4f}, "
        f"document_filter={'yes' if state['document_ids'] is not None else 'no'}"
    )
    return {
        "retrieved_chunks": chunks,
        "retrieval_sufficient": sufficient,
        "error": None,
        "execution_trace": trace,
    }
