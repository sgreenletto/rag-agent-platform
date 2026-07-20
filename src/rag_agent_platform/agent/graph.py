"""LangGraph StateGraph construction for Agentic RAG."""

from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from rag_agent_platform.agent.nodes.analyze import analyze_query
from rag_agent_platform.agent.nodes.evaluate import evaluate_answer
from rag_agent_platform.agent.nodes.generate import direct_generate, generate_answer
from rag_agent_platform.agent.nodes.retrieve import retrieve_chunks
from rag_agent_platform.agent.nodes.rewrite import insufficient_answer, rewrite_query
from rag_agent_platform.agent.router import QueryAnalyzer, QueryRewriter
from rag_agent_platform.agent.state import AgentState
from rag_agent_platform.evaluation.base import AnswerEvaluator
from rag_agent_platform.generation.base import AnswerGenerator
from rag_agent_platform.models import RetrievalStrategy
from rag_agent_platform.retrieval.base import BaseRetriever


def build_agent_graph(
    *,
    analyzer: QueryAnalyzer,
    rewriter: QueryRewriter,
    generator: AnswerGenerator,
    evaluator: AnswerEvaluator,
    naive_retriever: BaseRetriever,
    advanced_retriever: BaseRetriever,
    graph_retriever: BaseRetriever,
    top_k: int,
):
    """Compile the workflow, routing branches and bounded rewrite loop."""
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_query", partial(analyze_query, analyzer=analyzer))
    workflow.add_node("direct_generate", partial(direct_generate, generator=generator))
    workflow.add_node(
        "naive_retrieve", partial(retrieve_chunks, retriever=naive_retriever, top_k=top_k)
    )
    workflow.add_node(
        "advanced_retrieve",
        partial(retrieve_chunks, retriever=advanced_retriever, top_k=top_k),
    )
    workflow.add_node(
        "graph_retrieve", partial(retrieve_chunks, retriever=graph_retriever, top_k=top_k)
    )
    workflow.add_node("generate_answer", partial(generate_answer, generator=generator))
    workflow.add_node("evaluate_answer", partial(evaluate_answer, evaluator=evaluator))
    workflow.add_node("rewrite_query", partial(rewrite_query, rewriter=rewriter))
    workflow.add_node("insufficient_answer", insufficient_answer)

    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges(
        "analyze_query",
        _route_after_analysis,
        {
            "direct": "direct_generate",
            "naive": "naive_retrieve",
            "advanced": "advanced_retrieve",
            "graph": "graph_retrieve",
        },
    )
    for node in ("naive_retrieve", "advanced_retrieve", "graph_retrieve"):
        workflow.add_edge(node, "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_answer")
    workflow.add_conditional_edges(
        "evaluate_answer",
        _route_after_evaluation,
        {"passed": END, "rewrite": "rewrite_query", "insufficient": "insufficient_answer"},
    )
    workflow.add_edge("rewrite_query", "analyze_query")
    workflow.add_edge("direct_generate", END)
    workflow.add_edge("insufficient_answer", END)
    return workflow.compile()


def _route_after_analysis(
    state: AgentState,
) -> Literal["direct", "naive", "advanced", "graph"]:
    routes = {
        RetrievalStrategy.NONE: "direct",
        RetrievalStrategy.NAIVE: "naive",
        RetrievalStrategy.ADVANCED: "advanced",
        RetrievalStrategy.GRAPH: "graph",
    }
    return routes[state["retrieval_strategy"]]


def _route_after_evaluation(
    state: AgentState,
) -> Literal["passed", "rewrite", "insufficient"]:
    if state["answer_passed"]:
        return "passed"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite"
    return "insufficient"
