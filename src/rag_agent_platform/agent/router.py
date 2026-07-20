"""Query analysis and bounded query rewriting services."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_agent_platform.llm import ChatModel, extract_json_object
from rag_agent_platform.models import QueryType, RetrievedChunk


@dataclass(slots=True)
class QueryAnalysis:
    query_type: QueryType
    needs_retrieval: bool
    reason: str
    warning: str | None = None


@runtime_checkable
class QueryAnalyzer(Protocol):
    def analyze(self, query: str) -> QueryAnalysis:
        """Classify one query without selecting a manual retrieval mode."""
        ...


class StructuredQueryAnalyzer:
    """Prefer structured LLM classification and fall back to explicit rules."""

    _CHAT_PHRASES = ("你好", "您好", "你是谁", "你能做什么", "hello", "hi")
    _RELATION_WORDS = ("关系", "关联", "分别负责", "上下游", "影响路径", "联系", "依赖")
    _COMPLEX_WORDS = ("结合", "比较", "综合", "分别", "根据以上", "对比", "多个", "同时")

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

    def analyze(self, query: str) -> QueryAnalysis:
        if self._chat_model is None:
            return self._rule_analysis(query)
        prompt = (
            "将问题分类为 simple、complex、relation、chat。relation 表示实体关系、上下游或影响"
            "路径；complex 表示多条件、比较或综合；chat 表示普通对话。只输出 JSON："
            '{"query_type": str, "needs_retrieval": bool, "reason": str}。\n'
            f"问题：{query}"
        )
        try:
            payload = extract_json_object(self._chat_model.invoke(prompt))
            query_type = QueryType(str(payload["query_type"]).lower())
            needs_retrieval = payload["needs_retrieval"]
            reason = payload["reason"]
            if not isinstance(needs_retrieval, bool) or not isinstance(reason, str):
                raise TypeError("classification response has invalid fields")
            if query_type is QueryType.CHAT:
                needs_retrieval = False
            return QueryAnalysis(query_type, needs_retrieval, reason)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            fallback = self._rule_analysis(query)
            fallback.warning = f"LLM 分类失败，已使用规则兜底：{type(exc).__name__}: {exc}"
            return fallback

    def _rule_analysis(self, query: str) -> QueryAnalysis:
        normalized = query.strip()
        lowered = normalized.lower()
        if any(phrase == lowered or phrase in lowered for phrase in self._CHAT_PHRASES):
            return QueryAnalysis(QueryType.CHAT, False, "规则识别为普通对话")
        if any(word in normalized for word in self._RELATION_WORDS):
            return QueryAnalysis(QueryType.RELATION, True, "规则命中关系类关键词")
        condition_count = sum(word in normalized for word in self._COMPLEX_WORDS)
        if len(normalized) >= 60 or condition_count > 0 or normalized.count("，") >= 2:
            return QueryAnalysis(QueryType.COMPLEX, True, "规则识别为长问题或多条件问题")
        return QueryAnalysis(QueryType.SIMPLE, True, "规则默认识别为简单知识问题")


@runtime_checkable
class QueryRewriter(Protocol):
    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        evaluation_reason: str,
        chunks: list[RetrievedChunk],
        suggested_query: str | None,
    ) -> str:
        """Return a materially changed retrieval query."""
        ...


class BoundedQueryRewriter:
    """Use an optional LLM and always retain a deterministic non-identical fallback."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        evaluation_reason: str,
        chunks: list[RetrievedChunk],
        suggested_query: str | None,
    ) -> str:
        suggestion = (suggested_query or "").strip()
        if suggestion and suggestion != current_query.strip():
            return suggestion
        if self._chat_model is not None:
            evidence = "\n".join(chunk.content[:200] for chunk in chunks[:3]) or "无"
            prompt = (
                "请改写检索查询，补全实体和多条件关键词，不要回答问题，也不要原样重复当前查询。"
                "只输出一行新查询。\n"
                f"原始问题：{original_query}\n当前查询：{current_query}\n"
                f"失败原因：{evaluation_reason}\n已有证据：{evidence}"
            )
            try:
                rewritten = self._chat_model.invoke(prompt).strip()
                if rewritten and rewritten != current_query.strip():
                    return rewritten
            except RuntimeError:
                pass
        reason_terms = evaluation_reason.replace("：", " ").replace("，", " ").strip()
        suffix = reason_terms[:40] or "补充完整条件与明确实体"
        rewritten = f"{original_query.strip()}；补充检索：{suffix}"
        if rewritten == current_query.strip():
            rewritten = f"{original_query.strip()}；补充检索第 {len(chunks) + 1} 组相关依据"
        return rewritten
