"""Conservative grounded-answer evaluation."""

import re

from rag_agent_platform.evaluation.base import AnswerEvaluator, EvaluationResult
from rag_agent_platform.llm import ChatModel, extract_json_object
from rag_agent_platform.models import RetrievedChunk


class GroundedAnswerEvaluator(AnswerEvaluator):
    """Evaluate completeness, grounding, citations and hallucination risk."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        if not chunks:
            return EvaluationResult(False, "检索结果为空，无法形成有依据的回答", query)
        markers = {int(value) for value in re.findall(r"\[(\d+)]", answer)}
        valid_markers = set(range(1, len(chunks) + 1))
        if not markers:
            return EvaluationResult(False, "回答缺少知识库引用", query)
        if not markers.issubset(valid_markers):
            return EvaluationResult(False, "回答包含不存在的引用编号", query)
        if self._chat_model is None:
            return EvaluationResult(True, "本地保守校验通过：回答包含有效证据引用")

        context = "\n".join(f"[{i}] {chunk.content}" for i, chunk in enumerate(chunks, 1))
        prompt = (
            "请保守评估回答。检查 completeness、groundedness、citation_quality、hallucination。"
            "关键数字或事实无上下文依据、缺少问题部分、引用不匹配时 passed 必须为 false。"
            '只输出 JSON：{"passed": bool, "reason": str, '
            '"suggested_query": str|null}。\n'
            f"问题：{query}\n回答：{answer}\n上下文：\n{context}"
        )
        try:
            payload = extract_json_object(self._chat_model.invoke(prompt))
            passed = payload.get("passed")
            reason = payload.get("reason")
            suggested_query = payload.get("suggested_query")
            if not isinstance(passed, bool) or not isinstance(reason, str):
                raise TypeError("evaluation response has invalid fields")
            if suggested_query is not None and not isinstance(suggested_query, str):
                raise TypeError("suggested_query must be a string or null")
            return EvaluationResult(passed, reason, suggested_query)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return EvaluationResult(
                False,
                f"模型评估失败，按保守策略拒绝回答：{type(exc).__name__}: {exc}",
                query,
            )
