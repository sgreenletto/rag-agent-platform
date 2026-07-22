"""Grounded answer generation with an optional chat model."""

from rag_agent_platform.generation.base import AnswerGenerator, FallbackSynthesizer
from rag_agent_platform.generation.fallback import GroundedFallbackSynthesizer
from rag_agent_platform.llm import ChatModel
from rag_agent_platform.models import Citation, RetrievalStrategy, RetrievedChunk
from rag_agent_platform.responses import INSUFFICIENT_ANSWER


class GroundedAnswerGenerator(AnswerGenerator):
    """Generate only from retrieved evidence and preserve citation identity."""

    def __init__(
        self,
        chat_model: ChatModel | None = None,
        *,
        fallback_synthesizer: FallbackSynthesizer | None = None,
        max_context_chars: int = 8000,
    ) -> None:
        if max_context_chars < 1000:
            raise ValueError("max_context_chars must be at least 1000")
        self._chat_model = chat_model
        self._fallback_synthesizer = fallback_synthesizer or GroundedFallbackSynthesizer()
        self._max_context_chars = max_context_chars

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[Citation]]:
        if not chunks:
            return INSUFFICIENT_ANSWER, []
        citations = [
            Citation(index=index, source=chunk.source, page=chunk.page, chunk_id=chunk.chunk_id)
            for index, chunk in enumerate(chunks, start=1)
        ]
        if self._chat_model is None:
            fallback = self._fallback_synthesizer.synthesize(
                query,
                chunks,
                retrieval_strategy=_strategy_from_chunks(chunks),
            )
            answer = fallback.answer if fallback.sufficient else INSUFFICIENT_ANSWER
            citations = fallback.citations
        else:
            answer = self._chat_model.invoke(self._build_prompt(query, chunks, strict=False))
        return answer, citations

    def regenerate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        previous_answer: str,
        unsupported_claims: list[str],
    ) -> tuple[str, list[Citation]]:
        """Regenerate from the same evidence with explicit defect feedback."""
        if not chunks:
            return INSUFFICIENT_ANSWER, []
        citations = [
            Citation(index=index, source=chunk.source, page=chunk.page, chunk_id=chunk.chunk_id)
            for index, chunk in enumerate(chunks, start=1)
        ]
        if self._chat_model is None:
            fallback = self._fallback_synthesizer.synthesize(
                query,
                chunks,
                retrieval_strategy=_strategy_from_chunks(chunks),
            )
            return (
                fallback.answer if fallback.sufficient else INSUFFICIENT_ANSWER,
                fallback.citations,
            )
        correction = (
            "\n\n上一次回答包含以下不合规内容，必须删除，不得通过改写措辞保留：\n- "
            + "\n- ".join(unsupported_claims)
            if unsupported_claims
            else "\n\n上一次回答未通过依据或引用检查，请只保留上下文明确支持的事实。"
        )
        prompt = self._build_prompt(query, chunks, strict=True)
        prompt += f"{correction}\n上一次回答：{previous_answer}"
        return self._chat_model.invoke(prompt), citations

    def generate_chat(self, query: str) -> str:
        """Answer ordinary conversation without pretending to query the knowledge base."""
        if self._chat_model is None:
            return (
                "你好！我是文档问答助手，可以帮助检索已入库文档、比较多份材料，"
                "或分析实体之间的关系。请告诉我你想了解什么。"
            )
        return self._chat_model.invoke(
            f"你是一个中文文档问答助手。简洁回答普通对话，不要声称已经检索知识库。\n用户：{query}"
        )

    def _build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        strict: bool,
    ) -> str:
        contexts = []
        remaining_chars = self._max_context_chars
        for index, chunk in enumerate(chunks, start=1):
            if remaining_chars <= 0:
                break
            page = "未知" if chunk.page is None else str(chunk.page)
            content = _bounded_chunk_context(query, chunk.content, remaining_chars)
            if not content:
                continue
            remaining_chars -= len(content)
            contexts.append(
                f"[来源 {index}]\nsource: {chunk.source}\npage: {page}\n"
                f"retrieval_method: {chunk.retrieval_method}\ncontent: {content}"
            )
        return (
            "你是严格依据知识库上下文回答问题的助手。只能使用下方上下文明确支持的事实，"
            "不得用常识补全企业制度，也不得为追求完整而增加用户未问的子问题。\n"
            "供应商能力不可传递：信息技术供应商不等于所有信息技术产品的销售商，数据库维护"
            "服务也不等于数据库软件销售。只有上下文明确说明时，才能声称企业具有软件销售、"
            "代理、授权、许可证或资质。材料未说明时，应写“文档未说明”或“现有材料无法确认”。\n"
            "不得引入问题和上下文都没有的品牌、型号、Oracle、MySQL 企业版、许可证或授权"
            "代理等示例。用户询问制度流程时，只回答制度明确要求的步骤。\n"
            "每个关键结论使用最直接支持它的 [1]、[2] 等来源编号；同一事实不要堆叠重复引用，"
            "禁止伪造来源、页码或文档。回答应简洁、确定、可验证，语言跟随用户输入。"
            + (
                "这是重新生成，必须删除上一次回答中的所有无依据推断并重新核对每个引用。"
                if strict
                else ""
            )
            + "\n\n"
            f"原始问题：{query}\n\n" + "\n\n".join(contexts)
        )


def _strategy_from_chunks(chunks: list[RetrievedChunk]) -> RetrievalStrategy:
    if any("graph" in chunk.retrieval_method.lower() for chunk in chunks):
        return RetrievalStrategy.GRAPH
    if any("advanced" in chunk.retrieval_method.lower() for chunk in chunks):
        return RetrievalStrategy.ADVANCED
    return RetrievalStrategy.NAIVE


def _bounded_chunk_context(query: str, content: str, remaining_chars: int) -> str:
    if len(content) <= remaining_chars:
        return content
    limit = max(0, remaining_chars)
    if limit == 0:
        return ""
    normalized_query = query.replace("打钱", "付款").replace("送来", "交付")
    query_terms = {
        term
        for term in (
            "采购",
            "数据库",
            "审核",
            "审批",
            "验收",
            "付款",
            "交付",
            "合同",
            "发票",
            "预算",
            "紧急",
            "关系",
            "联系",
        )
        if term in normalized_query
    }
    sentences = [
        sentence.strip()
        for sentence in content.replace("！", "。！").replace("？", "。？").split("。")
        if sentence.strip()
    ]
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-sum(term in item[1] for term in query_terms), item[0]),
    )
    selected: list[tuple[int, str]] = []
    used = 0
    for position, sentence in ranked:
        rendered = f"{sentence}。"
        if used + len(rendered) > limit:
            continue
        selected.append((position, rendered))
        used += len(rendered)
        if used >= limit:
            break
    selected.sort(key=lambda item: item[0])
    return "".join(sentence for _, sentence in selected)
