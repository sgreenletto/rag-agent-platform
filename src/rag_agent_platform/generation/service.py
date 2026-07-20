"""Grounded answer generation with an optional chat model."""

from rag_agent_platform.generation.base import AnswerGenerator
from rag_agent_platform.llm import ChatModel
from rag_agent_platform.models import Citation, RetrievedChunk

INSUFFICIENT_ANSWER = "当前知识库中没有找到足够可靠的依据，无法确定回答。"


class GroundedAnswerGenerator(AnswerGenerator):
    """Generate only from retrieved evidence and preserve citation identity."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model

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
            answer = self._extractive_answer(query, chunks)
        else:
            answer = self._chat_model.invoke(self._build_prompt(query, chunks))
        return answer, citations

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

    @staticmethod
    def _extractive_answer(query: str, chunks: list[RetrievedChunk]) -> str:
        evidence = "\n".join(
            f"- {chunk.content.strip()} [{index}]" for index, chunk in enumerate(chunks, start=1)
        )
        return f"针对“{query}”，当前知识库提供了以下依据：\n\n{evidence}"

    @staticmethod
    def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
        contexts = []
        for index, chunk in enumerate(chunks, start=1):
            page = "未知" if chunk.page is None else str(chunk.page)
            contexts.append(
                f"[来源 {index}]\nsource: {chunk.source}\npage: {page}\n"
                f"retrieval_method: {chunk.retrieval_method}\ncontent: {chunk.content}"
            )
        return (
            "你是严格依据知识库上下文回答问题的助手。只使用下方上下文，不得补充上下文之外的"
            "事实；依据不足时必须明确说明。关键结论使用 [1]、[2] 等来源编号，禁止伪造来源、"
            "页码或文档。回答用户的原始问题，语言跟随用户输入。\n\n"
            f"原始问题：{query}\n\n" + "\n\n".join(contexts)
        )
