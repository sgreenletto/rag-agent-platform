"""Answer generation contract."""

from abc import ABC, abstractmethod

from rag_agent_platform.models import Citation, RetrievedChunk


class AnswerGenerator(ABC):
    """Generate an answer and its structured citations."""

    @abstractmethod
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[Citation]]:
        """Generate a grounded answer from normalized chunks."""
        raise NotImplementedError
