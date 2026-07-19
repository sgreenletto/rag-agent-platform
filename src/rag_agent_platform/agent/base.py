"""Agent orchestration contract."""

from abc import ABC, abstractmethod

from rag_agent_platform.models import AgentResult


class AgentService(ABC):
    """Run one complete question-answering workflow."""

    @abstractmethod
    def invoke(
        self,
        query: str,
        document_ids: list[str] | None = None,
        mode: str = "agent",
    ) -> AgentResult:
        """Return a stable AgentResult for the requested mode."""
        raise NotImplementedError
