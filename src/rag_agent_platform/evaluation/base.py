"""Answer evaluation contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_agent_platform.models import RetrievedChunk


@dataclass(slots=True)
class EvaluationResult:
    """Structured decision produced by an answer evaluator."""

    passed: bool
    reason: str
    suggested_query: str | None = None


class AnswerEvaluator(ABC):
    """Evaluate an answer against its query and evidence."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        """Return a pass decision and optional rewritten query."""
        raise NotImplementedError
