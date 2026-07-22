"""Answer evaluation contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from rag_agent_platform.models import RetrievedChunk


class EvaluationDecision(StrEnum):
    """Next workflow action selected from answer and evidence quality."""

    PASS = "pass"
    REGENERATE = "regenerate"
    REWRITE_RETRIEVE = "rewrite_retrieve"
    CLARIFY = "clarify"
    REFUSE = "refuse"


@dataclass(slots=True)
class EvaluationResult:
    """Structured decision produced by an answer evaluator."""

    passed: bool
    reason: str
    suggested_query: str | None = None
    decision: EvaluationDecision | None = None
    relevance_score: float = 0.0
    groundedness_score: float = 0.0
    completeness_score: float = 0.0
    citation_quality_score: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Keep the old positional contract while exposing an explicit decision."""
        if self.decision is None:
            self.decision = (
                EvaluationDecision.PASS if self.passed else EvaluationDecision.REWRITE_RETRIEVE
            )
        elif not isinstance(self.decision, EvaluationDecision):
            self.decision = EvaluationDecision(self.decision)
        self.passed = self.decision is EvaluationDecision.PASS
        for field_name in (
            "relevance_score",
            "groundedness_score",
            "completeness_score",
            "citation_quality_score",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        if self.passed and not any(
            (
                self.relevance_score,
                self.groundedness_score,
                self.completeness_score,
                self.citation_quality_score,
            )
        ):
            self.relevance_score = 1.0
            self.groundedness_score = 1.0
            self.completeness_score = 1.0
            self.citation_quality_score = 1.0


class AnswerEvaluator(ABC):
    """Evaluate an answer against its query and evidence."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> EvaluationResult:
        """Return scores and an explicit bounded-workflow decision."""
        raise NotImplementedError
