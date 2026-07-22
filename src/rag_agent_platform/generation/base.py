"""Answer generation contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_agent_platform.models import Citation, RetrievalStrategy, RetrievedChunk


@dataclass(frozen=True, slots=True)
class GroundedFallbackResult:
    """One bounded fallback answer and the exact evidence identities it used."""

    answer: str
    citations: list[Citation]
    sufficient: bool
    selected_sentences: list[str] = field(default_factory=list)
    reason: str = ""


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


@runtime_checkable
class FallbackSynthesizer(Protocol):
    """Synthesize a bounded evidence-only answer when free generation is unavailable."""

    def synthesize(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        retrieval_strategy: RetrievalStrategy,
    ) -> GroundedFallbackResult:
        """Return a grounded result without inventing or freely rewriting evidence."""
        ...
