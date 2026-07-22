"""Answer generation contracts and grounded implementation."""

from rag_agent_platform.generation.base import (
    AnswerGenerator,
    FallbackSynthesizer,
    GroundedFallbackResult,
)
from rag_agent_platform.generation.fallback import GroundedFallbackSynthesizer
from rag_agent_platform.generation.service import GroundedAnswerGenerator
from rag_agent_platform.responses import INSUFFICIENT_ANSWER

__all__ = [
    "AnswerGenerator",
    "FallbackSynthesizer",
    "GroundedAnswerGenerator",
    "GroundedFallbackResult",
    "GroundedFallbackSynthesizer",
    "INSUFFICIENT_ANSWER",
]
