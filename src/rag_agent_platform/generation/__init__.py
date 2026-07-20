"""Answer generation contracts and grounded implementation."""

from rag_agent_platform.generation.base import AnswerGenerator
from rag_agent_platform.generation.service import INSUFFICIENT_ANSWER, GroundedAnswerGenerator

__all__ = ["AnswerGenerator", "GroundedAnswerGenerator", "INSUFFICIENT_ANSWER"]
