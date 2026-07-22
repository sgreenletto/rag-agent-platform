"""Ingestion contracts and Mock implementation."""

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.ingestion.coordinated import CoordinatedIngestionPipeline
from rag_agent_platform.ingestion.mock import MockIngestionPipeline
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline

__all__ = [
    "CoordinatedIngestionPipeline",
    "IngestionPipeline",
    "MockIngestionPipeline",
    "RealIngestionPipeline",
]
