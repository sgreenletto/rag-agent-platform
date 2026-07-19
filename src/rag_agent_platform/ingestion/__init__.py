"""Ingestion contracts and Mock implementation."""

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.ingestion.mock import MockIngestionPipeline

__all__ = ["IngestionPipeline", "MockIngestionPipeline"]
