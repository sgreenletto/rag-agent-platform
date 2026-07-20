"""GraphRAG module — knowledge graph construction and retrieval.

Provides:
- ``TripletExtractor`` / ``MockTripletExtractor`` — entity-relation
  extraction from text (zero external API dependencies).
- ``NetworkXGraphStore`` — NetworkX DiGraph wrapper with JSON
  persistence.
- ``NetworkXGraphService`` — implements the ``GraphService`` contract
  (build / delete_document / retrieve).
- ``GraphRetriever`` — adapts a ``GraphService`` to the standard
  ``BaseRetriever`` interface.
- ``MockGraphRetriever`` — deterministic mock for contract testing.
"""

# Core internals — zero external dependencies, always safe to import.
from rag_agent_platform.graph.extractor import MockTripletExtractor, TripletExtractor
from rag_agent_platform.graph.models import Triplet
from rag_agent_platform.graph.store import NetworkXGraphStore

# The following depend on the public models package (member 2 / interfaces.md).
# They use lazy imports so graph internals remain usable for isolated testing
# before all branches are merged.
try:
    from rag_agent_platform.graph.base import GraphService
except ImportError:  # pragma: no cover — public models not yet merged
    GraphService = None  # type: ignore[assignment,misc]

try:
    from rag_agent_platform.graph.retriever import GraphRetriever
except ImportError:  # pragma: no cover
    GraphRetriever = None  # type: ignore[assignment,misc]

try:
    from rag_agent_platform.graph.service import NetworkXGraphService
except ImportError:  # pragma: no cover
    NetworkXGraphService = None  # type: ignore[assignment,misc]

try:
    from rag_agent_platform.graph.mock import MockGraphRetriever
except ImportError:  # pragma: no cover
    MockGraphRetriever = None  # type: ignore[assignment,misc]

__all__ = [
    "GraphRetriever",
    "GraphService",
    "MockGraphRetriever",
    "MockTripletExtractor",
    "NetworkXGraphService",
    "NetworkXGraphStore",
    "Triplet",
    "TripletExtractor",
]
