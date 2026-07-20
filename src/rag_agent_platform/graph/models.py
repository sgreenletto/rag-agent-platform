"""Graph-internal data models.

These are private to the graph module. Public models used across the
platform (RetrievedChunk, ChildChunk, etc.) live in
``rag_agent_platform.models``.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Triplet:
    """A single (subject, predicate, object) triple extracted from text.

    Attributes:
        subject: Head entity text.
        predicate: Relation / property text.
        object: Tail entity text.
        source_chunk_id: The ``ChildChunk.chunk_id`` this triple was
            extracted from.
        confidence: Value in ``[0.0, 1.0]`` indicating extraction
            confidence.
        metadata: Free-form provenance dict (e.g. sentence index,
            extractor name, rule id).
    """

    subject: str
    predicate: str
    object: str
    source_chunk_id: str
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("Triplet.subject must not be empty")
        if not self.predicate.strip():
            raise ValueError("Triplet.predicate must not be empty")
        if not self.object.strip():
            raise ValueError("Triplet.object must not be empty")
        # source_chunk_id may be empty when extracted from raw text
        # (e.g. during query analysis).
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Triplet.confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical dedup key: (subject_lower, predicate_lower, object_lower)."""
        return (
            self.subject.strip().lower(),
            self.predicate.strip().lower(),
            self.object.strip().lower(),
        )
