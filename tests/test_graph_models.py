"""Tests for graph-internal Triplet data model."""

import pytest

from rag_agent_platform.graph.models import Triplet

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_triplet_construction() -> None:
    t = Triplet(
        subject="人事部",
        predicate="负责",
        object="请假制度",
        source_chunk_id="doc-1::chunk-1",
        confidence=0.9,
        metadata={"rule": "cn-responsible"},
    )
    assert t.subject == "人事部"
    assert t.predicate == "负责"
    assert t.object == "请假制度"
    assert t.source_chunk_id == "doc-1::chunk-1"
    assert t.confidence == 0.9
    assert t.metadata == {"rule": "cn-responsible"}


def test_triplet_defaults() -> None:
    t = Triplet(subject="A", predicate="是", object="B", source_chunk_id="c1")
    assert t.confidence == 1.0
    assert t.metadata == {}


def test_triplet_empty_source_chunk_id_allowed() -> None:
    """Empty source_chunk_id is valid — used during query analysis."""
    t = Triplet(subject="A", predicate="是", object="B", source_chunk_id="")
    assert t.source_chunk_id == ""


def test_triplet_confidence_boundary() -> None:
    Triplet(subject="A", predicate="是", object="B", source_chunk_id="c1", confidence=0.0)
    Triplet(subject="A", predicate="是", object="B", source_chunk_id="c1", confidence=1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "predicate", "object"),
    [
        ("", "是", "B"),
        ("   ", "是", "B"),
        ("A", "", "B"),
        ("A", "   ", "B"),
        ("A", "是", ""),
        ("A", "是", "   "),
    ],
)
def test_triplet_rejects_empty_fields(subject: str, predicate: str, object: str) -> None:
    with pytest.raises(ValueError):
        Triplet(
            subject=subject,
            predicate=predicate,
            object=object,
            source_chunk_id="c1",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, -100, 100])
def test_triplet_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Triplet(
            subject="A",
            predicate="是",
            object="B",
            source_chunk_id="c1",
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------


def test_triplet_key_case_insensitive() -> None:
    t1 = Triplet(subject="人事部", predicate="负责", object="请假制度", source_chunk_id="c1")
    t2 = Triplet(subject="人事部", predicate="负责", object="请假制度", source_chunk_id="c2")
    assert t1.key == t2.key


def test_triplet_key_whitespace_normalized() -> None:
    t1 = Triplet(subject=" 人事部 ", predicate="负责", object="请假制度", source_chunk_id="c1")
    t2 = Triplet(subject="人事部", predicate="负责", object="请假制度", source_chunk_id="c2")
    assert t1.key == t2.key


def test_triplet_key_different_predicates() -> None:
    t1 = Triplet(subject="人事部", predicate="负责", object="请假制度", source_chunk_id="c1")
    t2 = Triplet(subject="人事部", predicate="管理", object="请假制度", source_chunk_id="c2")
    assert t1.key != t2.key
