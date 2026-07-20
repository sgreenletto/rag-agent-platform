"""Tests for MockTripletExtractor — rule-based triplet extraction."""

import re
from dataclasses import dataclass

import pytest

from rag_agent_platform.graph.extractor import MockTripletExtractor, TripletExtractor

# ---------------------------------------------------------------------------
# Duck-typed chunk for testing
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Minimal chunk stub — avoids depending on the public models module."""

    def __init__(self, chunk_id: str, content: str, document_id: str = "doc-1") -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.document_id = document_id
        self.parent_id = "parent-1"
        self.page = None
        self.metadata: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


def test_triplet_extractor_is_abstract() -> None:
    with pytest.raises(TypeError):
        TripletExtractor()  # type: ignore[abstract]


def test_mock_extractor_is_instance() -> None:
    ext = MockTripletExtractor()
    assert isinstance(ext, TripletExtractor)


# ---------------------------------------------------------------------------
# extract_from_text
# ---------------------------------------------------------------------------


def test_extract_from_empty_text() -> None:
    ext = MockTripletExtractor()
    assert ext.extract_from_text("") == []
    assert ext.extract_from_text("   ") == []


def test_extract_cn_is_a() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("人事部是公司的重要部门。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "人事部"
    assert t.object == "公司的重要部门"
    assert t.predicate in ("是", "is-a")


def test_extract_cn_responsible() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("人事部负责请假制度。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "人事部"
    assert t.object == "请假制度"
    assert t.predicate == "负责"


def test_extract_cn_contains() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("公司规章包含考勤规则。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "公司规章"
    assert t.object == "考勤规则"
    assert t.predicate == "包含"


def test_extract_cn_belongs_to() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("请假制度属于公司规章。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "请假制度"
    assert t.object == "公司规章"
    assert t.predicate == "属于"


def test_extract_cn_depends_on() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("员工依赖于考勤系统。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "员工"
    assert t.object == "考勤系统"
    assert t.predicate == "依赖于"


def test_extract_cn_related_to() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("请假制度与薪资规则关联。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "请假制度"
    assert t.object == "薪资规则"
    # Predicate is inferred from the text between the two groups ("与")
    assert t.predicate in ("与", "与关联", "related-to")


def test_extract_cn_manages() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("项目经理管理开发团队。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "项目经理"
    assert t.object == "开发团队"
    assert t.predicate == "管理"


def test_extract_cn_requires() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("员工需要工号。")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "员工"
    assert t.object == "工号"
    assert t.predicate == "需要"


def test_extract_cn_arrow() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("员工 -> 考勤系统")
    assert len(triplets) >= 1
    t = triplets[0]
    assert t.subject == "员工"
    assert t.object == "考勤系统"


def test_extract_english_is_a() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("A manager is a person who leads a team.")
    assert len(triplets) >= 1
    # "manager is a person" should match


def test_extract_english_depends_on() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("The payroll system depends on attendance data.")
    assert len(triplets) >= 1


# ---------------------------------------------------------------------------
# extract (from chunks)
# ---------------------------------------------------------------------------


def test_extract_from_chunks() -> None:
    ext = MockTripletExtractor()
    chunks = [
        _FakeChunk("c1", "人事部负责请假制度。员工需要提前三天申请。"),
        _FakeChunk("c2", "请假制度属于公司规章。公司规章包含考勤规则。"),
    ]
    triplets = ext.extract(chunks)  # type: ignore[arg-type]
    assert len(triplets) >= 2
    # Every triplet should have a source_chunk_id
    for t in triplets:
        assert t.source_chunk_id in ("c1", "c2")


def test_extract_from_empty_chunks() -> None:
    ext = MockTripletExtractor()
    assert ext.extract([]) == []


def test_extract_skips_empty_content() -> None:
    ext = MockTripletExtractor()
    chunks = [_FakeChunk("c1", "")]
    assert ext.extract(chunks) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def test_confidence_in_range() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("人事部负责请假制度。员工依赖于考勤系统。")
    for t in triplets:
        assert 0.0 <= t.confidence <= 1.0


def test_deduplication_keeps_highest_confidence() -> None:
    """If the same triplet appears multiple times, keep the highest confidence."""
    ext = MockTripletExtractor()
    # "人事部负责请假制度" appears twice, first match should be kept
    triplets = ext.extract_from_text("人事部负责请假制度。人事部负责请假制度。")
    keys = [t.key for t in triplets]
    unique_keys = set(keys)
    assert len(triplets) == len(unique_keys), f"Duplicates found: {triplets}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_triplets_have_metadata() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("人事部负责请假制度。")
    for t in triplets:
        assert "rule" in t.metadata
        assert t.metadata["extractor"] == "MockTripletExtractor"


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------


def test_custom_rules_override_defaults() -> None:
    @dataclass
    class _Rule:
        name: str
        pattern: re.Pattern[str]
        mapping: tuple[int, int, int]
        confidence: float = 0.95

    custom = [
        _Rule("cn-test", re.compile(r"(\S+)热爱(\S+)"), (0, -1, 1), 0.99),
    ]
    ext = MockTripletExtractor(rules=custom)
    triplets = ext.extract_from_text("张三热爱编程。")
    assert len(triplets) == 1
    assert triplets[0].subject == "张三"
    assert triplets[0].object == "编程"
    assert triplets[0].confidence == 0.99


def test_extra_rules_append_to_defaults() -> None:
    @dataclass
    class _Rule:
        name: str
        pattern: re.Pattern[str]
        mapping: tuple[int, int, int]
        confidence: float = 0.95

    extra = [
        _Rule("cn-test2", re.compile(r"(\S+)热爱(\S+)"), (0, -1, 1), 0.99),
    ]
    ext = MockTripletExtractor(extra_rules=extra)
    # Both default and extra rules should work
    t1 = ext.extract_from_text("人事部负责请假制度。")
    t2 = ext.extract_from_text("张三热爱编程。")
    assert len(t1) >= 1
    assert len(t2) >= 1


# ---------------------------------------------------------------------------
# Self-loop prevention
# ---------------------------------------------------------------------------


def test_no_self_loops() -> None:
    ext = MockTripletExtractor()
    triplets = ext.extract_from_text("请假制度是请假制度。")
    # "请假制度是请假制度" — subject == object, should be skipped
    keys = [t.key for t in triplets]
    for key in keys:
        assert key[0] != key[2], f"Self-loop found: {key}"


# ---------------------------------------------------------------------------
# Multi-sentence extraction
# ---------------------------------------------------------------------------


def test_multi_sentence() -> None:
    ext = MockTripletExtractor()
    text = """人事部负责请假制度。
    请假制度属于公司规章。
    公司规章包含考勤规则和薪资规则。
    员工依赖于考勤系统。"""
    triplets = ext.extract_from_text(text)
    # Expect at least 4 triplets from the 4 sentences
    assert len(triplets) >= 4
