"""Triplet extraction contracts and rule-based Mock implementation.

Provides a deterministic, zero-dependency extractor suitable for
automated testing and offline development — no LLM / API key needed.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from rag_agent_platform.graph.models import Triplet

# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class TripletExtractor(ABC):
    """Extract (subject, predicate, object) triples from child chunks."""

    @abstractmethod
    def extract(self, chunks: list[object]) -> list[Triplet]:
        """Extract triples from a batch of ChildChunk objects.

        Args:
            chunks: ``list[ChildChunk]`` whose ``.content`` fields will
                be processed.

        Returns:
            Deduplicated list of extracted ``Triplet`` objects.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_from_text(self, text: str) -> list[Triplet]:
        """Extract triples from a plain string (used for query analysis)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    """Internal rule descriptor.

    Attributes:
        name: Short label for the rule (e.g. ``"cn-is-a"``).
        pattern: Compiled regex.
        mapping: 3-tuple ``(subject_idx, predicate_idx, object_idx)``
            indicating which regex groups supply each slot.  A value of
            ``-1`` means the slot is filled by the literal text between
            capture groups (used for two-group patterns like
            ``"A负责B"``).
        confidence: Base confidence for matches from this rule.
    """

    name: str
    pattern: re.Pattern[str]
    mapping: tuple[int, int, int]
    confidence: float = 0.85


# ---------------------------------------------------------------------------
# Built-in rule set
# ---------------------------------------------------------------------------

# fmt: off
_DEFAULT_RULES: list[_Rule] = [
    # ── Chinese relational patterns ────────────────────────────
    _Rule("cn-is-a",
          re.compile(r"(\S{1,20})是(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-possessive",
          re.compile(r"(\S{1,20})的(\S{1,20})是(\S{1,20})"),
          (0, 1, 2), 0.80),
    _Rule("cn-responsible",
          re.compile(r"(\S{1,20})负责(\S{1,20})"),
          (0, -1, 1), 0.90),
    _Rule("cn-contains",
          re.compile(r"(\S{1,20})包含(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-belongs-to",
          re.compile(r"(\S{1,20})属于(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-depends-on",
          re.compile(r"(\S{1,20})依赖(?:于)?(\S{1,20})"),
          (0, -1, 1), 0.90),
    _Rule("cn-related-to",
          re.compile(r"(\S{1,20})与(\S{1,20})关联"),
          (0, -1, 1), 0.80),
    _Rule("cn-manages",
          re.compile(r"(\S{1,20})管理(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-part-of",
          re.compile(r"(\S{1,20})是(\S{1,20})的一部分"),
          (0, -1, 1), 0.85),
    _Rule("cn-causes",
          re.compile(r"(\S{1,20})导致(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-requires",
          re.compile(r"(\S{1,20})需要(\S{1,20})"),
          (0, -1, 1), 0.80),
    _Rule("cn-owns",
          re.compile(r"(\S{1,20})拥有(\S{1,20})"),
          (0, -1, 1), 0.80),
    _Rule("cn-produces",
          re.compile(r"(\S{1,20})生产(\S{1,20})"),
          (0, -1, 1), 0.85),
    _Rule("cn-provides",
          re.compile(r"(\S{1,20})提供(\S{1,20})"),
          (0, -1, 1), 0.80),

    # ── Arrow / list patterns ──────────────────────────────────
    _Rule("arrow",
          re.compile(r"(\S{1,30})\s*[-–>]+\s*(\S{1,30})"),
          (0, -1, 1), 0.70),

    # ── English patterns ───────────────────────────────────────
    _Rule("en-is-a",
          re.compile(r"(\w+(?:\s\w+){0,4})\s+is\s+(?:a|an)\s+(\w+(?:\s\w+){0,4})",
                     re.IGNORECASE),
          (0, -1, 1), 0.80),
    _Rule("en-has",
          re.compile(r"(\w+(?:\s\w+){0,4})\s+has\s+(\w+(?:\s\w+){0,4})",
                     re.IGNORECASE),
          (0, -1, 1), 0.75),
    _Rule("en-belongs-to",
          re.compile(r"(\w+(?:\s\w+){0,4})\s+belongs?\s+to\s+(\w+(?:\s\w+){0,4})",
                     re.IGNORECASE),
          (0, -1, 1), 0.80),
    _Rule("en-depends-on",
          re.compile(r"(\w+(?:\s\w+){0,4})\s+depends?\s+on\s+(\w+(?:\s\w+){0,4})",
                     re.IGNORECASE),
          (0, -1, 1), 0.85),
]
# fmt: on


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockTripletExtractor(TripletExtractor):
    """Deterministic rule-based triplet extractor.

    Uses a configurable set of regex patterns to extract (subject,
    predicate, object) triples from plain text.  No network, LLM, or
    API key is required — suitable for CI, offline development, and
    deterministic testing.

    Parameters:
        rules: Optional custom rule list.  When *None* the built-in
            default rules are used.
        extra_rules: Additional rules appended to the defaults (or to
            *rules* when both are supplied).
    """

    _SENTENCE_SEP: ClassVar[re.Pattern[str]] = re.compile(r"(?<=[。！？；\n\.!\?;])\s*")

    def __init__(
        self,
        rules: list[_Rule] | None = None,
        extra_rules: list[_Rule] | None = None,
    ) -> None:
        base: list[_Rule] = list(rules) if rules is not None else list(_DEFAULT_RULES)
        if extra_rules:
            base.extend(extra_rules)
        self._rules: list[_Rule] = base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, chunks: list[object]) -> list[Triplet]:
        """Extract triples from every chunk's ``.content``.

        Args:
            chunks: ``list[ChildChunk]``.

        Returns:
            Deduplicated ``list[Triplet]`` sorted by confidence
            descending.
        """
        if not chunks:
            return []

        all_triplets: list[Triplet] = []
        for chunk in chunks:
            # Duck-typing: avoid hard import of ChildChunk
            content: str = getattr(chunk, "content", "")
            chunk_id: str = getattr(chunk, "chunk_id", "")
            if not content or not chunk_id:
                continue
            for triplet in self.extract_from_text(content):
                triplet.source_chunk_id = chunk_id
                all_triplets.append(triplet)

        return self._deduplicate(all_triplets)

    def extract_from_text(self, text: str) -> list[Triplet]:
        """Extract triples from a plain string.

        The returned triples have ``source_chunk_id=""`` — callers
        should set it when the text originates from a chunk.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        triplets: list[Triplet] = []
        for sentence in sentences:
            for rule in self._rules:
                for triplet in self._apply_rule(rule, sentence):
                    triplets.append(triplet)
        return self._deduplicate(triplets)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentences, keeping non-empty parts."""
        parts = MockTripletExtractor._SENTENCE_SEP.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _apply_rule(rule: _Rule, sentence: str) -> list[Triplet]:
        """Return all triplets matched by *rule* in *sentence*."""
        results: list[Triplet] = []
        for match in rule.pattern.finditer(sentence):
            groups = match.groups()
            mapping = rule.mapping

            def _slot(idx: int, _groups: tuple = groups) -> str:
                if idx == -1:
                    return ""  # filled later by _infer_predicate
                if idx < len(_groups):
                    return _groups[idx].strip()
                return ""

            subj = _clean_entity(_slot(mapping[0]))
            pred = _clean_entity(_slot(mapping[1]))
            obj = _clean_entity(_slot(mapping[2]))

            if not pred:
                pred = _infer_predicate(rule, sentence, match)

            if not subj or not pred or not obj:
                continue
            if subj == obj:
                continue  # reject self-loops

            results.append(
                Triplet(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    source_chunk_id="",
                    confidence=rule.confidence,
                    metadata={"rule": rule.name, "extractor": "MockTripletExtractor"},
                )
            )
        return results

    @staticmethod
    def _deduplicate(triplets: list[Triplet]) -> list[Triplet]:
        """Merge triples with the same (subj, pred, obj) key.

        Keeps the highest confidence and collects source_chunk_ids.
        """
        seen: dict[tuple[str, str, str], Triplet] = {}
        order: list[tuple[str, str, str]] = []
        for t in triplets:
            k = t.key
            if k in seen:
                existing = seen[k]
                if t.confidence > existing.confidence:
                    existing.confidence = t.confidence
                # Merge chunk sources
                chunk_ids: set[str] = {
                    cid for cid in (existing.source_chunk_id, t.source_chunk_id) if cid
                }
                if chunk_ids:
                    existing.source_chunk_id = ";".join(sorted(chunk_ids))
                # Merge metadata
                existing.metadata = {**existing.metadata, **t.metadata}
            else:
                seen[k] = t
                order.append(k)
        # Sort by confidence descending, then by key
        result = [seen[k] for k in order]
        result.sort(key=lambda x: (-x.confidence, x.key))
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Punctuation stripped from entity boundaries after regex matching.
_TRAILING_PUNCT = re.compile(r"[。，！？；：、\.!\?;,:，\s]+$")
_LEADING_PUNCT = re.compile(r"^[。，！？；：、\.!\?;,:，\s]+")


def _clean_entity(text: str) -> str:
    """Strip common CJK + ASCII punctuation from entity boundaries."""
    text = _LEADING_PUNCT.sub("", text)
    text = _TRAILING_PUNCT.sub("", text)
    return text.strip()


def _infer_predicate(rule: _Rule, sentence: str, match: re.Match[str]) -> str:
    """Recover a predicate string when the rule mapping uses ``-1``.

    For two-group patterns like ``"A负责B"`` the predicate is the
    literal text between the first and second capture groups.
    """
    groups = match.groups()
    if len(groups) == 2:
        start = match.end(1)
        end = match.start(2)
        pred_text = sentence[start:end].strip()
        if pred_text:
            return pred_text
    # Fallback: rule name without leading "cn-" / "en-" prefix
    name = rule.name
    for prefix in ("cn-", "en-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-", " ")
