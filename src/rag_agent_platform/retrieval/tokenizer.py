"""Tokenization helpers for Chinese and Latin sparse retrieval."""

import re

import jieba

_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Return lowercase Chinese/Latin/number tokens without punctuation."""
    normalized = text.strip().lower()
    if not normalized:
        return []
    tokens: list[str] = []
    for segment in _SEGMENT_PATTERN.findall(normalized):
        if _CHINESE_PATTERN.fullmatch(segment):
            tokens.extend(token.strip() for token in jieba.lcut(segment) if token.strip())
        else:
            tokens.append(segment)
    return tokens
