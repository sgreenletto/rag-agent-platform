from rag_agent_platform.retrieval.tokenizer import tokenize


def test_tokenizer_handles_chinese_english_numbers_and_case() -> None:
    tokens = tokenize("员工年假 PASSWORD 2026！")

    assert "员工" in tokens
    assert {"年", "假"}.issubset(tokens)
    assert "password" in tokens
    assert "2026" in tokens


def test_tokenizer_preserves_technical_identifiers_and_versions() -> None:
    tokens = tokenize("LangChain_v0.3 API-2")

    assert tokens == ["langchain_v0.3", "api-2"]


def test_tokenizer_removes_punctuation_and_standalone_underscore() -> None:
    assert tokenize("，！？ _ ...") == []


def test_tokenizer_returns_empty_for_whitespace() -> None:
    assert tokenize("   ") == []
