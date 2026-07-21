import json
from io import BytesIO
from typing import Any

import pytest

from rag_agent_platform.config import Settings
from rag_agent_platform.embeddings import (
    HashEmbeddingModel,
    OpenAICompatibleEmbeddingModel,
    build_embedding_model,
)


class FakeHTTPResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return BytesIO(self._body).read()


def test_build_embedding_model_defaults_to_hash() -> None:
    settings = Settings(_env_file=None, embedding_provider="", hash_embedding_dimensions=32)

    model = build_embedding_model(settings)

    assert isinstance(model, HashEmbeddingModel)
    assert model.identity.provider == "hash"
    assert model.identity.dimensions == 32


def test_build_siliconflow_embedding_model_uses_default_base_url() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="siliconflow",
        embedding_model="BAAI/bge-m3",
        embedding_api_key="test-key",
    )

    model = build_embedding_model(settings)

    assert isinstance(model, OpenAICompatibleEmbeddingModel)
    assert model.identity.provider == "siliconflow"
    assert model.identity.model == "BAAI/bge-m3"


def test_openai_compatible_embedding_requires_credentials() -> None:
    settings = Settings(_env_file=None, embedding_provider="openai_compatible")

    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        build_embedding_model(settings)


def test_openai_compatible_embedding_parses_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(http_request: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        captured["authorization"] = http_request.headers["Authorization"]
        return FakeHTTPResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )

    monkeypatch.setattr(
        "rag_agent_platform.embeddings.openai_compatible.request.urlopen",
        fake_urlopen,
    )
    model = OpenAICompatibleEmbeddingModel(
        provider="siliconflow",
        model="BAAI/bge-m3",
        api_key="test-key",
        base_url="https://api.siliconflow.cn/v1",
    )

    vectors = model.embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert model.identity.dimensions == 2
    assert captured["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert captured["body"] == {"model": "BAAI/bge-m3", "input": ["first", "second"]}
    assert captured["authorization"] == "Bearer test-key"
