"""Minimal provider-neutral chat model initialization."""

import json
from typing import Any, Protocol, runtime_checkable
from urllib import error, request

from rag_agent_platform.config import Settings


@runtime_checkable
class ChatModel(Protocol):
    """Small synchronous chat boundary used by agent components."""

    def invoke(self, prompt: str) -> str:
        """Return one text completion for a fully assembled prompt."""
        ...


class OpenAICompatibleChatModel:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model.strip():
            raise ValueError("LLM_MODEL is required for the configured LLM provider")
        if not api_key.strip():
            raise ValueError("LLM_API_KEY is required for the configured LLM provider")
        self._model = model.strip()
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def invoke(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty response")
        return content.strip()


def build_chat_model(settings: Settings) -> ChatModel | None:
    """Build the configured provider, or signal use of local grounded components."""
    provider = settings.llm_provider.strip().lower()
    if provider in {"", "none", "local", "rule"}:
        return None
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleChatModel(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or "https://api.openai.com/v1",
        )
    raise ValueError(
        f"unsupported LLM_PROVIDER '{settings.llm_provider}'; "
        "use openai-compatible or leave it empty for the local grounded fallback"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from plain text or a fenced response."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("structured response must be a JSON object")
    return value
