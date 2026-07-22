"""Minimal provider-neutral chat model initialization."""

from __future__ import annotations

import json
import re
import socket
import ssl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable
from urllib import error, request

from rag_agent_platform.config import Settings


@runtime_checkable
class ChatModel(Protocol):
    """Small synchronous chat boundary used by agent components."""

    def invoke(self, prompt: str) -> str:
        """Return one text completion for a fully assembled prompt."""
        ...


class LLMRequestError(RuntimeError):
    """A non-retryable endpoint, authentication, configuration or response error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        original_exception_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.original_exception_type = original_exception_type


class LLMTransportError(RuntimeError):
    """A retryable transport error that remained after bounded retry attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        original_exception_type: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.original_exception_type = original_exception_type
        self.status_code = status_code


_TRANSPORT_EVENTS: ContextVar[list[str] | None] = ContextVar(
    "rag_agent_transport_events", default=None
)
_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
_NON_RETRYABLE_HTTP_STATUSES = {400, 401, 403, 404}
_SECRET_PATTERN = re.compile(r"(?i)(?:bearer\s+|sk-)[A-Za-z0-9._-]+")


@contextmanager
def capture_transport_events() -> Iterator[None]:
    """Collect transport events in the current Agent invocation context."""
    token = _TRANSPORT_EVENTS.set([])
    try:
        yield
    finally:
        _TRANSPORT_EVENTS.reset(token)


def consume_transport_events() -> list[str]:
    """Drain transport events produced since the current node began."""
    events = _TRANSPORT_EVENTS.get()
    if events is None:
        return []
    drained = list(events)
    events.clear()
    return drained


class OpenAICompatibleChatModel:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        max_transport_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        urlopen_func: Callable[..., Any] | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("LLM_MODEL is required for the configured LLM provider")
        if not api_key.strip():
            raise ValueError("LLM_API_KEY is required for the configured LLM provider")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if max_transport_retries < 0:
            raise ValueError("max_transport_retries must not be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        self._model = model.strip()
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_transport_retries = max_transport_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._urlopen = urlopen_func or request.urlopen
        self._sleep = sleep_func

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
        body = self._request_json(http_request)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty response")
        return content.strip()

    def _request_json(self, http_request: request.Request) -> dict[str, Any]:
        attempts = self._max_transport_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                with self._urlopen(http_request, timeout=self._timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise LLMRequestError("LLM response body must be a JSON object")
                if attempt > 1:
                    _record_transport_event(f"llm_transport: success_after_retries={attempt - 1}")
                return payload
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LLMRequestError(
                    f"LLM response could not be decoded: {_safe_exception_summary(exc)}",
                    original_exception_type=type(exc).__name__,
                ) from exc
            except LLMRequestError:
                raise
            except Exception as exc:
                retryable, status_code = _classify_transport_error(exc)
                safe_summary = _safe_exception_summary(exc)
                if not retryable:
                    _record_transport_event(
                        "llm_transport: not_retried, "
                        f"status={status_code if status_code is not None else 'n/a'}, "
                        f"error={type(exc).__name__}"
                    )
                    raise LLMRequestError(
                        f"LLM request failed without retry: {safe_summary}",
                        status_code=status_code,
                        original_exception_type=type(exc).__name__,
                    ) from exc
                if attempt >= attempts:
                    _record_transport_event(
                        f"llm_transport: failed_after_attempts={attempt}, "
                        f"error={type(exc).__name__}"
                    )
                    raise LLMTransportError(
                        f"LLM request failed after {attempt} attempts: {safe_summary}",
                        attempts=attempt,
                        original_exception_type=type(exc).__name__,
                        status_code=status_code,
                    ) from exc
                retry_number = attempt
                delay = self._retry_backoff_seconds * (2 ** (retry_number - 1))
                _record_transport_event(
                    f"llm_transport_retry: retry={retry_number}/{self._max_transport_retries}, "
                    f"backoff={delay:g}s, error={type(exc).__name__}, "
                    f"status={status_code if status_code is not None else 'n/a'}"
                )
                self._sleep(delay)
        raise AssertionError("bounded LLM request loop exited unexpectedly")


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


def _record_transport_event(event: str) -> None:
    events = _TRANSPORT_EVENTS.get()
    if events is not None:
        events.append(event)


def _classify_transport_error(exc: Exception) -> tuple[bool, int | None]:
    if isinstance(exc, error.HTTPError):
        if exc.code in _RETRYABLE_HTTP_STATUSES:
            return True, exc.code
        if exc.code in _NON_RETRYABLE_HTTP_STATUSES:
            return False, exc.code
        return False, exc.code
    if isinstance(exc, error.URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            retryable, status = _classify_transport_error(reason)
            return retryable, status
        lowered = str(reason).lower()
        return _looks_transient(lowered), None
    if isinstance(
        exc,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.timeout,
            ssl.SSLEOFError,
        ),
    ):
        return True, None
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror == 10054 or getattr(exc, "errno", None) == 10054:
            return True, None
        return _looks_transient(str(exc).lower()), None
    return False, None


def _looks_transient(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "10054",
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "unexpected eof",
            "eof occurred",
            "temporarily unavailable",
        )
    )


def _safe_exception_summary(exc: BaseException) -> str:
    summary = f"{type(exc).__name__}: {exc}"
    summary = _SECRET_PATTERN.sub("[redacted]", summary)
    return summary[:300]
