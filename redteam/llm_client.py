"""Lightweight LLM client wrapping Ollama's /api/chat endpoint.

Both the Attacker and the Judge use this client. Each component receives
its own OllamaClient instance so their conversation contexts are fully
independent — the judge never sees the attacker's prompt-generation history
and vice versa.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds


def get_model_digest(model_name: str, base_url: str = "http://localhost:11434") -> str:
    """Fetch the SHA-256 model digest from Ollama tags endpoint.

    Returns the digest string or 'unknown' if offline or not found.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code == 200:
            for m in resp.json().get("models", []):
                name = m.get("name", "")
                if name == model_name or name.split(":")[0] == model_name:
                    return m.get("digest", "unknown")
    except Exception:
        pass
    return "unknown"


class OllamaClient:
    """Stateless one-shot chat client for Ollama.

    Each call to .chat() constructs a fresh messages list and posts to
    /api/chat. No persistent conversation state is maintained between calls;
    callers are responsible for passing any history they want the model to
    see via *extra_messages*.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = options or {}
        self._digest: str | None = None

    @property
    def model_name(self) -> str:
        return self.model

    def get_digest(self) -> str:
        """Return the SHA-256 model digest, fetching and caching it on first call."""
        if self._digest is None:
            self._digest = get_model_digest(self.model, self.base_url)
        return self._digest

    def chat(
        self,
        system: str,
        user: str,
        extra_messages: list[dict] | None = None,
        override_options: dict[str, Any] | None = None,
    ) -> str:
        """Send a system + user message and return the model's reply.

        Args:
            system: System prompt text (sets the model's role/persona).
            user: The new user-turn text.
            extra_messages: Optional additional prior turns inserted between
                the system message and the new user turn. Used by the Attacker
                to pass a condensed failure history as assistant turns.
            override_options: Optional generation parameters (temperature, seed).

        Returns:
            The model's response text.

        Raises:
            RuntimeError: On HTTP errors or after exhausting retries.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user})

        combined_options = {**self.options, **(override_options or {})}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if combined_options:
            payload["options"] = combined_options

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_exc = exc
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "OllamaClient transient error (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(
                    f"Ollama HTTP error for model '{self.model}': {exc}"
                ) from exc

        raise RuntimeError(
            f"OllamaClient unreachable after {_MAX_RETRIES} attempts: {last_exc}"
        )
