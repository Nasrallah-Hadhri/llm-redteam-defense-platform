"""Ollama HTTP adapter implementing the TargetAdapter interface."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from redteam.targets.base import TargetAdapter

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds


class OllamaAdapter(TargetAdapter):
    """Sends prompts to a locally-running Ollama instance via its HTTP API.

    Retries up to _MAX_RETRIES times on transient network/server errors
    with exponential backoff. Raises RuntimeError on permanent failures.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    def send(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Send *prompt* to the Ollama model and return its response text.

        Args:
            prompt: The new user-turn text.
            conversation_history: Optional prior turns in
                [{"role": "user"|"assistant", "content": str}] format.
                Prepended before the new prompt in the messages list.
        """
        messages: list[dict[str, Any]] = list(conversation_history or [])
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
        }

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{self._base_url}/api/chat",
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
                    "Ollama transient error (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(
                    f"Ollama HTTP error for model '{self._model_name}': {exc}"
                ) from exc

        raise RuntimeError(
            f"Ollama unreachable after {_MAX_RETRIES} attempts: {last_exc}"
        )
