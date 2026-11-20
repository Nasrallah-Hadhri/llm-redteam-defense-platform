"""Scope enforcement — ensures attack prompts are only sent to local targets.

This module is the primary safety rail preventing the platform from
accidentally or intentionally contacting third-party hosted model APIs.
It is called at Campaign initialisation and again defensively inside every
attack chain.
"""
from __future__ import annotations

from typing import Any

# Only Ollama running on localhost is an allowlisted target by default.
# To extend the allowlist (e.g. a local VM at 192.168.x.x), add its URL
# prefix here AND document the change in the project README.
ALLOWED_TARGET_PREFIXES: tuple[str, ...] = (
    "http://localhost:11434",
    "http://127.0.0.1:11434",
)


class ScopeViolationError(RuntimeError):
    """Raised when an adapter, client, or endpoint URL is outside the allowlisted targets."""


def assert_in_scope(target_or_client: Any) -> None:
    """Raise ScopeViolationError if target_or_client points at a non-allowlisted URL.

    Accepts:
        - TargetAdapter instance (has .base_url)
        - OllamaClient instance (has .base_url)
        - Judge instance (has .model_client.base_url)
        - Any object with a base_url or url attribute
        - A raw URL string

    Args:
        target_or_client: The component or URL string to inspect.

    Raises:
        ScopeViolationError: If the target/client URL does not start with
            any prefix in ALLOWED_TARGET_PREFIXES.
    """
    if isinstance(target_or_client, str):
        url = target_or_client
    elif hasattr(target_or_client, "model_client") and hasattr(
        target_or_client.model_client, "base_url"
    ):
        url = getattr(target_or_client.model_client, "base_url", "")
    else:
        url = getattr(target_or_client, "base_url", getattr(target_or_client, "url", ""))

    if not url or not any(url.startswith(prefix) for prefix in ALLOWED_TARGET_PREFIXES):
        component_name = getattr(
            target_or_client,
            "model_name",
            getattr(target_or_client, "model", type(target_or_client).__name__),
        )
        raise ScopeViolationError(
            f"Component '{component_name}' points at URL '{url}', which is not in the allowlist "
            f"{ALLOWED_TARGET_PREFIXES}. "
            "This platform is strictly restricted to local Ollama targets and clients only. "
            "External model APIs (OpenAI, Anthropic, cloud endpoints) are blocked for safety."
        )
