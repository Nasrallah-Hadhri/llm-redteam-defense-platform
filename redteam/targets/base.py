"""Abstract base class for all target model adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TargetAdapter(ABC):
    """Unified interface for sending prompts to a target LLM."""

    @abstractmethod
    def send(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Send a prompt to the target model and return its text response.

        Args:
            prompt: The new user-turn text to send.
            conversation_history: Optional list of prior turns in
                [{"role": "user"|"assistant", "content": str}] format.
                The adapter prepends these before the new prompt so the
                model sees the full conversation context.

        Returns:
            The model's response text.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the underlying model (e.g. 'llama3:8b')."""
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL of the model endpoint (used by scope_guard)."""
        ...
