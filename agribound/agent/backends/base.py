"""
Provider-neutral interface between the agent loop and an LLM.

The loop in :mod:`agribound.agent.agent` talks to a model only through an
object that satisfies :class:`LLMBackend`. The backend owns the wire format
of the conversation: the loop keeps an opaque list of messages that the
backend itself builds with :meth:`LLMBackend.user_message`,
:meth:`LLMBackend.assistant_message` and :meth:`LLMBackend.tool_results_message`,
and receives each model response as a neutral :class:`ModelTurn`.

To plug in another provider, implement these five methods plus the ``name``
and ``model`` attributes; no Agribound code needs to change. The bundled
implementation is :class:`agribound.agent.backends.anthropic_backend.AnthropicBackend`,
which also reaches Anthropic-compatible local servers through ``base_url``.

This module imports only the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Stop reasons the loop understands. Backends map their provider's values onto these.
STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_PAUSE_TURN = "pause_turn"
STOP_REFUSAL = "refusal"
STOP_MAX_TOKENS = "max_tokens"


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as offered to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    input: Any


@dataclass(frozen=True)
class ToolResult:
    """The result returned to the model for one :class:`ToolCall`."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelTurn:
    """One model response, in provider-neutral form.

    Attributes
    ----------
    stop_reason : str
        Provider stop reason (``"end_turn"``, ``"tool_use"``, ``"pause_turn"``,
        ``"refusal"``, ``"max_tokens"`` or another provider value).
    text : str
        Concatenated text blocks.
    tool_calls : list of ToolCall
        Requested tool invocations, in order.
    raw_content : object
        The provider's content. :meth:`LLMBackend.assistant_message` echoes
        it back (the Anthropic backend drops only the blocks that the API
        says must not be echoed after a mid-output model fallback).
    usage : dict
        Token counts reported by the provider.
    stop_details : dict or None
        Provider detail for a refusal.
    model : str or None
        The model that produced the response.
    request_id : str or None
        Provider request ID.
    fallbacks : list of dict
        Server-side model switches that happened within this response.
    served_by_fallback : bool
        True if the provider reports that a fallback model produced the
        response (for Anthropic: a ``fallback_message`` entry in
        ``usage.iterations``, which also covers turns without a
        ``fallback`` block).
    iterations : list of dict
        Per-attempt token usage reported by the provider (for Anthropic:
        ``usage.iterations``); empty when not reported. The top-level
        ``usage`` then covers only the attempt that produced the response.
    """

    stop_reason: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    stop_details: dict[str, Any] | None = None
    model: str | None = None
    request_id: str | None = None
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    served_by_fallback: bool = False
    iterations: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class LLMBackend(Protocol):
    """What the agent loop needs from an LLM provider."""

    name: str
    model: str

    def info(self) -> dict[str, Any]:
        """Describe the backend for the transcript (model, endpoint, request options)."""
        ...

    def user_message(self, text: str) -> Any:
        """Build a user message carrying *text*."""
        ...

    def assistant_message(self, turn: ModelTurn) -> Any:
        """Build the assistant message that echoes *turn* (all content, not only text)."""
        ...

    def tool_results_message(self, results: Sequence[ToolResult]) -> Any:
        """Build ONE user message carrying all tool results of one assistant turn."""
        ...

    def complete(
        self,
        *,
        system: str,
        tools: Sequence[ToolDefinition],
        messages: Sequence[Any],
    ) -> ModelTurn:
        """Send the conversation and return the model's next turn."""
        ...


__all__ = [
    "LLMBackend",
    "ModelTurn",
    "STOP_END_TURN",
    "STOP_MAX_TOKENS",
    "STOP_PAUSE_TURN",
    "STOP_REFUSAL",
    "STOP_TOOL_USE",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
]
