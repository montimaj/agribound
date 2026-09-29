"""
Anthropic Messages API backend (manual tool-use loop).

Requests
--------
First-party API (no custom base URL): ``client.beta.messages.create`` with

- ``model`` (default ``"claude-opus-5"``, overridable by argument or the
  ``AGRIBOUND_AGENT_MODEL`` environment variable),
- ``max_tokens=16000``, ``thinking={"type": "adaptive"}``,
  ``output_config={"effort": "high"}``, ``tool_choice={"type": "auto"}``,
- the system prompt as one text block with ``cache_control`` (ephemeral),
- server-side refusal fallbacks: ``betas=["server-side-fallback-2026-07-01"]``
  and ``fallbacks="default"`` (both typed parameters of
  ``client.beta.messages.create`` in anthropic 1.8.0).

Custom ``base_url`` (e.g. Ollama >= 0.14 or vLLM Anthropic-compatible
``/v1/messages`` endpoints): ``client.messages.create`` without betas,
fallbacks, ``cache_control`` or ``tool_choice`` (these servers do not all
support them; ``auto`` is the API default anyway). ``thinking`` and ``effort``
are sent only when given explicitly. The options actually used are recorded
in :meth:`AnthropicBackend.info` and therefore in the session transcript.

No sampling parameters (``temperature``, ``top_p``, ``top_k``) are sent;
anthropic 1.x removed them from ``messages.create``.

Responses
---------
When a response contains ``fallback`` blocks (one per model that declined
mid-output), :meth:`AnthropicBackend.to_turn` returns only the ``tool_use``
blocks *after* the last ``fallback`` block as tool calls, and
:meth:`AnthropicBackend.assistant_message` omits the ``thinking``,
``redacted_thinking`` and ``tool_use`` blocks (and unpaired server-tool
blocks or unknown block types) that precede that boundary when the turn is
echoed back (:func:`echoable_content`); a WARNING is logged whenever a block
is omitted. Text blocks, paired server-tool blocks, the ``fallback`` blocks
and everything after the boundary are echoed unchanged. ``usage.iterations``
(per-attempt usage) is recorded, and a ``fallback_message`` entry marks a
response served by a fallback model.

The two sources available when this was written disagree about pre-boundary
thinking blocks:

- the claude-api skill (bundled with Claude Code 2.1.282, read 2026-09-27;
  ``shared/model-migration.md``, "Echoing fallback turns back") says to omit
  ``thinking``, ``redacted_thinking`` and ``tool_use`` blocks before the final
  ``fallback`` block; this module follows it;
- the ``BetaFallbackBlockParam`` docstring of anthropic 1.8.0 says to echo the
  assistant turn back verbatim with the ``fallback`` block in its original
  position, and that the server validates thinking runs on both sides of it.

Both agree that the declined model's ``tool_use`` blocks are not run and not
echoed without results. The rule has not been checked against the live API.
The same skill page says that for non-streaming requests (the only kind this
backend sends) a mid-output decline omits the declined partial entirely, so
the pre-boundary branch is not expected to be reached in practice.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from agribound.agent.backends.base import ModelTurn, ToolCall, ToolDefinition, ToolResult
from agribound.agent.errors import AgentDependencyError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
MODEL_ENV_VAR = "AGRIBOUND_AGENT_MODEL"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "high"
FALLBACK_BETA_DEFAULT = "server-side-fallback-2026-07-01"
"""Beta header for ``fallbacks="default"``."""
FALLBACK_BETA_ARRAY = "server-side-fallback-2026-06-01"
"""Beta header for the array form ``fallbacks=[{"model": ...}]``."""

_FIRST_PARTY_HOST = "api.anthropic.com"
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class _Auto:
    def __repr__(self) -> str:
        return "AUTO"


AUTO: Any = _Auto()
"""Sentinel: choose the option from the endpoint (first-party vs custom base URL)."""


def is_first_party(base_url: str | None) -> bool:
    """True if *base_url* is empty or points at ``api.anthropic.com``."""
    if not base_url:
        return True
    return (urlparse(str(base_url)).hostname or "").lower() == _FIRST_PARTY_HOST


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return str(value)


def _block_type(block: Any) -> Any:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _last_fallback_index(content: Sequence[Any]) -> int:
    """Index of the last ``fallback`` block in *content*, or -1."""
    for i in range(len(content) - 1, -1, -1):
        if _block_type(content[i]) == "fallback":
            return i
    return -1


def echoable_content(content: Sequence[Any]) -> list[Any]:
    """Content of a response as it may be sent back in the conversation history.

    Implements the rule for echoing a turn after a mid-output model fallback:
    before the last ``fallback`` block, keep ``text`` blocks, the ``fallback``
    blocks themselves (ignored audit markers), and server-tool blocks whose
    ``server_tool_use`` has a matching result before the boundary; drop
    ``thinking``, ``redacted_thinking``, ``tool_use``, unpaired server-tool
    blocks and unknown block types. Everything from the last ``fallback``
    block on is kept. Without a ``fallback`` block, *content* is returned
    unchanged. See the module docstring for the conflicting documentation of
    pre-boundary thinking blocks; a WARNING is logged when blocks are omitted.
    """
    content = list(content)
    boundary = _last_fallback_index(content)
    if boundary < 0:
        return content
    before = content[:boundary]
    result_ids = {
        _block_attr(b, "tool_use_id")
        for b in before
        if str(_block_type(b) or "").endswith("_tool_result")
    }
    paired = {
        _block_attr(b, "id")
        for b in before
        if _block_type(b) == "server_tool_use" and _block_attr(b, "id") in result_ids
    }

    def keep(block: Any) -> bool:
        kind = _block_type(block)
        if kind in ("text", "fallback"):
            return True
        if kind == "server_tool_use":
            return _block_attr(block, "id") in paired
        if str(kind or "").endswith("_tool_result"):
            return _block_attr(block, "tool_use_id") in paired
        return False  # thinking, redacted_thinking, tool_use, unknown block types

    kept = [b for b in before if keep(b)]
    omitted = [str(_block_type(b)) for b in before if not keep(b)]
    if omitted:
        logger.warning(
            "Echoing a turn served after a mid-output model fallback: omitting %d block(s) "
            "before the fallback boundary (%s). The documentation of this rule is ambiguous "
            "(see agribound.agent.backends.anthropic_backend).",
            len(omitted),
            ", ".join(omitted),
        )
    return kept + content[boundary:]


class AnthropicBackend:
    """:class:`~agribound.agent.backends.base.LLMBackend` for the Anthropic Messages API.

    Parameters
    ----------
    model : str or None
        Model ID. Default: ``$AGRIBOUND_AGENT_MODEL`` or ``"claude-opus-5"``.
    base_url : str or None
        Custom endpoint (Anthropic-compatible local server). *None* uses the
        SDK's resolution (``ANTHROPIC_BASE_URL``, profile, then
        ``https://api.anthropic.com``).
    api_key : str or None
        API key; *None* lets the SDK resolve credentials. Local servers need
        a placeholder key (e.g. ``"ollama"``).
    client : object or None
        Pre-built client (anything with ``messages.create`` and
        ``beta.messages.create``); used by tests.
    max_tokens : int
        Output token limit per response (default 16000).
    effort, thinking, fallbacks, cache_system_prompt
        ``AUTO`` (default) enables ``"high"`` / ``{"type": "adaptive"}`` /
        ``"default"`` / *True* for the first-party API and disables them for
        a custom base URL. *None* (or *False*) disables; any other value is
        sent as given. Non-default models may not accept these options (for
        example ``claude-haiku-4-5`` does not support adaptive thinking).
    timeout, max_retries
        Passed to :class:`anthropic.Anthropic` when *client* is not given.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: Any = AUTO,
        thinking: Any = AUTO,
        fallbacks: Any = AUTO,
        cache_system_prompt: Any = AUTO,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
        self.sdk_version: str | None = None
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise AgentDependencyError(
                    "The Anthropic backend needs the 'anthropic' package: "
                    'pip install "agribound[agent]"'
                ) from exc
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            if timeout is not None:
                kwargs["timeout"] = timeout
            if max_retries is not None:
                kwargs["max_retries"] = max_retries
            client = anthropic.Anthropic(**kwargs)
            self.sdk_version = getattr(anthropic, "__version__", None)
        self._client = client
        resolved = getattr(client, "base_url", None) or base_url
        self.base_url = str(resolved) if resolved else None
        self.first_party = is_first_party(self.base_url)
        self.max_tokens = int(max_tokens)

        def pick(value: Any, first_party_default: Any) -> Any:
            if value is AUTO:
                return first_party_default if self.first_party else None
            return value or None

        self.effort = pick(effort, DEFAULT_EFFORT)
        self.thinking = pick(thinking, {"type": "adaptive"})
        self.fallbacks = pick(fallbacks, "default")
        self.cache_system_prompt = bool(pick(cache_system_prompt, True))
        self.tool_choice = {"type": "auto"} if self.first_party else None
        if not self.first_party:
            logger.info(
                "Custom Anthropic-compatible endpoint %s: fallbacks, prompt caching and "
                "tool_choice are not sent%s",
                self.base_url,
                "" if (self.thinking or self.effort) else "; neither are thinking and effort",
            )

    # -- LLMBackend ------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "first_party": self.first_party,
            "sdk": "anthropic",
            "sdk_version": self.sdk_version,
            "request_options": {
                "endpoint": "beta.messages.create" if self.fallbacks else "messages.create",
                "max_tokens": self.max_tokens,
                "thinking": self.thinking,
                "effort": self.effort,
                "fallbacks": self.fallbacks,
                "betas": self._betas(),
                "tool_choice": self.tool_choice,
                "cache_system_prompt": self.cache_system_prompt,
            },
        }

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, turn: ModelTurn) -> dict[str, Any]:
        # All content (thinking, text, tool_use, fallback blocks) is echoed, except the
        # blocks before a mid-output fallback boundary that must not be (echoable_content).
        return {"role": "assistant", "content": echoable_content(turn.raw_content or [])}

    def tool_results_message(self, results: Sequence[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                    "is_error": bool(r.is_error),
                }
                for r in results
            ],
        }

    def _betas(self) -> list[str] | None:
        if not self.fallbacks:
            return None
        return [FALLBACK_BETA_DEFAULT if self.fallbacks == "default" else FALLBACK_BETA_ARRAY]

    def request_kwargs(
        self,
        *,
        system: str,
        tools: Sequence[ToolDefinition],
        messages: Sequence[Any],
    ) -> dict[str, Any]:
        """Keyword arguments of the ``messages.create`` call (exposed for tests)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": list(messages),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ],
        }
        if self.cache_system_prompt:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            kwargs["system"] = system
        if self.tool_choice is not None:
            kwargs["tool_choice"] = dict(self.tool_choice)
        if self.thinking:
            kwargs["thinking"] = dict(self.thinking)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            kwargs["betas"] = self._betas()
            kwargs["fallbacks"] = self.fallbacks
        return kwargs

    def complete(
        self,
        *,
        system: str,
        tools: Sequence[ToolDefinition],
        messages: Sequence[Any],
    ) -> ModelTurn:
        kwargs = self.request_kwargs(system=system, tools=tools, messages=messages)
        if "fallbacks" in kwargs:
            response = self._client.beta.messages.create(**kwargs)
        else:
            response = self._client.messages.create(**kwargs)
        return self.to_turn(response)

    @staticmethod
    def to_turn(response: Any) -> ModelTurn:
        """Convert an SDK ``Message``/``BetaMessage`` into a :class:`ModelTurn`.

        ``text`` joins every text block. ``tool_calls`` holds only the
        ``tool_use`` blocks after the last ``fallback`` block (a declined
        model's tool calls are not run; see the module docstring).
        """
        content = list(getattr(response, "content", None) or [])
        boundary = _last_fallback_index(content)
        texts, calls, fallbacks = [], [], []
        for i, block in enumerate(content):
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(block.text)
            elif kind == "tool_use" and i > boundary:
                calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
            elif kind == "fallback":
                trigger = getattr(block, "trigger", None)
                fallbacks.append(
                    {
                        "from": getattr(getattr(block, "from_", None), "model", None),
                        "to": getattr(getattr(block, "to", None), "model", None),
                        "trigger_category": getattr(trigger, "category", None),
                    }
                )
        usage_obj = getattr(response, "usage", None)
        usage = {}
        for key in _USAGE_KEYS:
            value = getattr(usage_obj, key, None)
            if value is not None:
                usage[key] = value
        iterations = []
        for entry in getattr(usage_obj, "iterations", None) or []:
            item: dict[str, Any] = {"type": getattr(entry, "type", None)}
            model = getattr(entry, "model", None)
            if model is not None:
                item["model"] = str(model)
            for key in _USAGE_KEYS:
                value = getattr(entry, key, None)
                if value is not None:
                    item[key] = value
            iterations.append(item)
        return ModelTurn(
            stop_reason=str(getattr(response, "stop_reason", None)),
            text="\n".join(texts),
            tool_calls=calls,
            raw_content=content,
            usage=usage,
            stop_details=_dump(getattr(response, "stop_details", None)),
            model=getattr(response, "model", None),
            request_id=getattr(response, "_request_id", None),
            fallbacks=fallbacks,
            served_by_fallback=any(it["type"] == "fallback_message" for it in iterations),
            iterations=iterations,
        )


__all__ = [
    "AUTO",
    "AnthropicBackend",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "FALLBACK_BETA_ARRAY",
    "FALLBACK_BETA_DEFAULT",
    "MODEL_ENV_VAR",
    "echoable_content",
    "is_first_party",
]
