"""
LLM backends for the agent loop.

- :mod:`agribound.agent.backends.base` -- the provider-neutral
  :class:`~agribound.agent.backends.base.LLMBackend` protocol and data types.
- :mod:`agribound.agent.backends.anthropic_backend` -- the Anthropic Messages
  API backend (``pip install "agribound[agent]"``); its ``base_url`` option
  also reaches Anthropic-compatible local servers.

Importing this package does not import any provider SDK.
"""

from __future__ import annotations

from typing import Any

from agribound.agent.backends.base import (
    LLMBackend,
    ModelTurn,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

BACKENDS: dict[str, str] = {
    "anthropic": "agribound.agent.backends.anthropic_backend:AnthropicBackend",
}
"""Backend name -> ``"module:Class"``."""


def get_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Instantiate a registered backend by name.

    Parameters
    ----------
    name : str
        Backend name (see :data:`BACKENDS`).
    **kwargs
        Passed to the backend constructor.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    import importlib

    key = str(name).lower().strip()
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown agent backend {name!r}. Available: {sorted(BACKENDS)}; or pass an object "
            "implementing agribound.agent.backends.base.LLMBackend."
        )
    module_name, cls_name = BACKENDS[key].split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


__all__ = [
    "BACKENDS",
    "LLMBackend",
    "ModelTurn",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "get_backend",
]
