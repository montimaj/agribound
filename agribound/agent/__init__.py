"""
Optional agent layer: plan Agribound runs from a natural-language request.

Install with ``pip install "agribound[agent]"`` (``anthropic`` and ``mcp``).

The agent works at a low level of autonomy. The model investigates with
read-only tools and proposes one configuration; a human approves or denies
the exact plan (hash-bound, single-use approval); at most one approved plan
runs per session, and the session ends after the run. There is no automatic
re-run or re-tuning step. See :func:`agribound.agent.agent.agent`.

Usage
-----
>>> import agribound
>>> result = agribound.agent(                       # doctest: +SKIP
...     "Delineate fields in this AOI for 2024 with a label-free approach",
...     study_area="fields.geojson",
...     gee_project="my-gee-project",
...     dry_run=True,
... )
>>> print(result.report)                              # doctest: +SKIP

``agribound.agent(...)`` and ``agribound.agent.agent(...)`` are the same call
(the package module is callable).

Components
----------
- :mod:`agribound.agent.tools` -- provider-neutral typed tools.
- :mod:`agribound.agent.plans` / :mod:`agribound.agent.gate` -- plans and the
  confirmation gate.
- :mod:`agribound.agent.session` -- JSON transcript.
- :mod:`agribound.agent.backends` -- LLM backends (Anthropic Messages API,
  including Anthropic-compatible local servers via ``base_url``).
- :mod:`agribound.agent.mcp_server` -- the same tools over MCP
  (``agribound mcp serve``).

Importing this package imports neither ``anthropic``, ``mcp`` nor
``pydantic``; they are loaded when a session or server starts.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from agribound.agent.agent import STATUSES, AgentResult, agent

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AgentSession": ("agribound.agent.session", "AgentSession"),
    "AgentToolError": ("agribound.agent.errors", "AgentToolError"),
    "AnthropicBackend": ("agribound.agent.backends.anthropic_backend", "AnthropicBackend"),
    "ConfirmationGate": ("agribound.agent.gate", "ConfirmationGate"),
    "LLMBackend": ("agribound.agent.backends.base", "LLMBackend"),
    "Plan": ("agribound.agent.plans", "Plan"),
    "TOOL_SPECS": ("agribound.agent.tools", "TOOL_SPECS"),
    "ToolContext": ("agribound.agent.tools", "ToolContext"),
    "ToolRegistry": ("agribound.agent.tools", "ToolRegistry"),
    "build_server": ("agribound.agent.mcp_server", "build_server"),
    "deny_all": ("agribound.agent.gate", "deny_all"),
    "prompt_confirm": ("agribound.agent.gate", "prompt_confirm"),
}

__all__ = ["STATUSES", "AgentResult", "agent", *sorted(_LAZY_ATTRS)]


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'agribound.agent' has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


class _CallableModule(types.ModuleType):
    """Module type that makes ``agribound.agent(...)`` call :func:`agent`."""

    def __call__(self, *args: Any, **kwargs: Any) -> AgentResult:
        return agent(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
