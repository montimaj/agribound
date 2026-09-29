"""
Exceptions of the agent layer.

Every *expected* failure of an agent tool (invalid arguments, a missing file,
no network permission, a plan that was not approved) is an
:class:`AgentToolError`. The local tool-use loop returns its message to the
model as an ``is_error`` tool result, and the MCP server maps it to
``mcp.server.mcpserver.exceptions.ToolError`` so that MCP hosts show the
message instead of a generic failure.

This module imports only the standard library.
"""

from __future__ import annotations


class AgentToolError(Exception):
    """An anticipated tool failure whose message is shown to the model."""


class GateError(AgentToolError):
    """Base class of confirmation-gate refusals."""


class ExecutionDisabledError(GateError):
    """``execute_plan`` was called while execution is disabled (dry run)."""


class ExecutionDeniedError(GateError):
    """The human reviewer (or the non-interactive default) denied a plan."""


class ExecutionNotApprovedError(GateError):
    """No unused approval exists for the plan's hash."""


class ExecutionLimitError(GateError):
    """The session already ran ``max_executions`` plans."""


class PlanChangedError(GateError):
    """The plan's configuration or inputs no longer match its hash."""


class UnknownPlanError(AgentToolError):
    """No plan with the requested ID exists in this session."""


class NetworkDisabledError(AgentToolError):
    """A tool (or a plan's pipeline run) needs a remote service while network access is off."""


class AgentDependencyError(ImportError):
    """An optional dependency of the agent layer is not installed."""


__all__ = [
    "AgentDependencyError",
    "AgentToolError",
    "ExecutionDeniedError",
    "ExecutionDisabledError",
    "ExecutionLimitError",
    "ExecutionNotApprovedError",
    "GateError",
    "NetworkDisabledError",
    "PlanChangedError",
    "UnknownPlanError",
]
