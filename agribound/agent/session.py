"""
Agent session transcript.

:class:`AgentSession` records one agent session as JSON
(``<workdir>/agent_session_<session_id>.json``): the request, backend and
model ID, package/SDK versions, every model turn (stop reason, text, tool
calls, token usage, duration), every tool call (validated arguments, error or
result summary, duration), the plans, the confirmation gate's approvals and
denials (who, how, when), the executions, and the final status. The file is
rewritten after every turn and once more when an approved plan starts to run
(with the approval and a ``"running"`` execution record), so an interrupted
session, including one that crashes during a long pipeline run, still leaves
a record.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_SCHEMA_VERSION = "1"

#: Longest serialised tool result kept verbatim in the transcript.
RESULT_SUMMARY_CHARS = 4000

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def summarize_result(value: Any, limit: int = RESULT_SUMMARY_CHARS) -> str:
    """Serialise *value* as JSON, truncated to *limit* characters (marked when truncated)."""
    from agribound.provenance import to_jsonable

    text = value if isinstance(value, str) else json.dumps(to_jsonable(value), sort_keys=True)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} characters in total]"


class AgentSession:
    """Collects and writes the transcript of one agent session.

    Parameters
    ----------
    request : str
        The user's natural-language request.
    workdir : str or Path
        Session directory; the transcript is written there.
    session_id : str or None
        Identifier (default :func:`agribound._repro.new_run_id`).
    backend_info : dict or None
        Backend description (name, model, base URL, request options).
    options : dict or None
        Session options (dry run, limits, study area, ...).
    """

    def __init__(
        self,
        request: str,
        *,
        workdir: str | Path,
        session_id: str | None = None,
        backend_info: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        from agribound._repro import collect_versions, new_run_id

        self.request = str(request)
        self.workdir = Path(workdir)
        self.session_id = session_id or new_run_id()
        self.path = self.workdir / f"agent_session_{self.session_id}.json"
        self.backend_info = dict(backend_info or {})
        self.options = dict(options or {})
        self.versions = collect_versions(extra=("anthropic", "mcp", "mcp-types", "pydantic"))
        self.started_utc = _utc_now()
        self.finished_utc: str | None = None
        self._t0 = time.perf_counter()
        self.turns: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] = []
        self.executions: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.gate_state: dict[str, Any] | None = None
        self.status = "running"
        self.final_text = ""
        self.report = ""
        self.error: str | None = None
        self.stop_details: dict[str, Any] | None = None

    # -- recording -------------------------------------------------------------

    def record_turn(self, turn: Any, *, duration_s: float) -> dict[str, Any]:
        """Record one model response (a :class:`~agribound.agent.backends.base.ModelTurn`)."""
        entry = {
            "index": len(self.turns),
            "stop_reason": turn.stop_reason,
            "model": turn.model,
            "request_id": turn.request_id,
            "text": turn.text,
            "tool_calls": [{"id": c.id, "name": c.name} for c in turn.tool_calls],
            "usage": dict(turn.usage or {}),
            "iterations": [dict(it) for it in getattr(turn, "iterations", None) or []],
            "stop_details": turn.stop_details,
            "fallbacks": list(turn.fallbacks or []),
            "served_by_fallback": bool(getattr(turn, "served_by_fallback", False)),
            "duration_s": round(float(duration_s), 3),
        }
        self.turns.append(entry)
        return entry

    def record_tool_call(
        self,
        *,
        turn_index: int,
        call_id: str,
        name: str,
        outcome: Any,
        duration_s: float,
    ) -> dict[str, Any]:
        """Record one tool call and its :class:`~agribound.agent.tools.ToolOutcome`."""
        entry = {
            "turn": turn_index,
            "id": call_id,
            "name": name,
            "arguments": outcome.arguments,
            "arguments_valid": outcome.arguments_valid,
            "is_error": not outcome.ok,
            "error": outcome.error,
            "result_summary": summarize_result(outcome.output) if outcome.ok else None,
            "duration_s": round(float(duration_s), 3),
        }
        self.tool_calls.append(entry)
        return entry

    def record_plan(self, plan: Any) -> None:
        """Record a proposed plan.

        A re-proposal of the same configuration has the same plan ID (the ID
        hashes the configuration and inputs, not the agent's text). Its entry
        is replaced by the latest proposal, which is the version the
        confirmation gate shows to the reviewer, and ``n_proposals`` counts
        how often it was proposed.
        """
        entry = plan.to_dict()
        for i, existing in enumerate(self.plans):
            if existing["plan_id"] == plan.plan_id:
                entry["n_proposals"] = int(existing.get("n_proposals", 1)) + 1
                self.plans[i] = entry
                return
        entry["n_proposals"] = 1
        self.plans.append(entry)

    def record_execution(self, execution: dict[str, Any]) -> None:
        """Record an execution attempt, or update it.

        An entry with the same ``"attempt"`` number is replaced (a
        ``"running"`` record is written when the run starts and replaced by
        the final ``"success"``/``"failed"`` record).
        """
        entry = dict(execution)
        attempt = entry.get("attempt")
        if attempt is not None:
            for i, existing in enumerate(self.executions):
                if existing.get("attempt") == attempt:
                    self.executions[i] = entry
                    return
        self.executions.append(entry)

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def set_gate(self, gate: Any) -> None:
        """Snapshot the confirmation gate's approvals, denials and counters."""
        self.gate_state = gate.to_dict() if gate is not None else None

    def finish(
        self,
        status: str,
        *,
        final_text: str = "",
        report: str = "",
        error: str | None = None,
        stop_details: dict[str, Any] | None = None,
    ) -> None:
        """Set the final status and timing."""
        self.status = status
        self.final_text = final_text
        self.report = report
        self.error = error
        self.stop_details = stop_details
        self.finished_utc = _utc_now()

    # -- output ----------------------------------------------------------------

    def usage_totals(self) -> dict[str, int]:
        """Token usage summed over all turns.

        For a turn that reports per-attempt ``iterations`` (e.g. a declined
        attempt followed by a fallback model), the iterations are summed,
        because the top-level usage then covers only the final attempt;
        otherwise the top-level usage is used.
        """
        totals = dict.fromkeys(_USAGE_KEYS, 0)
        for turn in self.turns:
            entries = turn.get("iterations") or [turn.get("usage") or {}]
            for entry in entries:
                for key in _USAGE_KEYS:
                    value = entry.get(key)
                    if isinstance(value, int):
                        totals[key] += value
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "request": self.request,
            "status": self.status,
            "error": self.error,
            "stop_details": self.stop_details,
            "backend": self.backend_info,
            "options": self.options,
            "versions": self.versions,
            "platform": platform.platform(),
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "wall_s": round(time.perf_counter() - self._t0, 3),
            "usage_totals": self.usage_totals(),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "plans": self.plans,
            "gate": self.gate_state,
            "executions": self.executions,
            "warnings": self.warnings,
            "final_text": self.final_text,
            "report": self.report,
        }

    def write(self) -> Path:
        """Write the transcript atomically and return its path."""
        from agribound.provenance import to_jsonable

        self.workdir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(to_jsonable(self.to_dict()), f, indent=2)
            f.write("\n")
        os.replace(tmp, self.path)
        return self.path


__all__ = ["AgentSession", "RESULT_SUMMARY_CHARS", "SESSION_SCHEMA_VERSION", "summarize_result"]
