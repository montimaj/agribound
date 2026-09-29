"""
The Agribound agent: a tool-use loop with a human confirmation gate.

:func:`agent` sends a natural-language request to an LLM together with the
typed tools of :mod:`agribound.agent.tools`, runs the tools the model calls,
and returns an :class:`AgentResult`. Its autonomy is deliberately low:

- the model can only *propose* a run (``propose_run``); a human approves or
  denies the exact plan (hash-bound, single-use approval;
  :mod:`agribound.agent.gate`);
- the gate enforces a hard execution limit (``max_executions``, default 1);
- the loop stops right after an execution attempt or a denial, without
  another model turn, so at most one plan runs per call and the model cannot
  re-run or re-tune anything; a follow-up run needs a new call by the human;
- ``dry_run=True`` removes ``execute_plan`` altogether and leaves a plan YAML
  for ``agribound delineate --config``.

Every model turn and tool call is recorded in a JSON transcript
(:mod:`agribound.agent.session`).

This module imports only the standard library at import time; the tools
(``pydantic``) and the backend SDK are imported when :func:`agent` runs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Session statuses reported in :attr:`AgentResult.status`.
STATUSES = (
    "completed",  # the model ended its turn (no execution happened)
    "executed",  # an approved plan ran successfully; the session stopped
    "execution_failed",  # an approved plan was started and failed; the session stopped
    "denied",  # the reviewer denied a plan; the session stopped
    "refused",  # the model refused (stop_reason "refusal")
    "max_tokens",  # a response hit max_tokens
    "max_turns",  # the turn limit was reached
    "error",  # backend or unexpected error
)


@dataclass
class AgentResult:
    """Outcome of one :func:`agent` session.

    Attributes
    ----------
    status : str
        One of :data:`STATUSES`.
    final_text : str
        The most recent non-empty text written by the model in a complete
        turn. Partial text of a refused or ``max_tokens``-truncated response
        is kept only in the transcript. After an execution attempt or a
        denial the model gets no further turn, so this is text written
        *before* the execution.
    report : str
        Deterministic summary written by Agribound (not by the model): plans,
        gate decisions, execution results and the transcript path.
    plans : list of dict
        Every plan proposed in the session (:meth:`Plan.to_dict`).
    plan_yaml_paths : list of str
        YAML files of the plans (``agribound delineate --config <file>``).
    executions : list of dict
        Execution attempts with their summaries or errors.
    transcript_path : str
        The JSON transcript.
    session_id, workdir : str
        Session identifier and directory.
    error : str or None
        Error message for ``"error"``, ``"max_tokens"`` and failed executions.
    stop_details : dict or None
        Provider detail for ``"refused"``.
    """

    status: str
    final_text: str
    report: str
    plans: list[dict[str, Any]] = field(default_factory=list)
    plan_yaml_paths: list[str] = field(default_factory=list)
    executions: list[dict[str, Any]] = field(default_factory=list)
    transcript_path: str = ""
    session_id: str = ""
    workdir: str = ""
    error: str | None = None
    stop_details: dict[str, Any] | None = None

    @property
    def executed(self) -> bool:
        """True if a plan ran successfully."""
        return self.status == "executed"


def _make_backend(backend: Any, model: str | None, base_url: str | None) -> Any:
    if isinstance(backend, str):
        from agribound.agent.backends import get_backend

        kwargs: dict[str, Any] = {"model": model}
        if base_url is not None:
            kwargs["base_url"] = base_url
        return get_backend(backend, **kwargs)
    if model is not None or base_url is not None:
        raise ValueError("model and base_url apply to named backends; configure the instance.")
    return backend


def _make_callback(confirm: Any) -> Any:
    from agribound.agent import gate

    if confirm is None:
        return gate.default_callback()
    if confirm == "prompt":
        return gate.prompt_confirm
    if confirm == "deny":
        return gate.deny_all
    if callable(confirm):
        return confirm
    raise ValueError(f"confirm must be None, 'prompt', 'deny' or a callable, got {confirm!r}")


def _render_report(session: Any, status: str, error: str | None) -> str:
    lines = [f"Agribound agent session {session.session_id}: {status}"]
    if error:
        lines.append(f"Error: {error}")
    for plan in session.plans:
        cfg = plan["config"]
        lines.append(
            f"- Plan {plan['plan_id']}: source={cfg['source']} engine={cfg['engine']} "
            f"year={cfg['year']} -> {cfg['output_path']}"
        )
        lines.append(f"  YAML: {plan['yaml_path']} (agribound delineate --config <file>)")
        for warning in plan["warnings"]:
            lines.append(f"  warning: {warning}")
    gate_state = session.gate_state or {}
    for approval in gate_state.get("approvals", []):
        lines.append(
            f"- Approved {approval['plan_id']} by {approval['approver']} via "
            f"{approval['method']} at {approval['approved_utc']}"
        )
    for denial in gate_state.get("denials", []):
        lines.append(f"- Denied {denial['plan_id']} ({denial['method']}): {denial['reason']}")
    for execution in session.executions:
        if execution.get("status") == "success":
            summary = execution.get("summary") or {}
            lines.append(
                f"- Executed {execution['plan_id']}: {summary.get('n_polygons')} polygons -> "
                f"{summary.get('output_path')} (provenance: {summary.get('provenance_path')})"
            )
        elif execution.get("status") == "running":
            lines.append(f"- Execution of {execution['plan_id']} started but did not finish")
        else:
            lines.append(f"- Execution of {execution['plan_id']} failed: {execution.get('error')}")
    if status in ("executed", "execution_failed", "denied"):
        lines.append(
            "The session stopped here. The agent does not re-run or re-tune plans; start a new "
            "request for another run."
        )
    lines.append(f"Transcript: {session.path}")
    return "\n".join(lines)


def agent(
    request: str,
    *,
    study_area: str | None = None,
    gee_project: str | None = None,
    workdir: str | Path | None = None,
    backend: Any = "anthropic",
    model: str | None = None,
    base_url: str | None = None,
    confirm: Any = None,
    dry_run: bool = False,
    max_turns: int = 20,
    max_executions: int = 1,
    reference_boundaries: str | None = None,
    allow_network: bool = True,
) -> AgentResult:
    """Plan (and, after human approval, run) an Agribound delineation from a request.

    Parameters
    ----------
    request : str
        Natural-language request, e.g. ``"Delineate fields in this AOI for
        2024 with a label-free approach"``.
    study_area : str or None
        Default study area for the tools (vector file, GEE asset ID,
        ``"bbox:..."`` or WKT).
    gee_project : str or None
        Earth Engine project used in plans and live availability checks.
    workdir : str, Path or None
        Session directory (default ``./agribound_agent/<session_id>``). Plans,
        their YAML files and outputs, the shared cache and the transcript are
        written here.
    backend : str or LLMBackend
        ``"anthropic"`` (default) or an object implementing
        :class:`~agribound.agent.backends.base.LLMBackend`.
    model : str or None
        Model ID for a named backend (default ``$AGRIBOUND_AGENT_MODEL`` or
        ``"claude-opus-5"``).
    base_url : str or None
        Anthropic-compatible endpoint for a named backend (e.g. a local
        Ollama or vLLM server).
    confirm : None, "prompt", "deny" or callable
        Confirmation callback of the gate. *None*: typed ``yes`` prompt when
        standard input is a terminal, otherwise every plan is denied.
        ``"prompt"`` always prompts (e.g. in notebooks), ``"deny"`` never
        approves, and a callable ``confirm(plan) -> bool`` decides itself.
    dry_run : bool
        Disable execution: ``execute_plan`` is not offered; plans are written
        as YAML.
    max_turns : int
        Maximum number of model responses (default 20).
    max_executions : int
        Execution limit of the confirmation gate (default 1). The loop stops
        after the first execution attempt, so at most one plan runs per call
        whatever the value; ``0`` keeps ``execute_plan`` offered but lets the
        gate refuse it without asking the reviewer.
    reference_boundaries : str or None
        Default reference layer for resolvability/evaluation tools.
    allow_network : bool
        Allow tools to contact Earth Engine, TESSERA, Source Cooperative or
        the USGS NAIP Plus ImageServer. With *False*, the read-only tools skip
        or refuse their network parts and ``execute_plan`` refuses, without
        asking the reviewer, every plan whose pipeline run needs one of these
        services. Not affected: the LLM backend, and model-weight downloads
        from Hugging Face during a run (prefetch them with ``agribound
        prefetch``).

    Returns
    -------
    AgentResult

    Raises
    ------
    ValueError
        For invalid arguments.
    ImportError
        If the backend's SDK or ``pydantic`` is not installed
        (``pip install "agribound[agent]"``).
    """
    if not str(request or "").strip():
        raise ValueError("request must be a non-empty string")
    if int(max_turns) < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    from agribound._repro import new_run_id
    from agribound.agent.backends.base import ToolResult
    from agribound.agent.errors import AgentDependencyError
    from agribound.agent.gate import ConfirmationGate
    from agribound.agent.prompts import SYSTEM_PROMPT, session_preamble
    from agribound.agent.session import AgentSession

    try:
        from agribound.agent.tools import ToolContext, ToolRegistry
    except ModuleNotFoundError as exc:
        if exc.name != "pydantic":
            raise
        raise AgentDependencyError(
            'The agent tools need pydantic: pip install "agribound[agent]"'
        ) from exc

    llm = _make_backend(backend, model, base_url)
    session_id = new_run_id()
    root = Path(workdir) if workdir is not None else Path("agribound_agent") / session_id
    gate = ConfirmationGate(_make_callback(confirm), max_executions=max_executions)
    ctx = ToolContext(
        workdir=root,
        study_area=study_area,
        gee_project=gee_project,
        reference_boundaries=reference_boundaries,
        allow_network=allow_network,
        execution_enabled=not dry_run,
        gate=gate,
    )
    session = AgentSession(
        request,
        workdir=ctx.workdir,
        session_id=session_id,
        backend_info=llm.info(),
        options={
            "study_area": study_area,
            "gee_project": gee_project,
            "reference_boundaries": reference_boundaries,
            "dry_run": dry_run,
            "max_turns": int(max_turns),
            "max_executions": int(max_executions),
            "allow_network": allow_network,
            "workdir": str(ctx.workdir),
        },
    )
    ctx.session = session
    registry = ToolRegistry(ctx)
    tools = registry.definitions()
    session.options["tools"] = [t.name for t in tools]

    messages: list[Any] = [
        llm.user_message(
            session_preamble(
                request,
                study_area=study_area,
                gee_project=gee_project,
                reference_boundaries=reference_boundaries,
                dry_run=dry_run,
                workdir=str(ctx.workdir),
                allow_network=allow_network,
            )
        )
    ]
    status, final_text, error, stop_details = "max_turns", "", None, None
    session.write()

    for _ in range(int(max_turns)):
        t0 = time.perf_counter()
        try:
            turn = llm.complete(system=SYSTEM_PROMPT, tools=tools, messages=messages)
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            status, error = "error", f"LLM request failed: {type(exc).__name__}: {exc}"
            if not session.turns:
                error += (
                    " (The first request failed: check the backend's credentials, endpoint and "
                    "network access, e.g. ANTHROPIC_API_KEY or `ant auth login` for the "
                    "Anthropic API, or base_url / --base-url for a local server.)"
                )
            break
        entry = session.record_turn(turn, duration_s=time.perf_counter() - t0)

        # Check the stop reason before using the content: a refused or truncated
        # response is partial and is neither run nor reported as the final text.
        if turn.stop_reason == "refusal":
            status, stop_details = "refused", turn.stop_details
            break
        if turn.stop_reason == "max_tokens":
            status = "max_tokens"
            error = "The model response hit max_tokens before finishing; no tools were run."
            break
        if turn.stop_reason == "model_context_window_exceeded":
            status = "error"
            error = (
                "The conversation exceeded the model's context window; no tools were run. "
                "Start a new, narrower request."
            )
            break
        if turn.text:
            final_text = turn.text
        if turn.stop_reason == "pause_turn":
            # Resend the paused assistant content unchanged so the model continues.
            messages.append(llm.assistant_message(turn))
            session.write()
            continue
        if turn.stop_reason in ("end_turn", "stop_sequence"):
            status = "completed"
            break
        if turn.stop_reason != "tool_use":
            status = "error"
            error = f"Unexpected stop_reason {turn.stop_reason!r}"
            break
        if not turn.tool_calls:
            status, error = "error", "stop_reason 'tool_use' without tool_use blocks"
            break

        messages.append(llm.assistant_message(turn))
        results, stop_status = _run_tool_calls(registry, ctx, session, entry["index"], turn)
        messages.append(
            llm.tool_results_message(
                [ToolResult(tool_call_id=cid, content=c, is_error=e) for cid, c, e in results]
            )
        )
        session.set_gate(gate)
        session.write()
        if stop_status is not None:
            status = stop_status
            if stop_status == "execution_failed":
                error = next(
                    (x.get("error") for x in reversed(ctx.executions) if x.get("error")), None
                )
            break

    session.set_gate(gate)
    report = _render_report(session, status, error)
    session.finish(
        status, final_text=final_text, report=report, error=error, stop_details=stop_details
    )
    path = session.write()
    return AgentResult(
        status=status,
        final_text=final_text,
        report=report,
        plans=list(session.plans),
        plan_yaml_paths=[p["yaml_path"] for p in session.plans if p.get("yaml_path")],
        executions=list(session.executions),
        transcript_path=str(path),
        session_id=session.session_id,
        workdir=str(ctx.workdir),
        error=error,
        stop_details=stop_details,
    )


def _run_tool_calls(
    registry: Any, ctx: Any, session: Any, turn_index: int, turn: Any
) -> tuple[list[tuple[str, str, bool]], str | None]:
    """Run the tool calls of one turn in order.

    Returns the ``(tool_call_id, content, is_error)`` triples (one per call)
    and the status that ends the session, if any. After an execution attempt
    or a denial the remaining calls of the turn are not run; they receive an
    error result so that every ``tool_use`` block has its ``tool_result``, and
    they are recorded in the transcript as errors.
    """
    results: list[tuple[str, str, bool]] = []
    stop_status: str | None = None
    for call in turn.tool_calls:
        if stop_status is not None:
            from agribound.agent.tools import ToolOutcome

            skipped = ToolOutcome(
                name=call.name,
                ok=False,
                error="Not run: the session ended after the previous tool call.",
                arguments=call.input,
            )
            session.record_tool_call(
                turn_index=turn_index,
                call_id=call.id,
                name=call.name,
                outcome=skipped,
                duration_s=0.0,
            )
            results.append((call.id, skipped.content(), True))
            continue
        n_exec = len(ctx.executions)
        n_denials = len(ctx.gate.denials) if ctx.gate is not None else 0
        t0 = time.perf_counter()
        outcome = registry.call(call.name, call.input)
        session.record_tool_call(
            turn_index=turn_index,
            call_id=call.id,
            name=call.name,
            outcome=outcome,
            duration_s=time.perf_counter() - t0,
        )
        results.append((call.id, outcome.content(), not outcome.ok))
        if len(ctx.executions) > n_exec:
            stop_status = "executed" if outcome.ok else "execution_failed"
        elif ctx.gate is not None and len(ctx.gate.denials) > n_denials:
            stop_status = "denied"
        # Other execute_plan refusals (unknown plan, changed inputs, no executions left,
        # network needed while offline) asked nothing of the reviewer and ran nothing;
        # the model sees the error and the loop goes on.
    return results, stop_status


__all__ = ["STATUSES", "AgentResult", "agent"]
