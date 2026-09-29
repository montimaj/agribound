"""
MCP server exposing the Agribound agent tools (``mcp`` >= 2.2).

``agribound mcp serve`` starts an :class:`mcp.server.mcpserver.MCPServer`
named ``"agribound"`` over stdio (default) or Streamable HTTP. Any MCP host
(Claude Desktop/Code or a local-LLM MCP host) can then call the same typed
tools that the built-in agent loop uses (:mod:`agribound.agent.tools`).

Tools and annotations
---------------------
- Read-only tools carry ``ToolAnnotations(read_only_hint=True)``.
  ``query_published_ftw`` and ``propose_run`` write files into the server's
  work directory (downloaded polygons, plan YAML), so they are annotated
  ``read_only_hint=False, destructive_hint=False``.
- ``propose_run`` is always registered; it never runs anything.
- ``execute_plan`` is registered **only** when the server is started with
  ``--allow-execute``. At most ``max_executions`` (default 1) plans run per
  server process.
- Arguments are validated by the same pydantic models as in the local loop,
  including the rejection of unknown argument names (MCPServer's own argument
  model would silently drop them; the raw request arguments are checked).

Work directory
--------------
Default: ``$XDG_DATA_HOME/agribound/mcp`` or ``~/.local/share/agribound/mcp``
(:func:`default_workdir`), not a path relative to the working directory the
MCP host happens to start the server in. ``--workdir`` overrides it.

Human confirmation of ``execute_plan``
--------------------------------------
- ``confirm="elicit"`` (default): before anything runs, the server sends an
  MCP elicitation showing the full plan and asks the user to type ``yes``.
  Clients that did not declare the elicitation capability get an error
  explaining the alternative. MCP lets a client answer elicitations itself
  (for example an automated host), so the server cannot prove that a human
  answered; the answer and method are recorded with the approval.
- ``confirm="host"``: no elicitation; the server relies on the host's own
  per-tool approval prompt. Use it only with hosts that ask the user before
  every tool call. The approval is recorded with the method "host
  tool-approval prompt" and an approver the server cannot verify.

In both modes a plan that cannot run (unknown plan, no executions left,
changed inputs, or remote services needed while ``allow_network=False``) is
refused before the user is asked (:func:`agribound.agent.tools.preflight_execution`).

Streamable HTTP
---------------
The ``streamable-http`` transport has no authentication: any process or user
that can reach the host and port can call the tools, and with
``--allow-execute`` such a client can answer the ``execute_plan`` elicitation
itself. :func:`serve` therefore refuses (:func:`check_http_exposure`, before
the server is built) ``--allow-execute`` over ``streamable-http`` and a
``--host`` that is not a loopback address, unless
``allow_unauthenticated_http=True`` (``--allow-unauthenticated-http``), which
logs a WARNING. The default host is ``127.0.0.1``.

Expected failures are raised as ``ToolError`` so the model reads the reason.
For ``execute_plan`` every failure after the approval (the pipeline, the
gate's final checks, or an unexpected exception) is reported as a
``ToolError`` with its type and message. For the other tools, an exception
that is not an :class:`~agribound.agent.errors.AgentToolError` is reported by
MCPServer as a generic "Error executing tool" and logged by the server.
While a tool runs, ``sys.stdout`` is redirected to ``sys.stderr``; the mcp
stdio transport additionally points file descriptor 1 at stderr while
serving (best effort, ``mcp.server.stdio``), so stray prints, including those
of native code, do not corrupt the protocol stream. Long runs send
``report_progress`` notifications every 15 s.
"""

from __future__ import annotations

import contextlib
import inspect
import ipaddress
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

logger = logging.getLogger(__name__)

CONFIRM_MODES = ("elicit", "host")
PROGRESS_INTERVAL_S = 15.0
_CTX_PARAM = "mcp_ctx"
"""Name of the injected :class:`mcp.server.mcpserver.Context` parameter of the wrappers."""


TRANSPORTS = ("stdio", "streamable-http")


class UnsafeTransportError(ValueError):
    """A ``streamable-http`` configuration refused because the transport has no authentication."""


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (``127.0.0.0/8``, ``::1``).

    Host names other than ``localhost`` are not resolved and count as not loopback.
    """
    name = str(host).strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def check_http_exposure(
    *,
    transport: str,
    host: str,
    port: int,
    allow_execute: bool,
    allow_unauthenticated_http: bool = False,
) -> None:
    """Refuse ``streamable-http`` set-ups that expose the tools to unauthenticated clients.

    Nothing is checked for ``stdio``. For ``streamable-http`` (no authentication),
    ``allow_execute`` and a *host* that is not loopback (:func:`is_loopback_host`) are
    refused unless *allow_unauthenticated_http* is set; with it, a WARNING describes
    what is exposed.

    Raises
    ------
    UnsafeTransportError
        If a refused combination is requested without *allow_unauthenticated_http*.
    """
    if transport != "streamable-http":
        return
    exposed = []
    if allow_execute:
        exposed.append(
            f"with --allow-execute, any client that can reach {host}:{port} can answer the "
            "execute_plan confirmation itself and run a plan"
        )
    if not is_loopback_host(host):
        exposed.append(
            f"--host {host} is not a loopback address, so other machines can reach the tools"
        )
    if allow_unauthenticated_http:
        detail = "; ".join(exposed) or (
            f"any local process or user that can reach {host}:{port} can call the tools"
        )
        logger.warning(
            "Serving MCP over streamable-http WITHOUT authentication "
            "(--allow-unauthenticated-http): %s.",
            detail,
        )
        return
    if exposed:
        raise UnsafeTransportError(
            "Refusing to serve: the streamable-http transport has no authentication, and "
            + "; ".join(exposed)
            + ". Use the default stdio transport (the MCP host starts the server itself), or "
            "keep --host 127.0.0.1 and leave out --allow-execute. To accept the risk, add "
            "--allow-unauthenticated-http."
        )


def default_workdir() -> Path:
    """``$XDG_DATA_HOME/agribound/mcp``, else ``~/.local/share/agribound/mcp``."""
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "agribound" / "mcp"


def _raw_arguments(ctx: Any) -> Mapping[str, Any]:
    """The ``arguments`` of the ``tools/call`` request, as the client sent them."""
    try:
        params = ctx.request_context.params
    except (AttributeError, ValueError):  # no request context (direct call)
        return {}
    arguments = params.get("arguments") if isinstance(params, Mapping) else None
    return arguments if isinstance(arguments, Mapping) else {}


def _unknown_arguments(ctx: Any, known: Any) -> dict[str, Any]:
    """Raw arguments whose names are not in *known* (MCPServer drops them silently)."""
    return {k: v for k, v in _raw_arguments(ctx).items() if k not in known}


class _StdoutToStderr:
    """Thread-safe, re-entrant redirection of ``sys.stdout`` to ``sys.stderr``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._saved: Any = None

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        with self._lock:
            if self._depth == 0:
                self._saved = sys.stdout
                sys.stdout = sys.stderr
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    sys.stdout = self._saved
                    self._saved = None


stdout_to_stderr = _StdoutToStderr()


def _import_mcp() -> dict[str, Any]:
    try:
        from mcp.server.mcpserver import (
            AcceptedElicitation,
            Context,
            Elicit,
            ElicitationResult,
            MCPServer,
            Resolve,
        )
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        from agribound.agent.errors import AgentDependencyError

        raise AgentDependencyError(
            'The MCP server needs the "mcp" package (>= 2.2): pip install "agribound[agent]"'
        ) from exc
    return {
        "AcceptedElicitation": AcceptedElicitation,
        "Context": Context,
        "Elicit": Elicit,
        "ElicitationResult": ElicitationResult,
        "MCPServer": MCPServer,
        "Resolve": Resolve,
        "ToolAnnotations": ToolAnnotations,
        "ToolError": ToolError,
    }


def _signature_for(spec: Any, context_type: type) -> inspect.Signature:
    """Keyword-only signature mirroring ``spec.input_model`` (for MCP schema generation).

    A final ``mcp_ctx`` parameter annotated with *context_type* receives the
    MCP request context; MCPServer leaves it out of the input schema.
    """
    from pydantic import Field

    if _CTX_PARAM in spec.input_model.model_fields:  # pragma: no cover - guarded by a test
        raise ValueError(f"{spec.name}: input field {_CTX_PARAM!r} clashes with the context")
    params = []
    for name, info in spec.input_model.model_fields.items():
        extras: list[Any] = list(info.metadata)
        field_kwargs: dict[str, Any] = {}
        if info.description:
            field_kwargs["description"] = info.description
        default: Any = inspect.Parameter.empty
        if info.default_factory is not None:
            field_kwargs["default_factory"] = info.default_factory
        elif not info.is_required():
            default = info.default
        if field_kwargs:
            extras.append(Field(**field_kwargs))
        annotation = Annotated[(info.annotation, *extras)] if extras else info.annotation
        params.append(
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=default
            )
        )
    params.append(
        inspect.Parameter(_CTX_PARAM, inspect.Parameter.KEYWORD_ONLY, annotation=context_type)
    )
    return inspect.Signature(params, return_annotation=spec.output_model)


def _annotations_for(spec: Any, tool_annotations: Any) -> Any:
    if spec.read_only:
        return tool_annotations(
            read_only_hint=True, destructive_hint=False, open_world_hint=spec.open_world
        )
    return tool_annotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=spec.name != "execute_plan",
        open_world_hint=spec.open_world,
    )


def _wrap_tool(spec: Any, context: Any, tool_error: type[Exception], context_type: type) -> Any:
    """Build a sync MCP tool function for a provider-neutral :class:`ToolSpec`.

    The arguments MCPServer parsed are validated again with ``spec.input_model``
    together with any unknown raw argument, so unknown names are rejected with
    the same message as in the local loop.
    """
    from pydantic import ValidationError

    from agribound.agent.errors import AgentToolError
    from agribound.agent.tools import format_validation_error

    def tool(**kwargs: Any) -> Any:
        ctx = kwargs.pop(_CTX_PARAM, None)
        with stdout_to_stderr():
            try:
                arguments = {**kwargs, **_unknown_arguments(ctx, spec.input_model.model_fields)}
                validated = spec.input_model.model_validate(arguments)
                return spec.func(context, validated)
            except ValidationError as exc:
                raise tool_error(format_validation_error(spec.name, exc)) from exc
            except AgentToolError as exc:
                raise tool_error(str(exc)) from exc

    signature = _signature_for(spec, context_type)
    tool.__name__ = spec.name
    tool.__qualname__ = spec.name
    tool.__doc__ = spec.description
    tool.__signature__ = signature  # type: ignore[attr-defined]
    tool.__annotations__ = {
        **{name: p.annotation for name, p in signature.parameters.items()},
        "return": spec.output_model,
    }
    return tool


def _elicitation_text(plan: Any) -> str:
    return (
        "The agent asks to run this Agribound plan. Type 'yes' to run exactly this plan; any "
        "other answer (or declining) cancels it.\n\n" + plan.render()
    )


def build_server(
    *,
    allow_execute: bool = False,
    confirm: Literal["elicit", "host"] = "elicit",
    workdir: str | Path | None = None,
    study_area: str | None = None,
    gee_project: str | None = None,
    reference_boundaries: str | None = None,
    allow_network: bool = True,
    max_executions: int = 1,
) -> Any:
    """Create the Agribound MCP server.

    Parameters
    ----------
    allow_execute : bool
        Register ``execute_plan`` (default *False*: plans can be proposed but
        not run through MCP).
    confirm : {"elicit", "host"}
        How ``execute_plan`` obtains human confirmation (module docstring).
    workdir : str, Path or None
        Directory for plans, outputs, downloads and the cache (default
        :func:`default_workdir`).
    study_area, gee_project, reference_boundaries : str or None
        Session defaults for the tools.
    allow_network : bool
        Allow tools to contact Earth Engine, TESSERA, Source Cooperative or
        the USGS NAIP Plus ImageServer. With *False*, ``execute_plan`` also
        refuses plans whose pipeline run needs one of them
        (:attr:`agribound.agent.tools.ToolContext.allow_network`).
    max_executions : int
        Maximum number of plans executed by this server process.

    Returns
    -------
    mcp.server.mcpserver.MCPServer
        The server; its tool context is available as ``server.agribound_context``.
    """
    if confirm not in CONFIRM_MODES:
        raise ValueError(f"confirm must be one of {CONFIRM_MODES}, got {confirm!r}")
    m = _import_mcp()
    from agribound._version import __version__
    from agribound.agent.gate import ConfirmationGate, deny_all
    from agribound.agent.prompts import MCP_INSTRUCTIONS
    from agribound.agent.tools import TOOL_SPECS, ToolContext

    context = ToolContext(
        workdir=Path(workdir) if workdir is not None else default_workdir(),
        study_area=study_area,
        gee_project=gee_project,
        reference_boundaries=reference_boundaries,
        allow_network=allow_network,
        execution_enabled=bool(allow_execute),
        # Approvals come only from record_approval below; the callback never approves.
        gate=ConfirmationGate(deny_all, max_executions=max_executions),
    )
    logger.info("Agribound MCP work directory: %s", context.workdir)
    server = m["MCPServer"](name="agribound", instructions=MCP_INSTRUCTIONS, version=__version__)
    tool_error = m["ToolError"]
    for spec in TOOL_SPECS:
        if spec.name == "execute_plan":
            continue
        server.add_tool(
            _wrap_tool(spec, context, tool_error, m["Context"]),
            name=spec.name,
            description=spec.description,
            annotations=_annotations_for(spec, m["ToolAnnotations"]),
            structured_output=True,
        )
    if allow_execute:
        spec = next(s for s in TOOL_SPECS if s.name == "execute_plan")
        server.add_tool(
            _execute_tool(context, confirm, m),
            name=spec.name,
            description=spec.description
            + (
                " The user confirms through an MCP elicitation."
                if confirm == "elicit"
                else " Confirmation relies on the host's tool-approval prompt."
            ),
            annotations=_annotations_for(spec, m["ToolAnnotations"]),
            structured_output=True,
        )
    server.agribound_context = context
    return server


def _execute_tool(context: Any, confirm: str, m: dict[str, Any]) -> Any:
    import anyio
    from pydantic import BaseModel, Field

    from agribound.agent.errors import AgentToolError
    from agribound.agent.tools import (
        ExecutePlanInput,
        ExecutionSummary,
        preflight_execution,
        run_approved_plan,
    )

    tool_error = m["ToolError"]
    context_type = m["Context"]
    gate = context.gate
    known_arguments = set(ExecutePlanInput.model_fields)

    def get_plan(plan_id: str, ctx: Any) -> Any:
        """Look up the plan and refuse, before anyone is asked, if it cannot run."""
        unknown = sorted(_unknown_arguments(ctx, known_arguments))
        if unknown:
            raise tool_error(
                "Invalid arguments for execute_plan: "
                + "; ".join(f"{name}: Extra inputs are not permitted" for name in unknown)
            )
        if gate.remaining_executions == 0:
            raise tool_error(
                f"This server already ran {gate.executions} plan(s) (max_executions="
                f"{gate.max_executions}). Restart it for another run."
            )
        try:
            plan = context.get_plan(plan_id)
            preflight_execution(context, plan)
        except AgentToolError as exc:
            raise tool_error(str(exc)) from exc
        return plan

    async def report(ctx: Any, progress: float, total: float | None, message: str) -> None:
        try:
            await ctx.report_progress(progress, total, message)
        except Exception as exc:  # progress is best effort; never fail the run over it
            logger.warning("Could not send a progress notification: %s", exc)

    async def run_with_progress(plan: Any, ctx: Any) -> ExecutionSummary:
        step = 0
        await report(ctx, step, None, f"Running plan {plan.plan_id}")
        done = anyio.Event()
        t0 = time.monotonic()
        outcome: dict[str, Any] = {}

        async def heartbeat() -> None:
            nonlocal step
            while not done.is_set():
                with anyio.move_on_after(PROGRESS_INTERVAL_S):
                    await done.wait()
                if not done.is_set():
                    step += 1
                    await report(
                        ctx,
                        step,
                        None,
                        f"Plan {plan.plan_id} running for {time.monotonic() - t0:.0f} s",
                    )

        def run_sync() -> ExecutionSummary:
            with stdout_to_stderr():
                return run_approved_plan(context, plan)

        async with anyio.create_task_group() as tg:
            tg.start_soon(heartbeat)
            try:
                outcome["summary"] = await anyio.to_thread.run_sync(run_sync)
            except Exception as exc:
                # Kept out of the task group: an exception leaving it is wrapped in an
                # ExceptionGroup, which MCPServer would report only as a generic error.
                outcome["error"] = exc
            finally:
                done.set()
        error = outcome.get("error")
        if error is not None:
            await report(ctx, step + 1, step + 1, f"Plan {plan.plan_id} failed")
            if isinstance(error, AgentToolError):
                message = str(error)
            else:
                logger.warning("Execution of %s raised", plan.plan_id, exc_info=error)
                message = (
                    f"Execution of plan {plan.plan_id} failed: {type(error).__name__}: {error}"
                )
            raise tool_error(message) from error
        await report(ctx, step + 1, step + 1, f"Plan {plan.plan_id} finished")
        return outcome["summary"]

    # Annotations are assigned as objects below: this module uses postponed
    # (string) annotations, which MCP could not resolve for these local names.
    if confirm == "host":

        async def execute_plan_host(plan_id, ctx):
            plan = get_plan(plan_id, ctx)
            gate.record_approval(
                plan,
                approver="MCP client user (not verified by the server)",
                method="host tool-approval prompt (server started with --allow-execute "
                "--confirm host)",
            )
            return await run_with_progress(plan, ctx)

        execute_plan_host.__annotations__ = {
            "plan_id": str,
            "ctx": context_type,
            "return": ExecutionSummary,
        }
        return execute_plan_host

    class ConfirmExecution(BaseModel):
        confirm: str = Field(description="Type yes to run this exact plan")

    # Hash of the plan each elicitation showed, by plan ID: the answer approves
    # that hash only, even if the ID were to name a different plan by then.
    shown_hashes: dict[str, str] = {}

    async def ask_confirmation(plan_id, ctx):
        plan = get_plan(plan_id, ctx)
        shown_hashes[plan.plan_id] = plan.plan_hash
        caps = ctx.client_capabilities
        if caps is None or caps.elicitation is None:
            raise tool_error(
                "This MCP client did not declare the elicitation capability, so the server "
                "cannot ask the user to confirm the plan. Use a client that supports "
                "elicitation, or restart the server with `agribound mcp serve --allow-execute "
                "--confirm host` to rely on the host's own tool-approval prompt."
            )
        return m["Elicit"](_elicitation_text(plan), ConfirmExecution)

    ask_confirmation.__annotations__ = {
        "plan_id": str,
        "ctx": context_type,
        "return": m["Elicit"][ConfirmExecution],
    }

    async def execute_plan_elicit(plan_id, confirmation, ctx):
        plan = get_plan(plan_id, ctx)
        accepted = isinstance(confirmation, m["AcceptedElicitation"])
        answer = confirmation.data.confirm if accepted else None
        shown = shown_hashes.get(plan.plan_id)
        if shown != plan.plan_hash:
            gate.record_denial(
                plan,
                method="MCP elicitation",
                reason=f"the plan shown (sha256 {shown}) is not the plan now under this ID",
            )
            raise tool_error(
                f"Plan {plan.plan_id} changed while the user was asked (shown sha256 {shown}, "
                f"now {plan.plan_hash}); nothing was run. Report this to the user."
            )
        if accepted and str(answer).strip().lower() == "yes":
            gate.record_approval(
                plan,
                approver="MCP client user (answer not verified as human by the server)",
                method="MCP elicitation (typed 'yes')",
            )
            return await run_with_progress(plan, ctx)
        reason = (
            f"answered {answer!r} instead of 'yes'"
            if accepted
            else f"elicitation {getattr(confirmation, 'action', 'unknown')}"
        )
        gate.record_denial(plan, method="MCP elicitation", reason=reason)
        raise tool_error(
            f"Plan {plan.plan_id} was not approved ({reason}). Do not modify the plan to obtain "
            "approval; report it to the user and stop."
        )

    elicitation_result = m["ElicitationResult"][ConfirmExecution]
    execute_plan_elicit.__annotations__ = {
        "plan_id": str,
        "confirmation": Annotated[elicitation_result, m["Resolve"](ask_confirmation)],
        "ctx": context_type,
        "return": ExecutionSummary,
    }
    return execute_plan_elicit


def serve(
    *,
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_unauthenticated_http: bool = False,
    **kwargs: Any,
) -> None:
    """Build the server (``**kwargs`` go to :func:`build_server`) and run it.

    ``streamable-http`` is checked first with :func:`check_http_exposure`:
    ``allow_execute=True`` or a non-loopback *host* raise
    :class:`UnsafeTransportError` unless *allow_unauthenticated_http* is set.
    """
    if transport not in TRANSPORTS:
        raise ValueError(f"Unknown transport {transport!r}")
    check_http_exposure(
        transport=transport,
        host=host,
        port=port,
        allow_execute=bool(kwargs.get("allow_execute", False)),
        allow_unauthenticated_http=allow_unauthenticated_http,
    )
    server = build_server(**kwargs)
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=host, port=port)


__all__ = [
    "CONFIRM_MODES",
    "TRANSPORTS",
    "UnsafeTransportError",
    "build_server",
    "check_http_exposure",
    "default_workdir",
    "is_loopback_host",
    "serve",
    "stdout_to_stderr",
]
