"""
Command-line entry points of the agent layer.

- ``agribound agent "<request>" --study-area ...`` runs one agent session.
  Confirmation is interactive: the full plan is printed and you must type
  ``yes`` for it to run. Without an interactive terminal the command refuses
  to start unless ``--dry-run`` is given (which only writes plan YAML files).
  There is deliberately no option to skip the confirmation.
- ``agribound mcp serve [--allow-execute] [--transport stdio|streamable-http]``
  serves the tools over MCP (:mod:`agribound.agent.mcp_server`).
  ``streamable-http`` has no authentication, so ``--allow-execute`` with it, or a
  non-loopback ``--host``, is refused unless ``--allow-unauthenticated-http`` is
  given (which logs a WARNING).

:mod:`agribound.cli` registers both commands. This module imports only
``click`` at import time.
"""

from __future__ import annotations

import sys

import click


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


@click.command("agent")
@click.argument("request")
@click.option(
    "--study-area",
    default=None,
    help="Study area: vector file, GEE asset ID, 'bbox:minx,miny,maxx,maxy' or WKT.",
)
@click.option("--gee-project", default=None, help="Google Earth Engine project ID.")
@click.option(
    "--reference",
    "reference_boundaries",
    default=None,
    help="Reference field boundaries available to the tools (resolvability, evaluation, "
    "fine-tuning).",
)
@click.option(
    "--model",
    default=None,
    help="Model ID (default: $AGRIBOUND_AGENT_MODEL or claude-opus-5).",
)
@click.option(
    "--base-url",
    default=None,
    help="Anthropic-compatible endpoint, e.g. http://localhost:11434 for Ollama.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Propose a plan and write its YAML; never execute anything.",
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(file_okay=False),
    help="Session directory (default ./agribound_agent/<session id>).",
)
@click.option("--max-turns", default=20, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--offline",
    is_flag=True,
    help="Tools may not contact Earth Engine, TESSERA, Source Cooperative or the USGS "
    "ImageServer, and a plan that needs one of them is refused before you are asked. Model "
    "weights may still be downloaded from Hugging Face unless cached (agribound prefetch).",
)
def agent(
    request: str,
    study_area: str | None,
    gee_project: str | None,
    reference_boundaries: str | None,
    model: str | None,
    base_url: str | None,
    dry_run: bool,
    workdir: str | None,
    max_turns: int,
    offline: bool,
) -> None:
    """Plan an Agribound run from a natural-language REQUEST (human-confirmed).

    The agent proposes one configuration; you review the full plan and type
    'yes' to run it. At most one plan runs, and the session ends afterwards.
    """
    if not dry_run and not _stdin_is_interactive():
        raise click.UsageError(
            "agribound agent asks you to confirm the plan at the terminal, but standard input "
            "is not interactive. Run it in a terminal, or use --dry-run to only write the plan "
            "YAML (then run: agribound delineate --config <plan yaml>)."
        )
    from agribound.agent.agent import agent as run_agent
    from agribound.agent.gate import prompt_confirm

    try:
        result = run_agent(
            request,
            study_area=study_area,
            gee_project=gee_project,
            workdir=workdir,
            model=model,
            base_url=base_url,
            confirm=prompt_confirm,
            dry_run=dry_run,
            max_turns=max_turns,
            reference_boundaries=reference_boundaries,
            allow_network=not offline,
        )
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    from agribound.agent.plans import display_safe

    # The model's text (and values it chose, quoted in the report) is shown with
    # control characters escaped, so it cannot rewrite what the terminal shows.
    if result.final_text:
        click.echo(display_safe(result.final_text, keep_newlines=True))
        click.echo("")
    click.echo(display_safe(result.report, keep_newlines=True))
    if result.status in ("error", "refused", "max_tokens", "execution_failed"):
        sys.exit(1)


@click.group("mcp")
def mcp() -> None:
    """Serve the Agribound agent tools over the Model Context Protocol."""


@mcp.command("serve")
@click.option(
    "--allow-execute",
    is_flag=True,
    help="Register execute_plan (one approved plan per server process).",
)
@click.option(
    "--confirm",
    type=click.Choice(["elicit", "host"]),
    default="elicit",
    show_default=True,
    help="How execute_plan is confirmed: an MCP elicitation where the user types 'yes' "
    "(elicit), or the host's own tool-approval prompt (host).",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    show_default=True,
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="streamable-http host. A host that is not loopback needs --allow-unauthenticated-http.",
)
@click.option("--port", default=8000, show_default=True, type=int, help="streamable-http port.")
@click.option(
    "--allow-unauthenticated-http",
    is_flag=True,
    help="streamable-http has no authentication: allow it with --allow-execute or a "
    "non-loopback --host anyway (logs a WARNING). Without this flag both are refused.",
)
@click.option(
    "--workdir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory for plans and outputs (default $XDG_DATA_HOME/agribound/mcp or "
    "~/.local/share/agribound/mcp).",
)
@click.option("--study-area", default=None, help="Default study area for the tools.")
@click.option("--gee-project", default=None, help="Default Earth Engine project.")
@click.option("--reference", "reference_boundaries", default=None, help="Default reference layer.")
@click.option(
    "--offline",
    is_flag=True,
    help="Tools may not contact Earth Engine, TESSERA, Source Cooperative or the USGS "
    "ImageServer, and execute_plan refuses plans that need one of them. Model weights may "
    "still be downloaded from Hugging Face unless cached (agribound prefetch).",
)
def serve(
    allow_execute: bool,
    confirm: str,
    transport: str,
    host: str,
    port: int,
    allow_unauthenticated_http: bool,
    workdir: str | None,
    study_area: str | None,
    gee_project: str | None,
    reference_boundaries: str | None,
    offline: bool,
) -> None:
    """Run the MCP server (stdio by default, for Claude Desktop/Code or other MCP hosts).

    streamable-http has no authentication: whoever reaches the port can call the
    tools. It is therefore refused with --allow-execute or a non-loopback --host
    unless --allow-unauthenticated-http is given.
    """
    from agribound.agent.mcp_server import UnsafeTransportError
    from agribound.agent.mcp_server import serve as run_server

    try:
        run_server(
            transport=transport,
            host=host,
            port=port,
            allow_unauthenticated_http=allow_unauthenticated_http,
            allow_execute=allow_execute,
            confirm=confirm,
            workdir=workdir,
            study_area=study_area,
            gee_project=gee_project,
            reference_boundaries=reference_boundaries,
            allow_network=not offline,
        )
    except UnsafeTransportError as exc:
        raise click.UsageError(str(exc)) from exc
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["agent", "mcp"]
