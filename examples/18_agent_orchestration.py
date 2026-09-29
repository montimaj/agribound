"""
18 — Agent-Planned Runs with a Human Confirmation Gate (dry run)

The optional agent layer (``pip install "agribound[agent]"``) turns a
natural-language request into one proposed agribound configuration. Its
autonomy is deliberately low:

    - the model investigates with read-only tools and can only *propose* a
      run (``propose_run`` validates a configuration, freezes it into a plan
      with a SHA-256 hash and writes its YAML);
    - ``execute_plan`` shows the full plan to a human, who must approve that
      exact plan (typed ``yes``; the approval is bound to the plan hash and
      used once);
    - at most one plan runs per session, and the session stops right after
      the run or a denial: there is no automatic re-run or re-tuning;
    - with ``dry_run=True`` (``agribound agent --dry-run``) ``execute_plan``
      is not offered at all; the plan YAML can be run later with
      ``agribound delineate --config <plan.yaml>``.

Every model turn and tool call is written to a JSON transcript.

This script is a dry run in both parts and never executes a plan:

    Part A (default, no LLM, no API key): calls the agent's read-only tools
        directly (describe the study area, check availability, estimate
        resolvability, rank configurations) and freezes the top-ranked
        configuration with ``propose_run``. The tools run offline
        (``allow_network=False``): no live Earth Engine or TESSERA checks.
    Part B (``--llm``): the same request through ``agribound.agent(...,
        dry_run=True)``. It needs Anthropic API credentials
        (``ANTHROPIC_API_KEY``), or an Anthropic-compatible local server with
        ``--base-url`` (set ``ANTHROPIC_API_KEY`` to the placeholder that
        server expects). The model is ``$AGRIBOUND_AGENT_MODEL`` or
        ``claude-opus-5``; ``--model`` overrides it.

MCP: ``agribound mcp serve`` exposes the same tools to MCP hosts such as
Claude Desktop or Claude Code (stdio). ``execute_plan`` is registered only
with ``--allow-execute`` and is confirmed through an MCP elicitation (the
user types ``yes``) by default. The script prints a configuration snippet.

Runtime: about a second for part A (tested 2026-09-27); part B depends on the
model and was not run for this release (no API key was available).

Prerequisites:
    pip install "agribound[agent]"
    Run from the repository root: python examples/18_agent_orchestration.py [--llm]
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import agribound

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --- Configuration ---
OUTPUT_DIR = Path("outputs/agent_orchestration")
STUDY_AREA = "bbox:1.40,48.10,1.55,48.20"  # Beauce, France (example 04)
YEAR = 2024
REQUEST = (
    "Delineate agricultural field boundaries in this study area for 2024 with a "
    "label-free approach. I have no reference boundaries."
)
FIELD_AREA_HA = 10.0  # representative field size for the resolvability estimate


def call_tool(registry, name, **arguments):
    """Call one agent tool; return its output dict or raise with the tool's error."""
    outcome = registry.call(name, arguments)
    if not outcome.ok:
        raise RuntimeError(f"{name}: {outcome.error}")
    return outcome.output


def part_a_tools(gee_project):
    """Drive the read-only tools and propose_run without an LLM."""
    from agribound.agent import ToolContext, ToolRegistry

    ctx = ToolContext(
        workdir=OUTPUT_DIR / "tools_session",
        study_area=STUDY_AREA,
        gee_project=gee_project,
        allow_network=False,
        execution_enabled=False,  # execute_plan is not registered
    )
    registry = ToolRegistry(ctx)
    print(f"Tools offered: {sorted(registry.specs)}")

    area = call_tool(registry, "describe_study_area")
    print(
        f"\nStudy area: {area['area_km2']:.1f} km^2, UTM zone {area['utm_zone']} "
        f"(EPSG:{area['utm_epsg']})"
    )
    for est in area["composite_estimates"][:4]:
        print(f"  {est['source']:<18} {est['resolution_m']} m, {est['uncompressed_mb']} MB")

    avail = call_tool(
        registry,
        "check_availability",
        year=YEAR,
        sources=["sentinel2", "landsat", "naip", "google-embedding", "tessera-embedding"],
    )
    print(f"\nAvailability in {YEAR} (registry ranges only; no live check):")
    for row in avail["results"]:
        print(f"  {row['source']:<18} in range: {row['in_registry_range']}")

    resolv = call_tool(registry, "estimate_resolvability", median_field_area_ha=FIELD_AREA_HA)
    print(f"\nResolvability of one square {FIELD_AREA_HA} ha field:")
    for row in resolv["per_source"]:
        sam = row["sam_refinement"]
        median_px = (row["pixels_per_field"] or {}).get("median")
        print(
            f"  {row['source']:<18} GSD {row['gsd_m']:>5} m  {median_px:>10,.0f} pixels/field  "
            f"SAM-refinable fraction: {sam.get('eligible_fraction')}"
        )

    rec = call_tool(
        registry,
        "recommend_configurations",
        year=YEAR,
        median_field_area_ha=FIELD_AREA_HA,
        prefer_label_free=True,
        max_candidates=5,
    )
    print("\nRanked candidates (deterministic rules; accuracy is not predicted):")
    for cand in rec["candidates"]:
        n_warn = len(cand["warnings"])
        print(f"  {cand['rank']}. {cand['source']} + {cand['engine']} ({n_warn} warnings)")
    if not rec["candidates"]:
        print("  none")
        return None

    top = rec["candidates"][0]
    plan = call_tool(
        registry,
        "propose_run",
        **top["proposal"],
        rationale="Top-ranked label-free candidate (example 18, part A).",
    )
    print(f"\nProposed plan {plan['plan_id']} (sha256 {plan['plan_hash'][:12]}...)")
    print(f"  YAML: {plan['yaml_path']}")
    print(f"  Remote services: {plan['network_services']}")
    for warning in plan["warnings"]:
        print(f"  warning: {warning}")
    print(
        "  Nothing has run. Review the YAML, then run it yourself with\n"
        f"    agribound delineate --config {plan['yaml_path']}"
    )
    return plan


def part_b_llm(gee_project, model, base_url):
    """Run the agent loop with an LLM in dry-run mode (no execution possible)."""
    result = agribound.agent(
        REQUEST,
        study_area=STUDY_AREA,
        gee_project=gee_project,
        workdir=OUTPUT_DIR / "llm_session",
        model=model,
        base_url=base_url,
        dry_run=True,
    )
    print(f"\nStatus: {result.status}")
    if result.final_text:
        print(f"\nModel summary:\n{result.final_text}")
    print(f"\n{result.report}")
    return result


def print_mcp_instructions():
    """Print how to expose the tools over MCP."""
    # The agribound executable of this Python environment (MCP hosts do not use your shell).
    local = Path(sys.executable).with_name("agribound")
    command = str(local) if local.exists() else shutil.which("agribound") or "agribound"
    snippet = {"mcpServers": {"agribound": {"command": command, "args": ["mcp", "serve"]}}}
    print(f"\n{'=' * 70}\nMCP\n{'=' * 70}")
    print("Claude Desktop (claude_desktop_config.json), read-only tools + propose_run:")
    print(json.dumps(snippet, indent=2))
    print(
        'Add "--allow-execute" to "args" to register execute_plan (one approved plan per server '
        "process; confirmed by an MCP elicitation where you type 'yes').\n"
        f"Claude Code: claude mcp add agribound -- {command} mcp serve"
    )


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(description="Agent-planned runs (dry run).")
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project for plans (default: $GEE_PROJECT, then the gcloud project, "
            "then the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument("--llm", action="store_true", help="Also run part B (needs an LLM).")
    parser.add_argument("--model", default=None, help="Model ID for part B.")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible endpoint.")
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    return parser.parse_args(argv)


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}\nPart A: agent tools without an LLM\n{'=' * 70}")
    try:
        part_a_tools(args.gee_project)
    except ImportError as exc:
        print(f'The agent tools need the agent extra: pip install "agribound[agent]" ({exc})')
    except RuntimeError as exc:
        print(f"Tool error: {exc}")

    if args.llm:
        print(f"\n{'=' * 70}\nPart B: agribound.agent(dry_run=True)\n{'=' * 70}")
        try:
            part_b_llm(args.gee_project, args.model, args.base_url)
        except ImportError as exc:
            print(f'The agent needs the agent extra: pip install "agribound[agent]" ({exc})')
    else:
        print("\nPart B skipped (pass --llm to call a model).")

    print_mcp_instructions()

    from agribound.io.vector import read_study_area

    web_map = agribound.show_boundaries(
        read_study_area(STUDY_AREA),
        layer_name="Study area of the proposed plan",
        output_html=str(OUTPUT_DIR / "map_study_area.html"),
    )
    show_in_notebook(web_map)
    print(f"\nStudy-area map: {OUTPUT_DIR / 'map_study_area.html'}")


if __name__ == "__main__":
    main()
