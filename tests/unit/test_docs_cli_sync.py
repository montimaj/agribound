"""The user guide and the example scripts document the current command-line interface.

These tests read the source checkout (``docs/``, ``examples/``) and are skipped
when it is not available, e.g. when the suite runs from an installed wheel.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE = ROOT / "docs" / "user-guide"
EXAMPLES = ROOT / "examples"


def _page(name: str) -> str:
    path = USER_GUIDE / name
    if not path.is_file():
        pytest.skip(f"docs/user-guide/{name} is not available")
    return path.read_text(encoding="utf-8")


def _option_names(command) -> list[str]:
    """Long option names of a Click command (``--flag`` and ``--no-flag`` forms)."""
    import click

    names = []
    for param in command.params:
        if isinstance(param, click.Option):
            names += [o for o in (*param.opts, *param.secondary_opts) if o.startswith("--")]
    return names


def test_cli_and_hpc_pages_list_every_tiles_subcommand():
    from agribound.hpc.cli import tiles

    cli_page, hpc_page = _page("cli.md"), _page("hpc.md")
    assert "gee-project" in tiles.commands
    for name in tiles.commands:
        assert f"tiles {name}" in cli_page, f"cli.md does not list 'tiles {name}'"
        assert f"tiles {name}" in hpc_page, f"hpc.md does not list 'tiles {name}'"


def test_agent_page_documents_every_mcp_serve_option():
    from agribound.agent.cli import mcp

    page = _page("agent.md")
    options = _option_names(mcp.commands["serve"])
    assert "--allow-unauthenticated-http" in options
    missing = [o for o in options if o not in page]
    assert not missing, f"agent.md does not mention {missing}"


def test_cli_page_documents_every_query_ftw_option():
    from agribound.cli import main

    page = _page("cli.md")
    options = _option_names(main.commands["query-ftw"])
    missing = [o for o in options if o not in page]
    assert not missing, f"cli.md does not mention {missing}"


def test_sam_page_documents_every_overlap_mode():
    from agribound.engines.samgeo_engine import SAM_OVERLAP_MODES

    page = _page("sam-refinement.md")
    assert 'engine_params["sam_overlaps"]' in page
    for mode in SAM_OVERLAP_MODES:
        assert f'`"{mode}"`' in page, f"sam-refinement.md does not describe sam_overlaps={mode!r}"


def _gee_project_help(script: Path) -> str | None:
    """The help text of the script's ``--gee-project`` argparse option, if any."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--gee-project"
        ):
            for keyword in node.keywords:
                if keyword.arg == "help":
                    return ast.literal_eval(keyword.value)
            return ""
    return None


def test_example_gee_project_help_states_the_lookup_order():
    scripts = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
    if not scripts:
        pytest.skip("examples/ scripts are not available")
    helps = {s.name: _gee_project_help(s) for s in scripts}
    helps = {name: text for name, text in helps.items() if text is not None}
    assert len(helps) >= 10, "expected most numbered examples to take --gee-project"
    # agribound's lookup (AgriboundConfig and agribound.auth.setup_gee): GEE_PROJECT,
    # then gcloud, then the project_id of the credentials file.
    order = (
        "$GEE_PROJECT",
        "gcloud",
        "project_id",
        "$AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY",
        "$GOOGLE_APPLICATION_CREDENTIALS",
    )
    for name, text in helps.items():
        positions = [text.find(term) for term in order]
        assert -1 not in positions, f"{name}: --gee-project help lacks {order}: {text!r}"
        assert positions == sorted(positions), f"{name}: lookup order wrong in {text!r}"
