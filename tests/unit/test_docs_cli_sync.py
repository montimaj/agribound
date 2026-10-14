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


def test_cli_page_documents_every_evaluate_option():
    from agribound.cli import main

    page = _page("cli.md")
    section = page.split("## evaluate", 1)[1].split("\n## ", 1)[0]
    options = _option_names(main.commands["evaluate"])
    assert "--baselines" in options
    missing = [o for o in options if f"`{o}" not in section and f"/{o}`" not in section]
    assert not missing, f"the evaluate section of cli.md does not mention {missing}"


def test_cli_page_documents_the_scenes_command_and_date_layers():
    from agribound.cli import main

    page = _page("cli.md")
    assert "## scenes" in page and "| `scenes` |" in page
    section = page.split("## scenes", 1)[1].split("\n## ", 1)[0]
    own = ("--top", "--json", "--csv", "--scale-m")
    options = _option_names(main.commands["scenes"])
    assert set(own) <= set(options)
    missing = [o for o in own if o not in section]
    assert not missing, f"the scenes section of cli.md does not mention {missing}"
    missing = [o for o in options if o not in page]  # the stage-A options are listed above
    assert not missing, f"cli.md does not mention {missing}"
    composite = page.split("## composite", 1)[1].split("\n## ", 1)[0]
    assert "--date-layers" in _option_names(main.commands["composite"])
    assert "--date-layers" in composite


#: The acknowledgement the author decided on 2026-10-09 for the date-mixing question and
#: the SNIC work (an acknowledgement only: Jacob T. Stid is in no author list).
DECIDED_THANKS = (
    "Jacob Abramowitz (The University of Alabama in Huntsville), who raised the question "
    "of date mixing in tropical composites and shared his SNIC script",
    "Jacob T. Stid (Michigan State University) and Jeremy Rapp",
)


def _check_thanks(text: str, where: str) -> None:
    """*text* holds the decided wording (whitespace and bold aside) and no contact details."""
    flat = " ".join(text.replace("**", "").split())
    for phrase in DECIDED_THANKS:
        assert phrase in flat, f"{where}: the decided wording {phrase!r} is missing"
    for contact in ("@", "mailto:", "orcid", "http"):
        assert contact not in flat.lower(), f"{where}: contact detail {contact!r} in the thanks"


def test_satellite_sources_page_has_the_dates_section():
    page = _page("satellite-sources.md")
    assert "### Dates in a composite" in page
    assert "#### Superpixel segmentation (SNIC, SLIC)" in page
    dates = page.split("### Dates in a composite", 1)[1].split("\n### ", 1)[0]
    # One acknowledgement for the three collaborators, at the end of the section.
    assert dates.count("Thanks to") == 1
    thanks = dates[dates.index("Thanks to") :]
    _check_thanks(thanks, "satellite-sources.md")
    assert "Superpixel segmentation (SNIC, SLIC)" in dates.split("Thanks to")[0]


def test_the_snic_acknowledgement_everywhere_it_is_given():
    """The CHANGELOG's Documentation bullet and the README thank with the decided wording."""
    changelog, readme = ROOT / "CHANGELOG.md", ROOT / "README.md"
    if not (changelog.is_file() and readme.is_file()):
        pytest.skip("CHANGELOG.md and README.md are not available")
    unreleased = changelog.read_text(encoding="utf-8").split("\n## [", 2)[1]
    (bullet,) = [
        b for b in unreleased.split("\n- ") if b.startswith("**Documentation:**") and "SNIC" in b
    ]
    thanks = bullet[bullet.index("Thanks to Jacob Abramowitz") :]
    _check_thanks(thanks.split(" [", 1)[0], "CHANGELOG.md Documentation bullet")
    acknowledgments = readme.read_text(encoding="utf-8").split("## Acknowledgments", 1)[1]
    acknowledgments = acknowledgments.split("\n## ", 1)[0]
    (line,) = [ln for ln in acknowledgments.splitlines() if "Stid" in ln]
    _check_thanks(line, "README.md Acknowledgments")


def test_jacob_stid_is_acknowledged_only():
    """Jacob T. Stid is thanked, never listed as an author or in the citation files."""
    for name in ("CITATION.cff", "pyproject.toml", "docs/citation.md"):
        path = ROOT / name
        if not path.is_file():
            pytest.skip(f"{name} is not available")
        assert "Stid" not in path.read_text(encoding="utf-8"), name


def test_sam_page_documents_every_overlap_mode():
    from agribound.engines.samgeo_engine import SAM_OVERLAP_MODES

    page = _page("sam-refinement.md")
    assert 'engine_params["sam_overlaps"]' in page
    for mode in SAM_OVERLAP_MODES:
        assert f'`"{mode}"`' in page, f"sam-refinement.md does not describe sam_overlaps={mode!r}"


def test_sam_page_documents_the_coverage_test_and_its_default():
    from agribound.engines.samgeo_engine import DEFAULT_MIN_COVERAGE

    page = _page("sam-refinement.md")
    assert 'engine_params["sam_min_coverage"]' in page
    assert f"(default {DEFAULT_MIN_COVERAGE:g}" in page, "sam-refinement.md: wrong default"
    assert "n_low_coverage" in page


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
