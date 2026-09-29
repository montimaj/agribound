"""Regression tests for the agent review-screen and approval hardening (no network, no GEE)."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import io
import math
import time
from types import SimpleNamespace

import geopandas as gpd
import pytest
from click.testing import CliRunner
from shapely.geometry import box

pytest.importorskip("pydantic")

from agribound.agent import tools as tools_mod  # noqa: E402
from agribound.agent.errors import ExecutionNotApprovedError, PlanChangedError  # noqa: E402
from agribound.agent.gate import ConfirmationGate, prompt_confirm  # noqa: E402
from agribound.agent.plans import display_safe, make_plan  # noqa: E402
from agribound.agent.tools import ResolvabilityInput, ToolContext, ToolRegistry  # noqa: E402
from agribound.config import AgriboundConfig  # noqa: E402

CA_BBOX = (-119.30, 36.30, -119.28, 36.32)
PROPOSE = {"source": "sentinel2", "engine": "delineate-anything", "year": 2023}


def _aoi_file(tmp_path, bbox=CA_BBOX, name="aoi.geojson"):
    path = tmp_path / name
    gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture(autouse=True)
def _no_earth_engine(monkeypatch):
    def refuse(ctx):
        raise AssertionError("a test tried to initialise Earth Engine")

    monkeypatch.setattr(tools_mod, "_init_gee", refuse)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workdir=tmp_path / "work",
        study_area=_aoi_file(tmp_path),
        gee_project="test-project",
        allow_network=False,
    )


@pytest.fixture
def registry(ctx):
    return ToolRegistry(ctx)


def _config(tmp_path, **overrides):
    fields = {
        **PROPOSE,
        "study_area": _aoi_file(tmp_path),
        "gee_project": "test-project",
        "output_path": str(tmp_path / "out" / "fields.gpkg"),
    }
    fields.update(overrides)
    return AgriboundConfig(**fields)


# ---------------------------------------------------------------------------
# Review screen: agent text cannot steer the terminal
# ---------------------------------------------------------------------------


def test_display_safe_escapes_terminal_controls_and_keeps_text():
    assert display_safe("a\x1b[2Jb") == "a\\x1b[2Jb"
    assert display_safe("x\ny\tz\r") == "x\\ny\\tz\\r"
    assert display_safe("rtl‮override sep\x85nel\x7f") == (
        "rtl\\u202eoverride\\u2028sep\\x85nel\\x7f"
    )
    assert display_safe("a\nb\x1b", keep_newlines=True) == "a\nb\\x1b"
    text = "Naïve 中文 field → 😀 (ok)"
    assert display_safe(text) == text


def test_render_escapes_ansi_sequences_in_agent_text(tmp_path):
    """Regression (R01): ESC sequences in the rationale erased Agribound's warnings."""
    erase = "\x1b[20F\x1b[JWarnings (from Agribound):\x1b[1E  (none)"
    plan = make_plan(
        _config(tmp_path),
        rationale=erase,
        limitations=["\x1b[2Jcleared"],
        alternatives=["‮esrever"],
        warnings=["lulc_crop_threshold is 0.0"],
    )
    screen = plan.render()
    assert "\x1b" not in screen and "‮" not in screen
    assert "\\x1b[20F\\x1b[J" in screen and "\\x1b[2Jcleared" in screen
    assert "\\u202eesrever" in screen
    # The real warning section is still the only unmarked one.
    assert screen.count("Warnings (from Agribound):") == 2
    unmarked = [line for line in screen.splitlines() if not line.startswith("  | ")]
    assert sum(line == "Warnings (from Agribound):" for line in unmarked) == 1
    out = io.StringIO()
    prompt_confirm(plan, input_fn=lambda _: "no", out=out)
    assert "\x1b" not in out.getvalue()


def test_render_escapes_control_characters_in_header_values(tmp_path):
    """Header values (e.g. an output path) cannot add unmarked lines."""
    cfg = _config(tmp_path, output_path=str(tmp_path / "f\nWarnings (from Agribound):\x1b[J.gpkg"))
    screen = make_plan(cfg).render()
    header = screen.split("Warnings (from Agribound):\n", 1)[0]
    assert "\x1b" not in screen
    assert "\\nWarnings (from Agribound):\\x1b[J.gpkg" in header
    assert [line for line in screen.splitlines() if line == "Warnings (from Agribound):"] == [
        "Warnings (from Agribound):"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_name", "f\nWarnings (from Agribound):\n  (none)\nx.gpkg"),
        ("output_name", "f\x1b[2J.gpkg"),
        ("study_area", "projects/p/assets/aoi\n\x1b[1F  source=local"),
        ("config", {"sam_model": "tiny\x1b[2J"}),
        ("config", {"engine_params": {"note": "a‮b"}}),
    ],
)
def test_propose_run_rejects_control_characters(registry, field, value):
    out = registry.call("propose_run", {**PROPOSE, field: value})
    assert not out.ok
    assert "must not contain control or format characters" in out.error
    assert field in out.error


def test_propose_run_keeps_multiline_free_text(registry, ctx):
    out = registry.call(
        "propose_run", {**PROPOSE, "rationale": "line 1\nline 2", "limitations": ["a\nb"]}
    )
    assert out.ok, out.error
    screen = ctx.plans[out.output["plan_id"]].render()
    assert "  | line 1\n  | line 2" in screen and "  | - a\n  |   b" in screen


def test_agent_cli_escapes_the_model_text(monkeypatch):
    module = importlib.import_module("agribound.agent.agent")
    monkeypatch.setattr(
        module,
        "agent",
        lambda request, **k: SimpleNamespace(
            final_text="Done.\x1b[2J\nsecond line", report="REPORT\x1b[1F", status="completed"
        ),
    )
    from agribound.cli import main

    result = CliRunner().invoke(main, ["agent", "x", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "Done.\\x1b[2J\nsecond line" in result.output and "REPORT\\x1b[1F" in result.output


# ---------------------------------------------------------------------------
# Resolvability: bounded inputs and a terminating bisection
# ---------------------------------------------------------------------------


def test_min_refinable_side_terminates_for_huge_inputs():
    """Regression (R02): the bisection spun forever once the float spacing exceeded 0.1 m."""
    from agribound.engines.samgeo_engine import is_refinable

    t0 = time.perf_counter()
    side = tools_mod._min_refinable_square_side_m(1e17, 64, 0.15, is_refinable)
    assert side is not None and math.isfinite(side)
    assert tools_mod._min_refinable_square_side_m(1e10, 10**17, 0.15, is_refinable) is not None
    assert tools_mod._min_refinable_square_side_m(float("inf"), 64, 0.15, is_refinable) is None
    assert time.perf_counter() - t0 < 5
    # Ordinary inputs keep their exact answer: 10 m pixels, 64 px, 15 % padding.
    small = tools_mod._min_refinable_square_side_m(10.0, 64, 0.15, is_refinable)
    assert is_refinable((0, 0, small, small), (10.0, 10.0), min_crop_px=64, padding=0.15)
    assert not is_refinable(
        (0, 0, small - 0.2, small - 0.2), (10.0, 10.0), min_crop_px=64, padding=0.15
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_crop_px": 10**17},
        {"gsd_m": {"sentinel2": 1e17}},
        {"gsd_m": {"sentinel2": float("inf")}},
        {"gsd_m": {"sentinel2": 0}},
        {"crop_padding": float("nan")},
        {"crop_padding": 1e9},
    ],
)
def test_resolvability_input_bounds(kwargs):
    with pytest.raises(ValueError):
        ResolvabilityInput(median_field_area_ha=1.0, **kwargs)


def test_propose_run_with_huge_sam_min_crop_px_returns(registry):
    out = registry.call(
        "propose_run", {**PROPOSE, "config": {"sam_refine": True, "sam_min_crop_px": 10**17}}
    )
    assert out.ok, out.error


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_config_rejects_non_finite_sizes(tmp_path, value):
    for name in ("sam_crop_padding", "min_field_area_m2", "simplify_tolerance"):
        with pytest.raises(ValueError, match="finite"):
            _config(tmp_path, **{name: value})


# ---------------------------------------------------------------------------
# Proposal fields that redirect data or credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["embedding_cache_dir", "gee_project"])
def test_proposal_cannot_set_session_owned_fields(registry, tmp_path, name):
    out = registry.call("propose_run", {**PROPOSE, "config": {name: str(tmp_path / "x")}})
    assert not out.ok and "cannot be set by a proposal" in out.error and name in out.error


def test_destination_fields_carry_warnings_and_are_listed(registry, ctx):
    usgs = registry.call(
        "propose_run",
        {
            "source": "usgs-naip-plus",
            "engine": "delineate-anything",
            "year": 2022,
            "config": {"usgs_service_url": "https://evil.example.org/arcgis/ImageServer"},
        },
    )
    assert usgs.ok, usgs.error
    assert any(w.startswith("usgs_service_url is ") for w in usgs.output["warnings"])
    assert any("evil.example.org" in s for s in usgs.output["network_services"])
    assert not any(s.startswith("USGS NAIP Plus") for s in usgs.output["network_services"])

    gcs = registry.call(
        "propose_run", {**PROPOSE, "config": {"export_method": "gcs", "gcs_bucket": "their-bucket"}}
    )
    assert gcs.ok, gcs.error
    warned = [w.split(" is ")[0] for w in gcs.output["warnings"] if "remote service" in w]
    assert warned == ["export_method", "gcs_bucket"]
    assert (
        "Google Cloud Storage bucket 'their-bucket' (batch export)"
        in (gcs.output["network_services"])
    )
    default = registry.call("propose_run", PROPOSE)
    assert not any("remote service" in w for w in default.output["warnings"])


def test_propose_run_refuses_a_plan_id_held_by_another_plan(registry, ctx, tmp_path):
    first = registry.call("propose_run", PROPOSE)
    assert first.ok, first.error
    plan_id = first.output["plan_id"]
    other = make_plan(_config(tmp_path, year=2022))
    ctx.plans[plan_id] = dataclasses.replace(other, plan_id=plan_id)  # simulated collision
    again = registry.call("propose_run", PROPOSE)
    assert not again.ok and "already used by a different plan" in again.error
    assert ctx.plans[plan_id].plan_hash == other.plan_hash


# ---------------------------------------------------------------------------
# Gate: an approval that could not be used never authorises a later run
# ---------------------------------------------------------------------------


def test_approval_is_revoked_when_the_plan_changes_before_it_runs(tmp_path):
    aoi = tmp_path / "aoi.geojson"
    plan = make_plan(_config(tmp_path))
    asked = []
    gate = ConfirmationGate(lambda p: asked.append(p.plan_id) or True, max_executions=2)
    gate.request_approval(plan)
    original = aoi.read_bytes()
    _aoi_file(tmp_path, bbox=(-119.40, 36.30, -119.38, 36.32))
    with pytest.raises(PlanChangedError):
        gate.authorize_execution(plan)
    assert not gate.approvals[0].usable and gate.approvals[0].revoked_reason
    aoi.write_bytes(original)  # inputs restored: the hash matches again
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(plan)  # the revoked approval is not reused
    gate.request_approval(plan)
    assert asked == [plan.plan_id, plan.plan_id]  # the reviewer was asked again
    assert gate.authorize_execution(plan) is gate.approvals[1]
    assert gate.executions == 1


def test_request_approval_always_asks_and_supersedes_unused_approvals(tmp_path):
    plan = make_plan(_config(tmp_path))
    answers = iter([True, False])
    gate = ConfirmationGate(lambda p: next(answers), max_executions=2)
    gate.request_approval(plan)
    with pytest.raises(Exception, match="not approved"):
        gate.request_approval(plan)  # asked again, and the reviewer said no
    assert not gate.approvals[0].usable
    with pytest.raises(ExecutionNotApprovedError):
        gate.authorize_execution(plan)
    assert gate.executions == 0


def test_execute_plan_asks_again_after_a_failed_authorisation(tmp_path, monkeypatch):
    """Regression: an approval left unused by PlanChangedError was reused without asking."""
    runs, asked = [], []

    def fake(config=None, **kwargs):
        runs.append(config)
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32611"
        )
        gdf.attrs["run_id"] = "r"
        return gdf

    monkeypatch.setattr("agribound.pipeline.delineate", fake)
    aoi = _aoi_file(tmp_path)
    ctx = ToolContext(
        workdir=tmp_path / "w",
        study_area=aoi,
        gee_project="test-project",
        allow_network=True,
        execution_enabled=True,
        gate=ConfirmationGate(lambda p: asked.append(p.plan_id) or True),
    )
    reg = ToolRegistry(ctx)
    plan_id = reg.call("propose_run", PROPOSE).output["plan_id"]
    plan = ctx.plans[plan_id]
    checks = []

    def current_hash(self):
        # 1st check (preflight) passes; the 2nd (authorize_execution, after the
        # reviewer said yes) sees changed inputs; later checks pass again.
        checks.append(1)
        return "0" * 64 if len(checks) == 2 else self.plan_hash

    monkeypatch.setattr(type(plan), "current_hash", current_hash)
    first = reg.call("execute_plan", {"plan_id": plan_id})
    assert not first.ok and "no longer matches its hash" in first.error
    assert runs == [] and ctx.gate.executions == 0
    second = reg.call("execute_plan", {"plan_id": plan_id})
    assert second.ok, second.error
    assert asked == [plan_id, plan_id] and len(runs) == 1


# ---------------------------------------------------------------------------
# MCP: the elicited answer approves the hash that was shown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_mcp_elicitation_is_bound_to_the_shown_hash(tmp_path, monkeypatch, mode):
    """The answer approves the plan hash the user was shown, never a later plan under its ID.

    Legacy protocol: the question is asked once, mid-call, so a plan that takes
    over the ID meanwhile is refused. Modern protocol (2026-07-28): the resolver
    re-runs on the retry, the question text (with the new hash) differs, so MCP
    asks again and the approval is for the plan shown the second time.
    """
    pytest.importorskip("mcp")
    import mcp_types as types
    from mcp import Client

    from agribound.agent.mcp_server import build_server

    runs = []

    def fake(config=None, **kwargs):
        runs.append(config)
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32611"
        )
        gdf.attrs["run_id"] = "r"
        return gdf

    monkeypatch.setattr("agribound.pipeline.delineate", fake)
    server = build_server(
        workdir=tmp_path / "mcp",
        study_area=_aoi_file(tmp_path),
        gee_project="test-project",
        allow_network=True,
        allow_execute=True,
    )
    context = server.agribound_context
    other = make_plan(_config(tmp_path, year=2022))
    shown = []

    async def callback(_context, params):
        shown.append(next(line for line in params.message.splitlines() if "sha256" in line))
        if len(shown) == 1:  # while the user reads the plan, another takes over its ID
            plan_id = next(iter(context.plans))
            context.plans[plan_id] = dataclasses.replace(other, plan_id=plan_id)
        return types.ElicitResult(action="accept", content={"confirm": "yes"})

    async def scenario():
        async with Client(server, elicitation_callback=callback, mode=mode) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = asyncio.run(scenario())
    gate = context.gate
    if mode == "legacy":
        assert len(shown) == 1 and other.plan_hash not in shown[0]
        assert result.is_error and "changed while the user was asked" in result.content[0].text
        assert runs == [] and gate.executions == 0 and not gate.approvals
    else:
        assert len(shown) == 2 and other.plan_hash in shown[1]
        assert not result.is_error, result.content[0].text
        assert [a.plan_hash for a in gate.approvals] == [other.plan_hash]
        assert len(runs) == 1 and runs[0].year == 2022
