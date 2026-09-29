"""In-process tests of the Agribound MCP server (mcp >= 2.2)."""

from __future__ import annotations

import asyncio
import io
import sys
import threading

import geopandas as gpd
import pytest
from shapely.geometry import box

mcp = pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402

from agribound.agent import tools as tools_mod  # noqa: E402
from agribound.agent.mcp_server import build_server, default_workdir, stdout_to_stderr  # noqa: E402
from agribound.agent.tools import TOOL_SPECS  # noqa: E402

READ_ONLY = {
    "list_sources",
    "list_engines",
    "describe_study_area",
    "check_availability",
    "estimate_resolvability",
    "recommend_configurations",
    "evaluate_against_reference",
}
PROPOSE = {"source": "sentinel2", "engine": "delineate-anything", "year": 2023}


@pytest.fixture
def aoi(tmp_path):
    path = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(geometry=[box(-119.30, 36.30, -119.28, 36.32)], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return str(path)


@pytest.fixture(autouse=True)
def _no_earth_engine(monkeypatch):
    """Fail loudly if a test would initialise Earth Engine (tests stay offline)."""

    def refuse(ctx):
        raise AssertionError("a test tried to initialise Earth Engine")

    monkeypatch.setattr(tools_mod, "_init_gee", refuse)


def _server(tmp_path, aoi, **kwargs):
    """Offline server by default; execution tests pass allow_network=True (fake pipeline)."""
    kwargs.setdefault("allow_network", False)
    return build_server(
        workdir=tmp_path / "mcp",
        study_area=aoi,
        gee_project="test-project",
        **kwargs,
    )


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = []

    def fake(config=None, **kwargs):
        calls.append(config)
        print("stray print from the pipeline")  # must not reach the protocol stream
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [10_000.0]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32611"
        )
        gdf.attrs["run_id"] = "run-mcp"
        return gdf

    monkeypatch.setattr("agribound.pipeline.delineate", fake)
    return calls


def _answer(action, content=None):
    async def callback(context, params):
        callback.messages.append(params.message)
        return types.ElicitResult(action=action, content=content)

    callback.messages = []
    return callback


def test_tool_list_without_and_with_allow_execute(tmp_path, aoi):
    async def names(server):
        async with Client(server) as client:
            return [t.name for t in (await client.list_tools()).tools]

    without = _run(names(_server(tmp_path, aoi)))
    assert "execute_plan" not in without
    assert without == [s.name for s in TOOL_SPECS if s.name != "execute_plan"]
    with_exec = _run(names(_server(tmp_path, aoi, allow_execute=True)))
    assert with_exec[-1] == "execute_plan"


def test_annotations_and_schemas_match_the_tool_specs(tmp_path, aoi):
    async def tools(server):
        async with Client(server) as client:
            return (await client.list_tools()).tools

    listed = {t.name: t for t in _run(tools(_server(tmp_path, aoi, allow_execute=True)))}
    specs = {s.name: s for s in TOOL_SPECS}
    for name, tool in listed.items():
        ann = tool.annotations
        assert ann.read_only_hint is (name in READ_ONLY), name
        assert ann.destructive_hint is False
        assert tool.output_schema is not None, name
        if name == "execute_plan":
            assert set(tool.input_schema["properties"]) == {"plan_id"}
            continue
        expected = specs[name].input_schema()
        assert set(tool.input_schema.get("properties", {})) == set(expected.get("properties", {}))
        assert set(tool.input_schema.get("required", [])) == set(expected.get("required", []))
    year = listed["check_availability"].input_schema["properties"]["year"]
    assert year["minimum"] == 1950 and year["maximum"] == 2100
    assert listed["query_published_ftw"].annotations.open_world_hint is True
    # A study area can be a GEE asset, so every tool that reads one is open-world.
    assert listed["describe_study_area"].annotations.open_world_hint is True
    assert listed["list_sources"].annotations.open_world_hint is False
    # The injected MCP context parameter is not part of any input schema.
    assert all("mcp_ctx" not in t.input_schema.get("properties", {}) for t in listed.values())


def test_structured_results_and_tool_errors(tmp_path, aoi):
    async def scenario():
        async with Client(_server(tmp_path, aoi)) as client:
            ok = await client.call_tool("describe_study_area", {})
            invalid = await client.call_tool("check_availability", {"year": 1800})
            model_rule = await client.call_tool(
                "estimate_resolvability",
                {"median_field_area_ha": 1.0, "use_published_ftw": True},
            )
            offline = await client.call_tool("query_published_ftw", {})
            proposal = await client.call_tool(
                "propose_run", {**PROPOSE, "config": {"overwrite": True}}
            )
            return ok, invalid, model_rule, offline, proposal

    ok, invalid, model_rule, offline, proposal = _run(scenario())
    assert not ok.is_error and ok.structured_content["utm_epsg"] == 32611
    assert invalid.is_error and "year" in invalid.content[0].text
    assert model_rule.is_error and "exactly one field-size source" in model_rule.content[0].text
    # AgentToolError -> ToolError: the model reads the actual reason.
    assert offline.is_error and "network access" in offline.content[0].text
    assert proposal.is_error and "cannot be set by a proposal" in proposal.content[0].text


def test_execute_plan_runs_after_elicited_yes(tmp_path, aoi, fake_pipeline, capfd):
    server = _server(tmp_path, aoi, allow_execute=True, allow_network=True)
    callback = _answer("accept", {"confirm": "yes"})

    async def scenario():
        async with Client(server, elicitation_callback=callback) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            plan_id = plan.structured_content["plan_id"]
            first = await client.call_tool("execute_plan", {"plan_id": plan_id})
            second = await client.call_tool("execute_plan", {"plan_id": plan_id})
            return plan_id, first, second

    plan_id, first, second = _run(scenario())
    assert not first.is_error, first.content[0].text
    assert first.structured_content["n_polygons"] == 1
    assert first.structured_content["plan_id"] == plan_id
    assert len(fake_pipeline) == 1
    assert plan_id in callback.messages[0] and "Type 'yes'" in callback.messages[0]
    gate = server.agribound_context.gate
    assert gate.executions == 1
    assert gate.approvals[0].method == "MCP elicitation (typed 'yes')"
    # One execution per server process.
    assert second.is_error and "max_executions=1" in second.content[0].text
    assert len(fake_pipeline) == 1
    assert "stray print from the pipeline" in capfd.readouterr().err


@pytest.mark.parametrize(
    ("action", "content", "reason"),
    [("accept", {"confirm": "ok"}, "instead of 'yes'"), ("decline", None, "decline")],
)
def test_execute_plan_without_yes_is_denied(tmp_path, aoi, fake_pipeline, action, content, reason):
    server = _server(tmp_path, aoi, allow_execute=True, allow_network=True)

    async def scenario():
        async with Client(server, elicitation_callback=_answer(action, content)) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error and "not approved" in result.content[0].text
    assert reason in result.content[0].text
    assert fake_pipeline == []
    assert len(server.agribound_context.gate.denials) == 1


def test_client_without_elicitation_gets_instructions(tmp_path, aoi, fake_pipeline):
    server = _server(tmp_path, aoi, allow_execute=True, allow_network=True)

    async def scenario():
        async with Client(server) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error and "--confirm host" in result.content[0].text
    assert fake_pipeline == []


def test_host_confirmation_mode(tmp_path, aoi, fake_pipeline):
    server = _server(tmp_path, aoi, allow_execute=True, confirm="host", allow_network=True)

    async def scenario():
        async with Client(server) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            unknown = await client.call_tool("execute_plan", {"plan_id": "plan-nope"})
            ok = await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )
            return unknown, ok

    unknown, ok = _run(scenario())
    assert unknown.is_error and "Unknown plan_id" in unknown.content[0].text
    assert not ok.is_error and len(fake_pipeline) == 1
    assert server.agribound_context.gate.approvals[0].method.startswith("host tool-approval")


def test_unknown_arguments_are_rejected_like_in_the_local_loop(tmp_path, aoi, fake_pipeline):
    server = _server(tmp_path, aoi, allow_execute=True, confirm="host", allow_network=True)

    async def scenario():
        async with Client(server) as client:
            typo = await client.call_tool("describe_study_area", {"studyarea": "x"})
            typo2 = await client.call_tool(
                "estimate_resolvability", {"median_field_area": 2.0, "sources": ["sentinel2"]}
            )
            plan = await client.call_tool("propose_run", PROPOSE)
            execute = await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"], "force": True}
            )
            return typo, typo2, execute

    typo, typo2, execute = _run(scenario())
    local = tools_mod.ToolRegistry(server.agribound_context).call(
        "describe_study_area", {"studyarea": "x"}
    )
    assert typo.is_error and local.error in typo.content[0].text
    assert "studyarea: Extra inputs are not permitted" in typo.content[0].text
    assert typo2.is_error and "median_field_area: Extra inputs" in typo2.content[0].text
    assert execute.is_error and "force: Extra inputs are not permitted" in execute.content[0].text
    assert fake_pipeline == [] and server.agribound_context.gate.approvals == []


@pytest.mark.parametrize("confirm", ["elicit", "host"])
def test_execution_failure_reason_reaches_the_client(tmp_path, aoi, monkeypatch, confirm):
    def boom(config=None, **kwargs):
        raise RuntimeError("no imagery for 2023 in this AOI")

    monkeypatch.setattr("agribound.pipeline.delineate", boom)
    server = _server(tmp_path, aoi, allow_execute=True, confirm=confirm, allow_network=True)

    async def scenario():
        async with Client(server, elicitation_callback=_answer("accept", {"confirm": "yes"})) as c:
            plan = await c.call_tool("propose_run", PROPOSE)
            return await c.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error
    text = result.content[0].text
    assert "failed: RuntimeError: no imagery for 2023 in this AOI" in text
    assert server.agribound_context.executions[-1]["status"] == "failed"
    assert server.agribound_context.gate.executions == 1


def test_unexpected_execution_error_reaches_the_client(tmp_path, aoi, monkeypatch):
    def broken(ctx, plan):
        raise KeyError("summary field")

    monkeypatch.setattr(tools_mod, "run_approved_plan", broken)
    server = _server(tmp_path, aoi, allow_execute=True, confirm="host", allow_network=True)

    async def scenario():
        async with Client(server) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error and "failed: KeyError: 'summary field'" in result.content[0].text


def test_offline_server_refuses_a_networked_plan_before_eliciting(tmp_path, aoi, fake_pipeline):
    server = _server(tmp_path, aoi, allow_execute=True)  # allow_network=False
    callback = _answer("accept", {"confirm": "yes"})

    async def scenario():
        async with Client(server, elicitation_callback=callback) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error and "needs network access" in result.content[0].text
    assert callback.messages == [] and fake_pipeline == []
    assert server.agribound_context.gate.approvals == []


def test_changed_plan_is_refused_before_eliciting(tmp_path, aoi, fake_pipeline):
    server = _server(tmp_path, aoi, allow_execute=True, allow_network=True)
    callback = _answer("accept", {"confirm": "yes"})

    async def scenario():
        async with Client(server, elicitation_callback=callback) as client:
            plan = await client.call_tool("propose_run", PROPOSE)
            gpd.GeoDataFrame(
                geometry=[box(-119.30, 36.30, -119.27, 36.33)], crs="EPSG:4326"
            ).to_file(aoi, driver="GeoJSON")
            return await client.call_tool(
                "execute_plan", {"plan_id": plan.structured_content["plan_id"]}
            )

    result = _run(scenario())
    assert result.is_error and "no longer matches its hash" in result.content[0].text
    assert callback.messages == [] and fake_pipeline == []


def test_default_workdir_is_per_user(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert default_workdir() == tmp_path / "data" / "agribound" / "mcp"
    server = build_server(allow_network=False)
    assert server.agribound_context.workdir == (tmp_path / "data" / "agribound" / "mcp")
    monkeypatch.delenv("XDG_DATA_HOME")
    assert default_workdir().parts[-4:] == (".local", "share", "agribound", "mcp")


def test_invalid_confirm_mode(tmp_path):
    with pytest.raises(ValueError):
        build_server(workdir=tmp_path, confirm="auto")


def test_stdout_redirect_is_reentrant_and_thread_safe(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    barrier = threading.Barrier(4)

    def worker(i):
        with stdout_to_stderr():
            barrier.wait()
            print(f"worker {i}")
            with stdout_to_stderr():
                print(f"nested {i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sys.stdout is out
    assert out.getvalue() == ""
    assert all(f"worker {i}" in err.getvalue() for i in range(4))
