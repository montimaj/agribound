"""Tests for the agent loop and the Anthropic backend, with a scripted (stub) client."""

from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

pytest.importorskip("pydantic")

import agribound  # noqa: E402
from agribound.agent.agent import agent  # noqa: E402
from agribound.agent.backends.anthropic_backend import (  # noqa: E402
    DEFAULT_MODEL,
    FALLBACK_BETA_ARRAY,
    FALLBACK_BETA_DEFAULT,
    AnthropicBackend,
    is_first_party,
)
from agribound.agent.backends.base import LLMBackend, ToolDefinition  # noqa: E402
from agribound.agent.prompts import MCP_INSTRUCTIONS, SYSTEM_PROMPT, session_preamble  # noqa: E402

# ---------------------------------------------------------------------------
# Stub Anthropic client
# ---------------------------------------------------------------------------


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking(t=""):
    return SimpleNamespace(type="thinking", thinking=t, signature="sig")


def tool_use(block_id, name, args):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def response(stop, *blocks, stop_details=None, model=DEFAULT_MODEL):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        stop_details=stop_details,
        model=model,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=80,
        ),
        _request_id="req_test",
    )


class _Messages:
    def __init__(self, client, beta):
        self._client, self._beta = client, beta

    def create(self, **kwargs):
        self._client.calls.append({"beta": self._beta, **kwargs})
        if not self._client.script:
            raise AssertionError("the loop requested more model turns than scripted")
        item = self._client.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(kwargs) if callable(item) else item


class FakeClient:
    def __init__(self, script, base_url=None):
        self.script = list(script)
        self.calls = []
        self.messages = _Messages(self, beta=False)
        self.beta = SimpleNamespace(messages=_Messages(self, beta=True))
        if base_url is not None:
            self.base_url = base_url


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def last_tool_results(kwargs):
    """Tool results of the last user message of a request."""
    content = kwargs["messages"][-1]["content"]
    return {r["tool_use_id"]: r for r in content}


def plan_id_from(kwargs):
    for result in last_tool_results(kwargs).values():
        if not result["is_error"] and "plan_id" in result["content"]:
            return json.loads(result["content"])["plan_id"]
    raise AssertionError("no propose_run result in the request")


@pytest.fixture
def aoi(tmp_path):
    path = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(geometry=[box(-119.30, 36.30, -119.28, 36.32)], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return str(path)


@pytest.fixture(autouse=True)
def _no_model_env(monkeypatch):
    monkeypatch.delenv("AGRIBOUND_AGENT_MODEL", raising=False)


@pytest.fixture(autouse=True)
def _no_earth_engine(monkeypatch):
    """Fail loudly if a test would initialise Earth Engine (tests stay offline)."""

    def refuse(ctx):
        raise AssertionError("a test tried to initialise Earth Engine")

    monkeypatch.setattr("agribound.agent.tools._init_gee", refuse)


def run(client, tmp_path, aoi, **kwargs):
    """Run a session; tools are offline unless a test passes allow_network=True.

    Execution tests pass allow_network=True: the fake pipeline replaces every
    remote call, and the AOI is a local file.
    """
    kwargs.setdefault("confirm", "deny")
    kwargs.setdefault("allow_network", False)
    backend = kwargs.pop("backend", None) or AnthropicBackend(client=client)
    return agent(
        "Delineate fields in this AOI for 2023, label-free",
        study_area=aoi,
        gee_project="test-project",
        workdir=tmp_path / "session",
        backend=backend,
        **kwargs,
    )


PROPOSE = {"source": "sentinel2", "engine": "delineate-anything", "year": 2023, "rationale": "r"}


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = []

    def fake(config=None, **kwargs):
        calls.append(config)
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [20_000.0]}, geometry=[box(0, 0, 100, 200)], crs="EPSG:32611"
        )
        gdf.attrs["run_id"] = "run-x"
        return gdf

    monkeypatch.setattr("agribound.pipeline.delineate", fake)
    return calls


# ---------------------------------------------------------------------------
# Backend request construction
# ---------------------------------------------------------------------------

TOOLS = [ToolDefinition("t", "d", {"type": "object", "properties": {}})]


def test_first_party_request_options():
    backend = AnthropicBackend(client=FakeClient([]))
    assert isinstance(backend, LLMBackend)
    kw = backend.request_kwargs(
        system="S", tools=TOOLS, messages=[{"role": "user", "content": "x"}]
    )
    assert kw["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert kw["max_tokens"] == 16000
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["tool_choice"] == {"type": "auto"}
    assert kw["betas"] == [FALLBACK_BETA_DEFAULT] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["system"] == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    assert kw["tools"] == [{"name": "t", "description": "d", "input_schema": TOOLS[0].input_schema}]
    assert not {"temperature", "top_p", "top_k"} & set(kw)


def test_model_from_environment(monkeypatch):
    monkeypatch.setenv("AGRIBOUND_AGENT_MODEL", "claude-sonnet-5")
    assert AnthropicBackend(client=FakeClient([])).model == "claude-sonnet-5"
    assert AnthropicBackend("claude-opus-5-5", client=FakeClient([])).model == "claude-opus-5-5"


def test_custom_base_url_omits_first_party_features():
    client = FakeClient([response("end_turn", text("hi"))], base_url="http://localhost:11434")
    backend = AnthropicBackend(client=client)
    assert backend.first_party is False
    kw = backend.request_kwargs(system="S", tools=TOOLS, messages=[])
    for key in ("betas", "fallbacks", "tool_choice", "thinking", "output_config"):
        assert key not in kw
    assert kw["system"] == "S"
    backend.complete(system="S", tools=TOOLS, messages=[])
    assert client.calls[0]["beta"] is False  # plain messages.create
    explicit = AnthropicBackend(
        client=FakeClient([], base_url="http://localhost:8000"),
        thinking={"type": "adaptive"},
        effort="low",
    )
    kw = explicit.request_kwargs(system="S", tools=TOOLS, messages=[])
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "low"}


def test_first_party_detection():
    assert is_first_party(None) and is_first_party("https://api.anthropic.com/")
    assert not is_first_party("http://localhost:11434")
    assert not is_first_party("https://api.anthropic.com.example.org")


def test_array_fallbacks_use_their_own_beta():
    backend = AnthropicBackend(client=FakeClient([]), fallbacks=[{"model": "claude-opus-4-8"}])
    kw = backend.request_kwargs(system="S", tools=TOOLS, messages=[])
    assert kw["betas"] == [FALLBACK_BETA_ARRAY]
    disabled = AnthropicBackend(client=FakeClient([]), fallbacks=None)
    assert "betas" not in disabled.request_kwargs(system="S", tools=TOOLS, messages=[])


def test_request_kwargs_are_parameters_of_the_installed_sdk():
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="test-key")
    beta_params = set(inspect.signature(client.beta.messages.create).parameters)
    ga_params = set(inspect.signature(client.messages.create).parameters)
    kw = AnthropicBackend(client=client).request_kwargs(system="S", tools=TOOLS, messages=[])
    assert set(kw) <= beta_params
    local = AnthropicBackend(client=anthropic.Anthropic(api_key="x", base_url="http://localhost:1"))
    kw_local = local.request_kwargs(system="S", tools=TOOLS, messages=[])
    assert set(kw_local) <= ga_params
    assert local.first_party is False


def fallback_block(src="claude-opus-5", dst="claude-opus-4-8", category="cyber"):
    return SimpleNamespace(
        type="fallback",
        from_=SimpleNamespace(model=src),
        to=SimpleNamespace(model=dst),
        trigger=SimpleNamespace(category=category),
    )


def test_to_turn_converts_blocks():
    turn = AnthropicBackend.to_turn(
        response("tool_use", thinking(), text("a"), text("b"), tool_use("x", "t", {"k": 1}))
    )
    assert turn.text == "a\nb"
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [("x", "t", {"k": 1})]
    assert turn.fallbacks == [] and turn.served_by_fallback is False and turn.iterations == []
    assert turn.usage["cache_read_input_tokens"] == 80
    assert len(turn.raw_content) == 4  # every block is kept for the history


def test_to_turn_runs_only_tool_calls_after_the_last_fallback_block():
    turn = AnthropicBackend.to_turn(
        response(
            "tool_use",
            thinking("declined"),
            text("partial"),
            tool_use("declined", "list_sources", {}),
            fallback_block(),
            text("continued"),
            tool_use("kept", "list_engines", {}),
        )
    )
    assert [c.id for c in turn.tool_calls] == ["kept"]  # the declined model's call is not run
    assert turn.text == "partial\ncontinued"
    assert turn.fallbacks == [
        {"from": "claude-opus-5", "to": "claude-opus-4-8", "trigger_category": "cyber"}
    ]


def test_echoed_content_after_a_fallback_follows_the_api_rule(caplog):
    from agribound.agent.backends.anthropic_backend import echoable_content

    def block(kind, **kw):
        return SimpleNamespace(type=kind, **kw)

    paired_use = block("server_tool_use", id="s1", name="web_search", input={})
    paired_result = block("web_search_tool_result", tool_use_id="s1", content=[])
    unpaired_use = block("server_tool_use", id="s2", name="web_search", input={})
    before = [
        thinking("x"),
        block("redacted_thinking", data="d"),
        text("partial"),
        tool_use("t0", "list_sources", {}),
        paired_use,
        paired_result,
        unpaired_use,
        block("some_future_block"),
    ]
    after = [fallback_block(), thinking("y"), text("rest"), tool_use("t1", "list_engines", {})]
    with caplog.at_level("WARNING", logger="agribound.agent.backends.anthropic_backend"):
        echoed = echoable_content(before + after)
    # Before the boundary: text and the paired server-tool blocks survive; thinking,
    # redacted_thinking, tool_use, the unpaired server_tool_use and unknown blocks do not.
    assert echoed[:3] == [before[2], paired_use, paired_result]
    assert echoed[3:] == after  # everything from the boundary on is kept unchanged
    # Omitting blocks is logged (the documentation of this rule is ambiguous).
    warning = next(r for r in caplog.records if "omitting 5 block(s)" in r.getMessage())
    assert "thinking, redacted_thinking, tool_use" in warning.getMessage()
    # Without a fallback block nothing is dropped and nothing is logged.
    caplog.clear()
    plain = [thinking("x"), text("a"), tool_use("t", "list_sources", {})]
    with caplog.at_level("WARNING", logger="agribound.agent.backends.anthropic_backend"):
        assert echoable_content(plain) == plain
    assert not caplog.records
    # The backend uses the rule when it builds the assistant message.
    backend = AnthropicBackend(client=FakeClient([]))
    turn = AnthropicBackend.to_turn(response("tool_use", *(before + after)))
    assert backend.assistant_message(turn)["content"] == echoed


def test_usage_iterations_mark_a_fallback_served_turn():
    resp = response("end_turn", text("served"), model="claude-opus-4-8")
    resp.usage.iterations = [
        SimpleNamespace(
            type="message",
            model="claude-opus-5",
            input_tokens=50,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        SimpleNamespace(
            type="fallback_message",
            model="claude-opus-4-8",
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    ]
    turn = AnthropicBackend.to_turn(resp)
    assert turn.served_by_fallback is True
    assert [(i["type"], i["model"], i["input_tokens"]) for i in turn.iterations] == [
        ("message", "claude-opus-5", 50),
        ("fallback_message", "claude-opus-4-8", 100),
    ]


def test_real_sdk_wire_format_through_the_loop(tmp_path, aoi):
    """The installed anthropic SDK (first-party URL) with a mock HTTP transport."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    requests = []
    replies = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {"type": "text", "text": "Listing sources."},
                {"type": "tool_use", "id": "toolu_1", "name": "list_sources", "input": {}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 50},
        },
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "Done."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 10},
        },
    ]

    def handler(request):
        requests.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx2.Response(
            200, json=replies[len(requests) - 1], headers={"request-id": f"req_{len(requests)}"}
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    backend = AnthropicBackend(client=client)
    assert backend.first_party is True
    result = run(client=None, tmp_path=tmp_path, aoi=aoi, backend=backend)
    assert result.status == "completed" and result.final_text == "Done."
    assert len(requests) == 2

    first = requests[0]
    assert first["url"].startswith("https://api.anthropic.com/v1/messages")
    assert "server-side-fallback-2026-07-01" in first["headers"]["anthropic-beta"]
    body = first["body"]
    assert body["model"] == "claude-opus-5" and body["max_tokens"] == 16000
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert body["tool_choice"] == {"type": "auto"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert not {"temperature", "top_p", "top_k", "betas"} & set(body)
    assert {t["name"] for t in body["tools"]} >= {"list_sources", "propose_run"}

    second = requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    # The thinking block (with its signature) and the tool_use block are echoed as sent.
    assert second[1]["content"] == replies[0]["content"]
    [tool_result] = second[2]["content"]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert tool_result["is_error"] is False
    assert "sources" in json.loads(tool_result["content"])

    transcript = _read_json(result.transcript_path)
    assert [t["request_id"] for t in transcript["turns"]] == ["req_1", "req_2"]
    assert transcript["usage_totals"]["input_tokens"] == 2200


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def test_tool_use_then_end_turn_and_transcript(tmp_path, aoi):
    client = FakeClient(
        [
            response(
                "tool_use",
                thinking(),
                text("Checking."),
                tool_use("t1", "list_engines", {}),
                tool_use("t2", "describe_study_area", {}),
            ),
            response("end_turn", text("Done.")),
        ]
    )
    result = run(client, tmp_path, aoi)
    assert result.status == "completed" and result.final_text == "Done."
    second = client.calls[1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert len(second["messages"][1]["content"]) == 4  # thinking + text + two tool_use blocks
    results = last_tool_results(second)
    assert set(results) == {"t1", "t2"}  # all results in ONE user message
    assert not any(r["is_error"] for r in results.values())
    assert "engines" in json.loads(results["t1"]["content"])
    assert client.calls[0]["beta"] is True

    transcript = _read_json(result.transcript_path)
    assert transcript["request"].startswith("Delineate fields")
    assert transcript["status"] == "completed"
    assert transcript["backend"]["model"] == "claude-opus-5"
    assert transcript["backend"]["request_options"]["fallbacks"] == "default"
    assert "agribound" in transcript["versions"]
    assert [t["stop_reason"] for t in transcript["turns"]] == ["tool_use", "end_turn"]
    assert [c["name"] for c in transcript["tool_calls"]] == ["list_engines", "describe_study_area"]
    assert transcript["tool_calls"][1]["arguments"] == {"study_area": None}
    assert transcript["usage_totals"]["input_tokens"] == 200
    assert transcript["options"]["tools"][-1] == "execute_plan"  # execution enabled


def test_invalid_tool_input_returns_is_error_and_loop_continues(tmp_path, aoi):
    client = FakeClient(
        [
            response("tool_use", tool_use("t1", "check_availability", {"year": "soon"})),
            response("end_turn", text("Could not check.")),
        ]
    )
    result = run(client, tmp_path, aoi)
    res = last_tool_results(client.calls[1])["t1"]
    assert res["is_error"] is True and res["content"].startswith("Invalid arguments")
    assert result.status == "completed"
    transcript = _read_json(result.transcript_path)
    assert transcript["tool_calls"][0]["arguments_valid"] is False


def test_refusal_stops_with_details(tmp_path, aoi):
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    client = FakeClient([response("refusal", stop_details=details), response("end_turn")])
    result = run(client, tmp_path, aoi)
    assert result.status == "refused" and result.stop_details == details
    assert len(client.calls) == 1


def test_pause_turn_resends_the_assistant_content(tmp_path, aoi):
    client = FakeClient(
        [response("pause_turn", text("partial")), response("end_turn", text("final"))]
    )
    result = run(client, tmp_path, aoi)
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    assert second[1]["content"][0].text == "partial"
    assert result.status == "completed" and result.final_text == "final"


def test_fallback_turn_in_the_loop(tmp_path, aoi):
    declined_then_served = response(
        "tool_use",
        thinking("declined"),
        tool_use("declined", "describe_study_area", {}),
        fallback_block(),
        text("Listing sources."),
        tool_use("kept", "list_sources", {}),
        model="claude-opus-4-8",
    )
    declined_then_served.usage.iterations = [
        SimpleNamespace(type="message", model="claude-opus-5", input_tokens=40, output_tokens=5),
        SimpleNamespace(
            type="fallback_message", model="claude-opus-4-8", input_tokens=100, output_tokens=20
        ),
    ]
    client = FakeClient([declined_then_served, response("end_turn", text("Done."))])
    result = run(client, tmp_path, aoi)
    assert result.status == "completed"
    second = client.calls[1]["messages"]
    echoed_types = [b.type for b in second[1]["content"]]
    assert echoed_types == ["fallback", "text", "tool_use"]  # declined blocks not echoed
    assert set(last_tool_results(client.calls[1])) == {"kept"}  # declined call not run
    transcript = _read_json(result.transcript_path)
    turn = transcript["turns"][0]
    assert turn["served_by_fallback"] is True and turn["model"] == "claude-opus-4-8"
    assert [c["id"] for c in transcript["tool_calls"]] == ["kept"]
    # Turn 1 sums its two attempts (40 + 100); turn 2 has no iterations (top-level 100).
    assert transcript["usage_totals"]["input_tokens"] == 240
    assert transcript["usage_totals"]["output_tokens"] == 45


def test_refused_partial_text_is_not_the_final_text(tmp_path, aoi):
    client = FakeClient(
        [
            response("tool_use", text("Checking sources."), tool_use("t1", "list_sources", {})),
            response("refusal", text("partial answer"), stop_details=None),
        ]
    )
    result = run(client, tmp_path, aoi)
    assert result.status == "refused" and result.stop_details is None
    assert result.final_text == "Checking sources."
    transcript = _read_json(result.transcript_path)
    assert transcript["turns"][1]["text"] == "partial answer"  # kept for the record only


def test_context_window_exceeded_is_reported_clearly(tmp_path, aoi):
    client = FakeClient([response("model_context_window_exceeded", text("cut"))])
    result = run(client, tmp_path, aoi)
    assert result.status == "error" and "context window" in result.error
    assert result.final_text == ""


@pytest.mark.parametrize(
    ("stop", "status"),
    [
        ("max_tokens", "max_tokens"),
        ("compaction", "error"),
        ("model_context_window_exceeded", "error"),
    ],
)
def test_other_stop_reasons_end_the_session(tmp_path, aoi, stop, status):
    client = FakeClient([response(stop, tool_use("t1", "list_sources", {})), response("end_turn")])
    result = run(client, tmp_path, aoi)
    assert result.status == status and result.error
    assert len(client.calls) == 1  # no tools were run and no further turns requested


def test_backend_error_is_reported(tmp_path, aoi):
    client = FakeClient([RuntimeError("connection reset")])
    result = run(client, tmp_path, aoi)
    assert result.status == "error" and "connection reset" in result.error
    assert "check the backend's credentials" in result.error  # the first request failed
    later = FakeClient(
        [response("tool_use", tool_use("t1", "list_sources", {})), RuntimeError("overloaded")]
    )
    result = run(later, tmp_path / "later", aoi)
    assert "overloaded" in result.error and "credentials" not in result.error


def test_max_turns(tmp_path, aoi):
    client = FakeClient(
        [response("tool_use", tool_use(f"t{i}", "list_sources", {})) for i in range(5)]
    )
    result = run(client, tmp_path, aoi, max_turns=2)
    assert result.status == "max_turns" and len(client.calls) == 2


def test_dry_run_offers_no_execute_and_writes_yaml(tmp_path, aoi, fake_pipeline):
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            response("end_turn", text("Plan written.")),
        ]
    )
    result = run(client, tmp_path, aoi, dry_run=True)
    names = [t["name"] for t in client.calls[0]["tools"]]
    assert "execute_plan" not in names and "propose_run" in names
    assert "DRY RUN" in client.calls[0]["messages"][0]["content"]
    assert result.status == "completed" and len(result.plan_yaml_paths) == 1
    loaded = agribound.AgriboundConfig.from_yaml(result.plan_yaml_paths[0])
    assert loaded.source == "sentinel2" and loaded.year == 2023
    assert fake_pipeline == []


def test_approved_execution_runs_once_and_stops_the_loop(tmp_path, aoi, fake_pipeline):
    reviewed = []

    def approve(plan):
        reviewed.append(plan.plan_id)
        return True

    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use",
                text("Running."),
                tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)}),
            ),
            response("end_turn", text("never requested")),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=approve, allow_network=True)
    assert result.status == "executed" and result.executed
    assert len(client.calls) == 2  # no model turn after the execution
    assert len(fake_pipeline) == 1 and reviewed == [result.plans[0]["plan_id"]]
    assert result.executions[0]["status"] == "success"
    assert result.executions[0]["summary"]["n_polygons"] == 1
    assert "The session stopped here" in result.report
    assert result.final_text == "Running."
    transcript = _read_json(result.transcript_path)
    approval = transcript["gate"]["approvals"][0]
    assert approval["plan_hash"] == result.plans[0]["plan_hash"] and approval["used_utc"]
    assert transcript["executions"][0]["plan_id"] == result.plans[0]["plan_id"]


def test_denied_plan_is_not_run_and_stops_the_loop(tmp_path, aoi, fake_pipeline):
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use", tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)})
            ),
            response("end_turn", text("never requested")),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=lambda plan: False, allow_network=True)
    assert result.status == "denied" and fake_pipeline == []
    assert len(client.calls) == 2
    assert "Denied" in result.report


def test_calls_after_an_execution_in_the_same_turn_are_not_run(tmp_path, aoi, fake_pipeline):
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use",
                tool_use("e1", "execute_plan", {"plan_id": plan_id_from(kw)}),
                tool_use("e2", "execute_plan", {"plan_id": plan_id_from(kw)}),
                tool_use("l", "list_sources", {}),
            ),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=lambda plan: True, allow_network=True)
    assert result.status == "executed" and len(fake_pipeline) == 1
    transcript = _read_json(result.transcript_path)
    # Every tool call is in the transcript, including the ones not run.
    calls = transcript["tool_calls"]
    assert [c["id"] for c in calls] == ["p", "e1", "e2", "l"]
    assert [c["is_error"] for c in calls] == [False, False, True, True]
    for skipped in calls[2:]:
        assert skipped["error"].startswith("Not run: the session ended")
        assert skipped["duration_s"] == 0.0
    assert calls[2]["arguments"] == {"plan_id": calls[1]["arguments"]["plan_id"]}


def test_max_executions_zero_never_asks_or_runs(tmp_path, aoi, fake_pipeline):
    asked = []
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use", tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)})
            ),
            lambda kw: (
                response("end_turn", text(last_tool_results(kw)["e"]["content"]))
                if last_tool_results(kw)["e"]["is_error"]
                else response("end_turn", text("unexpected"))
            ),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=lambda p: asked.append(p) or True, max_executions=0)
    assert result.status == "completed"
    assert "max_executions=0" in result.final_text
    assert asked == [] and fake_pipeline == []


def test_callable_package_entry_point(tmp_path, aoi):
    client = FakeClient([response("end_turn", text("ok"))])
    result = agribound.agent(
        "hello",
        study_area=aoi,
        workdir=tmp_path / "s",
        backend=AnthropicBackend(client=client),
        confirm="deny",
    )
    assert result.status == "completed" and result.final_text == "ok"
    assert agribound.agent.agent is agent


def test_argument_validation(tmp_path):
    with pytest.raises(ValueError):
        agent("", workdir=tmp_path)
    with pytest.raises(ValueError):
        agent("x", workdir=tmp_path, backend=AnthropicBackend(client=FakeClient([])), model="m")
    with pytest.raises(ValueError):
        agent("x", workdir=tmp_path, backend=AnthropicBackend(client=FakeClient([])), confirm=1)
    with pytest.raises(ValueError, match="Unknown agent backend"):
        agent("x", workdir=tmp_path, backend="openai")


def test_offline_session_refuses_a_networked_plan_without_asking(tmp_path, aoi, fake_pipeline):
    asked = []
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use", tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)})
            ),
            lambda kw: response("end_turn", text(last_tool_results(kw)["e"]["content"])),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=lambda p: asked.append(p) or True)
    assert "OFFLINE" in client.calls[0]["messages"][0]["content"]
    proposal = json.loads(last_tool_results(client.calls[1])["p"]["content"])
    assert "Earth Engine (composite)" in proposal["network_services"]
    assert "execute_plan will refuse it" in proposal["next_step"]
    # Refused before the reviewer is asked; nothing ran; no execution was used up.
    assert "needs network access" in result.final_text
    assert "Earth Engine (composite)" in result.final_text
    assert asked == [] and fake_pipeline == [] and result.executions == []
    assert result.status == "completed"
    transcript = _read_json(result.transcript_path)
    assert transcript["gate"]["approvals"] == [] and transcript["gate"]["executions"] == 0


def test_changed_inputs_are_refused_before_the_reviewer_is_asked(tmp_path, aoi, fake_pipeline):
    asked = []

    def execute_after_editing_the_aoi(kw):
        gpd.GeoDataFrame(geometry=[box(-119.30, 36.30, -119.27, 36.33)], crs="EPSG:4326").to_file(
            aoi, driver="GeoJSON"
        )
        return response("tool_use", tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)}))

    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            execute_after_editing_the_aoi,
            lambda kw: response("end_turn", text(last_tool_results(kw)["e"]["content"])),
        ]
    )
    result = run(
        client, tmp_path, aoi, confirm=lambda p: asked.append(p) or True, allow_network=True
    )
    assert "no longer matches its hash" in result.final_text
    assert asked == [] and fake_pipeline == []
    assert _read_json(result.transcript_path)["gate"]["approvals"] == []


def test_transcript_holds_the_approval_while_the_plan_runs(tmp_path, aoi, monkeypatch):
    seen = {}

    def pipeline_reading_the_transcript(config=None, **kwargs):
        path = next((tmp_path / "session").glob("agent_session_*.json"))
        seen["transcript"] = _read_json(path)
        raise RuntimeError("simulated crash during the run")

    monkeypatch.setattr("agribound.pipeline.delineate", pipeline_reading_the_transcript)
    client = FakeClient(
        [
            response("tool_use", tool_use("p", "propose_run", PROPOSE)),
            lambda kw: response(
                "tool_use", tool_use("e", "execute_plan", {"plan_id": plan_id_from(kw)})
            ),
        ]
    )
    result = run(client, tmp_path, aoi, confirm=lambda p: True, allow_network=True)
    during = seen["transcript"]
    assert during["gate"]["approvals"][0]["used_utc"] is not None
    assert during["executions"] == [{**during["executions"][0], "attempt": 1, "status": "running"}]
    # The final record replaces the running one (same attempt number).
    assert result.status == "execution_failed"
    assert [e["status"] for e in result.executions] == ["failed"]
    assert result.executions[0]["attempt"] == 1


def test_reproposal_keeps_the_latest_agent_text_in_the_transcript(tmp_path, aoi):
    client = FakeClient(
        [
            response("tool_use", tool_use("p1", "propose_run", {**PROPOSE, "rationale": "first"})),
            response("tool_use", tool_use("p2", "propose_run", {**PROPOSE, "rationale": "second"})),
            response("end_turn", text("done")),
        ]
    )
    result = run(client, tmp_path, aoi, dry_run=True)
    assert len(result.plans) == 1
    assert result.plans[0]["rationale"] == "second" and result.plans[0]["n_proposals"] == 2


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_prompts_state_the_rules_and_are_stable():
    for text_ in (SYSTEM_PROMPT, MCP_INSTRUCTIONS):
        assert "autonomous" not in text_.lower()
        assert "democratiz" not in text_.lower()
    assert "low level of autonomy" in SYSTEM_PROMPT
    assert "Never change a threshold" in SYSTEM_PROMPT
    assert "Never re-run, re-tune or retry" in SYSTEM_PROMPT
    assert "human reviewer" in SYSTEM_PROMPT
    assert re.search(r"\d{4}", SYSTEM_PROMPT) is None  # no dates or session values
    pre = session_preamble(
        "req",
        study_area="a.geojson",
        gee_project="p",
        reference_boundaries=None,
        dry_run=True,
        workdir="/w",
    )
    assert "DRY RUN" in pre and "a.geojson" in pre and "req" in pre
    assert "OFFLINE" not in pre
    offline = session_preamble(
        "req",
        study_area=None,
        gee_project=None,
        reference_boundaries=None,
        dry_run=False,
        workdir="/w",
        allow_network=False,
    )
    assert "OFFLINE" in offline and "Do not change filters or thresholds" in offline
