"""Tests for the agent CLI commands and the import hygiene of the agent layer."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from agribound.cli import main


def test_importing_agribound_does_not_import_llm_or_mcp_sdks():
    code = textwrap.dedent(
        """
        import sys
        import agribound
        import agribound.cli
        assert "agribound.agent" in sys.modules  # registered as CLI commands
        pkg = agribound.agent
        assert callable(pkg) and callable(pkg.agent)
        heavy = [m for m in ("anthropic", "mcp", "mcp_types") if m in sys.modules]
        assert not heavy, heavy
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_import_agribound_alone_does_not_load_the_agent_package():
    code = "import sys, agribound; print('agribound.agent' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_lazy_package_attributes():
    import agribound.agent as pkg

    module = importlib.import_module("agribound.agent.agent")
    assert pkg.agent is module.agent
    assert pkg.AgentResult is module.AgentResult
    assert pkg.ConfirmationGate.__name__ == "ConfirmationGate"
    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018


def test_commands_are_registered_and_have_no_skip_confirmation_flag():
    runner = CliRunner()
    help_text = runner.invoke(main, ["agent", "--help"]).output
    assert "--dry-run" in help_text and "--study-area" in help_text
    for flag in ("--yes", "--yes-i-reviewed-the-plan", "--no-confirm", "--auto-approve"):
        assert flag not in help_text
    serve_help = runner.invoke(main, ["mcp", "serve", "--help"]).output
    for flag in ("--allow-execute", "--transport", "--confirm", "--allow-unauthenticated-http"):
        assert flag in serve_help


def test_agent_refuses_to_run_without_a_terminal(monkeypatch):
    called = []
    module = importlib.import_module("agribound.agent.agent")
    monkeypatch.setattr(module, "agent", lambda *a, **k: called.append(k))
    result = CliRunner().invoke(main, ["agent", "map fields", "--study-area", "a.geojson"])
    assert result.exit_code == 2
    assert "--dry-run" in result.output
    assert called == []


def test_agent_dry_run_passes_options_and_prints_report(monkeypatch):
    seen = {}
    module = importlib.import_module("agribound.agent.agent")

    def fake(request, **kwargs):
        seen.update(kwargs, request=request)
        return SimpleNamespace(final_text="Proposed.", report="REPORT", status="completed")

    monkeypatch.setattr(module, "agent", fake)
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "map fields",
            "--study-area",
            "a.geojson",
            "--dry-run",
            "--model",
            "claude-opus-5",
            "--base-url",
            "http://localhost:11434",
            "--offline",
            "--max-turns",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Proposed." in result.output and "REPORT" in result.output
    from agribound.agent.gate import prompt_confirm

    assert seen["request"] == "map fields" and seen["dry_run"] is True
    assert seen["confirm"] is prompt_confirm
    assert seen["base_url"] == "http://localhost:11434" and seen["allow_network"] is False
    assert seen["max_turns"] == 7


def test_agent_exit_code_on_failure(monkeypatch):
    module = importlib.import_module("agribound.agent.agent")
    monkeypatch.setattr(
        module,
        "agent",
        lambda request, **k: SimpleNamespace(final_text="", report="R", status="refused"),
    )
    result = CliRunner().invoke(main, ["agent", "x", "--dry-run"])
    assert result.exit_code == 1


def test_mcp_serve_forwards_options(monkeypatch):
    pytest.importorskip("mcp")
    seen = {}
    monkeypatch.setattr("agribound.agent.mcp_server.serve", lambda **kwargs: seen.update(kwargs))
    result = CliRunner().invoke(
        main,
        [
            "mcp",
            "serve",
            "--allow-execute",
            "--confirm",
            "host",
            "--transport",
            "streamable-http",
            "--port",
            "8123",
            "--allow-unauthenticated-http",
            "--offline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["allow_execute"] is True and seen["confirm"] == "host"
    assert seen["transport"] == "streamable-http" and seen["port"] == 8123
    assert seen["allow_unauthenticated_http"] is True
    assert seen["allow_network"] is False


def test_mcp_serve_default_is_read_only(monkeypatch):
    pytest.importorskip("mcp")
    seen = {}
    monkeypatch.setattr("agribound.agent.mcp_server.serve", lambda **kwargs: seen.update(kwargs))
    assert CliRunner().invoke(main, ["mcp", "serve"]).exit_code == 0
    assert seen["allow_execute"] is False and seen["transport"] == "stdio"
    assert seen["allow_unauthenticated_http"] is False and seen["host"] == "127.0.0.1"


# ---------------------------------------------------------------------------
# streamable-http has no authentication (agribound.agent.mcp_server.check_http_exposure)
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self, runs):
        self.runs = runs

    def run(self, **kwargs):
        self.runs.append(kwargs)


@pytest.fixture
def fake_build_server(monkeypatch):
    """Replace build_server so the real serve() (and its checks) run without mcp."""
    calls = {"built": [], "runs": []}

    def build(**kwargs):
        calls["built"].append(kwargs)
        return _FakeServer(calls["runs"])

    monkeypatch.setattr("agribound.agent.mcp_server.build_server", build)
    return calls


@pytest.mark.parametrize(
    "args",
    [
        ["--transport", "streamable-http", "--allow-execute"],
        ["--transport", "streamable-http", "--allow-execute", "--confirm", "host"],
        ["--transport", "streamable-http", "--host", "0.0.0.0"],
        ["--transport", "streamable-http", "--host", "192.168.1.20", "--port", "9000"],
        ["--transport", "streamable-http", "--host", "::"],
    ],
)
def test_mcp_http_refuses_unauthenticated_exposure(fake_build_server, caplog, args):
    result = CliRunner().invoke(main, ["mcp", "serve", *args])
    assert result.exit_code == 2, result.output
    assert "no authentication" in result.output
    assert "--allow-unauthenticated-http" in result.output
    assert "stdio" in result.output  # the actionable alternative
    assert fake_build_server["built"] == [] and fake_build_server["runs"] == []


def test_mcp_http_allowed_with_explicit_flag_logs_warning(fake_build_server, caplog):
    args = ["--transport", "streamable-http", "--allow-execute", "--host", "0.0.0.0"]
    with caplog.at_level(logging.WARNING, logger="agribound.agent.mcp_server"):
        result = CliRunner().invoke(
            main, ["mcp", "serve", *args, "--port", "8123", "--allow-unauthenticated-http"]
        )
    assert result.exit_code == 0, result.output
    assert fake_build_server["runs"] == [
        {"transport": "streamable-http", "host": "0.0.0.0", "port": 8123}
    ]
    assert fake_build_server["built"][0]["allow_execute"] is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "WITHOUT authentication" in message and "execute_plan" in message
    assert "0.0.0.0 is not a loopback address" in message


def test_mcp_http_loopback_read_only_needs_no_flag(fake_build_server, caplog):
    with caplog.at_level(logging.WARNING, logger="agribound.agent.mcp_server"):
        result = CliRunner().invoke(main, ["mcp", "serve", "--transport", "streamable-http"])
    assert result.exit_code == 0, result.output
    assert fake_build_server["runs"] == [
        {"transport": "streamable-http", "host": "127.0.0.1", "port": 8000}
    ]
    assert fake_build_server["built"][0]["allow_execute"] is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_mcp_stdio_with_allow_execute_is_unaffected(fake_build_server, caplog):
    with caplog.at_level(logging.WARNING, logger="agribound.agent.mcp_server"):
        result = CliRunner().invoke(main, ["mcp", "serve", "--allow-execute", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert fake_build_server["runs"] == [{"transport": "stdio"}]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_check_http_exposure_direct():
    from agribound.agent.mcp_server import UnsafeTransportError, check_http_exposure

    ok = {"transport": "streamable-http", "host": "127.0.0.1", "port": 8000}
    check_http_exposure(**ok, allow_execute=False)
    with pytest.raises(UnsafeTransportError, match="answer the execute_plan confirmation"):
        check_http_exposure(**ok, allow_execute=True)
    with pytest.raises(ValueError):  # UnsafeTransportError is a ValueError
        check_http_exposure(**{**ok, "host": "10.0.0.5"}, allow_execute=False)
    check_http_exposure(**{**ok, "transport": "stdio", "host": "0.0.0.0"}, allow_execute=True)
    check_http_exposure(
        **{**ok, "host": "0.0.0.0"}, allow_execute=True, allow_unauthenticated_http=True
    )


def test_serve_checks_before_building(fake_build_server):
    from agribound.agent.mcp_server import UnsafeTransportError, serve

    with pytest.raises(UnsafeTransportError):
        serve(transport="streamable-http", allow_execute=True)
    with pytest.raises(ValueError, match="Unknown transport"):
        serve(transport="sse")
    assert fake_build_server["built"] == []


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LocalHost ", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.20", False),
        ("login01.cluster.example.org", False),
        ("", False),
    ],
)
def test_is_loopback_host(host, loopback):
    from agribound.agent.mcp_server import is_loopback_host

    assert is_loopback_host(host) is loopback
