"""tools/make_workflow_diagram.py imports, exposes ``main()``, and its facts hold.

Nothing is rendered: the tests load the module, call its command line with
``--help`` only, check that its default output is the image the README and the
docs home page show, and run its fact checks, which read agribound's
registries and defaults. The tool is not shipped in the wheel, so the tests
are skipped when the source checkout is not available.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "make_workflow_diagram.py"

if not TOOL.exists():  # e.g. running from an installed wheel
    pytest.skip("tools/make_workflow_diagram.py is not available", allow_module_level=True)

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

MODULE_NAME = "make_workflow_diagram"
IMAGE_URL = "https://raw.githubusercontent.com/montimaj/agribound/main/assets/{name}"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up in sys.modules while the class bodies run.
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)


def test_main_is_the_entry_point(tool):
    assert callable(tool.main)
    params = inspect.signature(tool.main).parameters
    assert list(params) == ["argv"]
    assert params["argv"].default is None
    assert 'if __name__ == "__main__":\n    raise SystemExit(main())' in TOOL.read_text(
        encoding="utf-8"
    )


def test_help_exits_before_rendering(tool, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("--help must not check facts or draw")

    monkeypatch.setattr(tool, "verify_facts", fail)
    monkeypatch.setattr(tool, "build", fail)
    with pytest.raises(SystemExit) as exit_info:
        tool.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for option in ("--out-dir", "--stem", "--no-verify"):
        assert option in out


def test_default_output_is_the_image_the_docs_show(tool):
    assert tool.DEFAULT_OUT_DIR == ROOT / "assets"
    png = f"{tool.DEFAULT_STEM}.png"
    assert (tool.DEFAULT_OUT_DIR / png).is_file(), f"run python {TOOL.relative_to(ROOT)}"
    url = IMAGE_URL.format(name=png)
    for page in ("README.md", "docs/index.md"):
        path = ROOT / page
        if path.is_file():
            assert url in path.read_text(encoding="utf-8"), f"{page} does not show {png}"


def test_diagram_facts_match_the_package(tool, monkeypatch):
    # verify_facts() puts the repository root first on sys.path; undo that afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    # As on a CI runner: no Earth Engine project from the environment, gcloud or a key.
    import agribound.auth

    for var in (
        "GEE_PROJECT",
        "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agribound.auth, "_get_gcloud_project", lambda: None)
    monkeypatch.setattr(agribound.auth, "project_from_credentials", lambda *a, **k: None)
    assert tool.verify_facts() == [], (
        "the workflow diagram no longer matches the code; update the labels in "
        f"{TOOL.relative_to(ROOT)} and re-run it"
    )


def test_inputs_card_counts_every_source(tool):
    """The Inputs card's "N sources" caption and its rows follow the source registry.

    Only the code that draws the card is searched: the fact checks
    (``verify_facts``) name Sentinel-1 too, and pass whatever the card says.
    """
    import re

    from agribound.registry import SAR_SOURCES, SOURCE_REGISTRY

    inputs = inspect.getsource(tool.draw_inputs)
    (count,) = re.findall(r'"(\d+) sources · ', inputs)
    assert int(count) == len(SOURCE_REGISTRY)
    assert SAR_SOURCES == ("sentinel1", "umbra", "capella")
    assert "Sentinel-1 SAR" in inputs and "Umbra · Capella SAR (AWS)" in inputs


def test_composite_card_names_the_builders_without_earth_engine(tool):
    """The Composite card's "(or ...)" lines name every builder that bypasses Earth Engine."""
    from agribound.registry import SOURCE_REGISTRY, VHR_SAR_SOURCES

    composite = inspect.getsource(tool.build)
    assert '"Earth Engine (or USGS ·", BODY_ST' in composite
    assert '"AWS SAR · TESSERA · local)", BODY_ST' in composite
    no_ee = {name for name, info in SOURCE_REGISTRY.items() if not info["requires_gee"]}
    assert no_ee == {"usgs-naip-plus", "tessera-embedding", "local", *VHR_SAR_SOURCES}
