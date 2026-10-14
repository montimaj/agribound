"""Platform gate for the Meta SAM 3 refinement backend (``sam_backend="sam3"``).

The Meta ``sam3`` package imports ``triton`` at import time and needs CUDA. agribound
gates on those capabilities rather than on the OS name, so Windows works with the
community ``triton-windows`` wheel (with a WARNING), while macOS is directed to the
triton-free ``sam3-hf`` backend.
"""

from __future__ import annotations

import logging
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

from agribound import __version__
from agribound.engines import samgeo_engine

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))


@pytest.fixture
def no_triton(monkeypatch):
    # A None entry makes `import triton` raise ImportError.
    monkeypatch.setitem(sys.modules, "triton", None)


def test_cpu_device_is_rejected_with_sam3_hf_hint(monkeypatch, fake_triton):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="sam3-hf"):
        samgeo_engine._require_meta_sam3_runtime("cpu")


def test_mps_device_is_rejected(monkeypatch, fake_triton):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        samgeo_engine._require_meta_sam3_runtime("mps")


def test_linux_cuda_without_triton_points_to_the_extra(monkeypatch, no_triton):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match=r"agribound\[sam3\]"):
        samgeo_engine._require_meta_sam3_runtime("cuda")


def test_linux_cuda_with_triton_passes(monkeypatch, fake_triton, caplog):
    monkeypatch.setattr(sys, "platform", "linux")
    with caplog.at_level(logging.WARNING, logger=samgeo_engine.__name__):
        samgeo_engine._require_meta_sam3_runtime("cuda:0")
    assert not caplog.records


def test_windows_cuda_with_triton_windows_passes_with_warning(monkeypatch, fake_triton, caplog):
    monkeypatch.setattr(sys, "platform", "win32")
    with caplog.at_level(logging.WARNING, logger=samgeo_engine.__name__):
        samgeo_engine._require_meta_sam3_runtime("cuda")
    assert any("triton-windows" in r.getMessage() for r in caplog.records)


def test_windows_cuda_without_triton_mentions_triton_windows(monkeypatch, no_triton):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="triton-windows"):
        samgeo_engine._require_meta_sam3_runtime("cuda")


def test_other_platform_with_triton_is_rejected(monkeypatch, fake_triton):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="not supported on platform 'darwin'"):
        samgeo_engine._require_meta_sam3_runtime("cuda")


def test_sam3_extra_declares_triton_windows_with_a_marker_that_matches_windows():
    packaging_markers = pytest.importorskip("packaging.markers")
    packaging_requirements = pytest.importorskip("packaging.requirements")
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    reqs = [packaging_requirements.Requirement(r) for r in extras["sam3"]]
    triton_req = next(r for r in reqs if r.name == "triton-windows")
    assert triton_req.marker.evaluate({"sys_platform": "win32"})
    assert not triton_req.marker.evaluate({"sys_platform": "linux"})
    assert not triton_req.marker.evaluate({"sys_platform": "darwin"})
    # The marker segment-geospatial 1.4.2 uses never matches on Windows.
    assert not packaging_markers.Marker('sys_platform == "windows"').evaluate(
        {"sys_platform": "win32"}
    )


@pytest.mark.parametrize("backend", ["sam3", "sam3-hf"])
def test_sam3_backends_warn_that_they_are_untested(backend, monkeypatch, caplog):
    # Stop right after the warning: no weights are downloaded.
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setattr(samgeo_engine, "_require_meta_sam3_runtime", _raise_runtime)
    with (
        caplog.at_level(logging.WARNING, logger="agribound.engines.samgeo_engine"),
        pytest.raises((ImportError, RuntimeError)),
    ):
        samgeo_engine._load_predictor(backend, "facebook/sam3", "cpu")
    assert "SAM 3 backends (sam3, sam3-hf) are untested" in caplog.text


def _raise_runtime(device):
    raise RuntimeError("stop")


#: A release named as the one in which the SAM 3 backends are untested: "untested in
#: 1.1.0", "**untested** in 1.1.0", "end to end with agribound 1.1.0" and "agribound
#: 1.1.0 has not run" (line breaks allowed between the words).
_SAM3_STATUS_RELEASE = re.compile(
    r"untested(?:\*\*)?\s+in\s+(?:agribound\s+)?(\d+\.\d+\.\d+)"
    r"|end\s+to\s+end\s+with\s+agribound\s+(\d+\.\d+\.\d+)"
    r"|agribound\s+(\d+\.\d+\.\d+)\s+has\s+not\s+run"
)


def _sam3_status_releases(root: Path = REPO_ROOT) -> list[tuple[str, str]]:
    """``(location, release)`` of each SAM 3 status statement in the package, docs and READMEs.

    The CHANGELOG (each release's own statement) and the blog posts (dated) are not read.
    """
    files = {
        *root.glob("agribound/**/*.py"),
        *root.glob("docs/**/*.md"),
        *root.glob("examples/**/*.md"),
        root / "README.md",
        root / "pyproject.toml",
    }
    found = []
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or relative.startswith("docs/blog/"):
            continue
        text = path.read_text(encoding="utf-8")
        for match in _SAM3_STATUS_RELEASE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append((f"{relative}:{line}", next(g for g in match.groups() if g)))
    return found


def test_sam3_status_names_the_current_release():
    """Statements that the SAM 3 backends are untested name agribound.__version__.

    A version bump must check the claim again (neither backend run end to end) and update
    these statements; a stale number ("untested in 1.0.1" in 1.1.0) reads as if a later
    release had tested them.
    """
    found = _sam3_status_releases()
    assert any(where.startswith("agribound/engines/samgeo_engine.py") for where, _ in found)
    stale = [f"{where}: {release}" for where, release in found if release != __version__]
    assert not stale, f"agribound.__version__ is {__version__}:\n" + "\n".join(stale)


def test_sam3_status_pattern_reads_each_wording(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text(
        "backends are untested in 1.0.1: they\n"
        "are **untested** in 9.9.9 (see\n"
        "    have not been run end to end with\n    agribound 7.8.9 (the weights\n"
        "(agribound 5.6.7 has not run either end to end)\n"
        "the 0.1.x vote rule, untested in practice\n"
        "a model run end to end with 3.1.4 of another package\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "blog").mkdir()
    (tmp_path / "docs" / "blog" / "post.md").write_text("untested in 1.0.0\n", encoding="utf-8")
    assert _sam3_status_releases(tmp_path) == [
        ("docs/page.md:1", "1.0.1"),
        ("docs/page.md:2", "9.9.9"),
        ("docs/page.md:3", "7.8.9"),
        ("docs/page.md:5", "5.6.7"),
    ]
