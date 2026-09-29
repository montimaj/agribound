"""Platform gate for the Meta SAM 3 refinement backend (``sam_backend="sam3"``).

The Meta ``sam3`` package imports ``triton`` at import time and needs CUDA. agribound
gates on those capabilities rather than on the OS name, so Windows works with the
community ``triton-windows`` wheel (with a WARNING), while macOS is directed to the
triton-free ``sam3-hf`` backend.
"""

from __future__ import annotations

import logging
import sys
import tomllib
import types
from pathlib import Path

import pytest

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
