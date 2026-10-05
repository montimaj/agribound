"""Offline checks for honest missing-result display and representative controls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def figures():
    path = Path(__file__).parents[2] / "examples/landsat_multispectral_figures.py"
    spec = importlib.util.spec_from_file_location("multispectral_figures_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("has_coverage", [False, True])
def test_stale_failed_vectors_are_excluded_from_maps(figures, tmp_path, has_coverage):
    suffix = "_evaluated" if has_coverage else ""
    valid = tmp_path / f"fields_pan{suffix}.gpkg"
    stale = tmp_path / f"fields_hybrid{suffix}.gpkg"
    valid.touch()
    stale.touch()
    (tmp_path / "run_status.json").write_text(
        json.dumps({"methods": {"pan": {"status": "complete"}, "hybrid": {"status": "unmeasured"}}})
    )
    assert figures.completed_prediction_paths(tmp_path, has_coverage) == {"pan": valid}
    assert stale.exists()


def test_vectors_without_completion_provenance_are_not_displayed(figures, tmp_path):
    (tmp_path / "fields_pan.gpkg").touch()
    assert figures.completed_prediction_paths(tmp_path, False) == {}


def test_controls_cover_remaining_frozen_methods_in_existing_order(figures):
    path = Path(__file__).parents[2] / "examples/landsat_multispectral_sites.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    primary = figures.representative_methods(config)
    controls = figures.representative_methods(config, controls=True)
    assert primary == ["pan", "sr", "false_color", "pan_nir_red", "hybrid"]
    assert controls == [None, "combined", "false_color_15m", "coarse_pan_nir_red"]
    assert set(primary).isdisjoint(controls)
    assert set(primary + controls[1:]) == {m["id"] for m in config["methods"]}
