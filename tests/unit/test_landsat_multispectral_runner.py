"""Regression checks for preserving verified inputs during suite resumption."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).parents[2] / "examples/27_landsat_multispectral_comparison.py"
    spec = importlib.util.spec_from_file_location("multispectral_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prepared", [{}, {"generated_sha256": {}}, {"signature": "parent"}])
def test_orphaned_input_is_rejected_without_modifying_file(runner, tmp_path, prepared):
    raster = tmp_path / "hybrid.tif"
    raster.write_bytes(b"unverified cached input")
    with pytest.raises(ValueError, match="lacks matching input provenance"):
        runner.validate_prepared_raster(raster, "hybrid", prepared)
    assert raster.read_bytes() == b"unverified cached input"


def test_input_tampering_is_rejected_even_with_matching_parent_metadata(runner, tmp_path):
    raster = tmp_path / "hybrid.tif"
    raster.write_bytes(b"verified input")
    prepared = {"generated_sha256": {"hybrid": runner.file_sha256(raster)}}
    runner.validate_prepared_raster(raster, "hybrid", prepared)
    raster.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.validate_prepared_raster(raster, "hybrid", prepared)
    assert raster.read_bytes() == b"changed input"
