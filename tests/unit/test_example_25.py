"""Helpers of example 25 (examples/25_multistate_openet_field_boundaries.py); no data needed."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "25_multistate_openet_field_boundaries.py"
)


@pytest.fixture(scope="module")
def ex25():
    spec = importlib.util.spec_from_file_location("example_25_multistate", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**changes):
    values = {
        "gee_project": "test-project",
        "overwrite": False,
        "crop_sequence": False,
        "sam_auto": False,
    }
    return argparse.Namespace(**{**values, **changes})


def test_the_default_runs_are_unchanged(ex25):
    assert ex25.parse_args([]).crop_sequence is False
    assert ex25.runs(_args()) == ["naip", "sentinel2"]
    naip = ex25.run_config("IA", "naip", _args())
    assert (naip.source, naip.year, naip.engine, naip.date_range) == (
        "naip",
        2023,
        "delineate-anything",
        None,
    )
    assert naip.output_path.endswith("ia/fields_naip_da_2023.gpkg")
    s2 = ex25.run_config("IA", "sentinel2", _args())
    assert s2.date_range == ("2024-04-01", "2024-10-31")
    assert s2.engine_params == {"max_detections": 1000}
    assert s2.output_path.endswith("ia/fields_sentinel2_da_2024.gpkg")
    assert ex25.run_dates("naip", 2023) == ""
    assert ex25.run_dates("sentinel2", 2024) == "2024-04-01 to 2024-10-31"


def test_the_crop_sequence_run(ex25):
    args = ex25.parse_args(["--crop-sequence"])
    assert args.crop_sequence is True
    assert ex25.runs(args) == ["naip", "sentinel2", "cdl-sequence"]
    config = ex25.run_config("WI", "cdl-sequence", _args(crop_sequence=True))
    assert (config.source, config.engine, config.year, config.crop_sequence_years) == (
        "cdl-sequence",
        "crop-sequence",
        2023,
        8,
    )
    assert config.lulc_filter is False and config.date_range is None
    assert config.output_path.endswith("wi/zones_cdl-sequence_2016-2023.gpkg")
    assert ex25.run_year("WI", "cdl-sequence") == 2023
    assert ex25.run_dates("cdl-sequence", 2023) == "2016-2023"
    assert ex25.RUN_LABELS["cdl-sequence"] == "CDL zones"


def test_the_sam_auto_runs_are_opt_in_with_their_own_summary(ex25):
    args = ex25.parse_args(["--sam-auto"])
    assert args.sam_auto is True and ex25.parse_args([]).sam_auto is False
    da = [("naip", "delineate-anything"), ("sentinel2", "delineate-anything")]
    assert ex25.state_runs(_args()) == da
    assert ex25.state_runs(_args(crop_sequence=True)) == [*da, ("cdl-sequence", "crop-sequence")]
    assert ex25.state_runs(_args(sam_auto=True)) == [
        *da,
        ("naip", "sam-auto"),
        ("sentinel2", "sam-auto"),
    ]
    assert ex25.SAM_AUTO_SUMMARY == "summary_sam_auto.csv"  # summary.csv stays DA's table


def test_a_sam_auto_run_has_the_configuration_of_the_released_run(ex25):
    """Only the engine, its engine_params and the output differ from Delineate-Anything's run
    on the same composite (as the pre-registered evaluation of the engine requires)."""
    for source in ("naip", "sentinel2"):
        sam = ex25.run_config("NY", source, _args(sam_auto=True), "sam-auto").to_dict()
        da = ex25.run_config("NY", source, _args(sam_auto=True)).to_dict()
        assert sam["engine"] == "sam-auto" and sam["engine_params"] == {}
        year = ex25.run_year("NY", source)
        assert sam["output_path"].endswith(f"ny/fields_{source}_sam-auto_{year}.gpkg")
        differ = {k for k in sam if sam[k] != da[k]}
        expected = {"engine", "output_path"} | ({"engine_params"} if da["engine_params"] else set())
        assert differ == expected, source


def test_summary_replaces_only_the_states_and_runs_scored(ex25, tmp_path):
    path = tmp_path / "summary.csv"
    earlier = [
        ("IA", "naip", 0.10),
        ("IA", "sentinel2", 0.11),
        ("IA", "cdl-sequence", 0.05),
        ("AZ", "naip", 0.30),
    ]
    pd.DataFrame(earlier, columns=["state", "imagery", "f1"]).to_csv(path, index=False)
    # A default run of Iowa: its two runs are replaced, its zones row and Arizona are kept.
    rows = [
        {"state": "IA", "imagery": "naip", "f1": 0.20},
        {"state": "IA", "imagery": "sentinel2", "f1": 0.21},
    ]
    summary = ex25.merged_summary(path, rows)
    assert list(summary.itertuples(index=False, name=None)) == [
        ("AZ", "naip", 0.30),
        ("IA", "cdl-sequence", 0.05),
        ("IA", "naip", 0.20),
        ("IA", "sentinel2", 0.21),
    ]
    assert len(ex25.merged_summary(tmp_path / "none.csv", rows)) == 2
