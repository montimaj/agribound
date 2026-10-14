"""Helpers of example 27 (examples/27_quebec_crop_sequence.py); no network, no model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box, shape

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "27_quebec_crop_sequence.py"


@pytest.fixture(scope="module")
def ex27():
    spec = importlib.util.spec_from_file_location("example_27_quebec_crop_sequence", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**changes):
    values = {"gee_project": "test-project", "overwrite": False, "device": "cpu"}
    return argparse.Namespace(**{**values, **changes})


def test_runs_of_a_square(ex27):
    (zones_run, zones), (imagery_run, imagery) = ex27.run_configs("monteregie", _args())
    assert (zones_run, imagery_run) == ("zones", "sentinel2")
    assert (zones.source, zones.engine, zones.year, zones.crop_sequence_years) == (
        "aci-sequence",
        "crop-sequence",
        2023,
        8,
    )
    assert zones.lulc_filter is False and imagery.lulc_filter is False
    assert zones.output_path.endswith("monteregie/zones_aci-sequence_2016-2023.gpkg")
    assert imagery.source == "sentinel2" and imagery.engine == "delineate-anything"
    assert imagery.date_range == ("2023-04-01", "2023-10-31")
    assert imagery.engine_params == {"max_detections": 1000} and imagery.device == "cpu"
    assert zones.study_area == imagery.study_area


def test_study_area_is_the_recorded_square(ex27):
    for name, spec in ex27.SQUARES.items():
        epsg, area = ex27.square(name)
        assert epsg == spec["epsg"] and area.bounds[:2] == spec["corner"]
        geometry = shape(ex27.study_area(name)["features"][0]["geometry"])
        back = gpd.GeoSeries([geometry], crs=4326).to_crs(epsg).iloc[0]
        assert back.symmetric_difference(area).area < 1.0  # m2, after a round trip


def test_adjacent_pairs_need_shared_boundary_within_the_tolerance(ex27):
    base = box(0, 0, 100, 100)
    parcels = gpd.GeoDataFrame(
        geometry=[
            base,
            box(100, 0, 200, 100),  # shares 100 m: adjacent
            box(0, 110, 100, 210),  # 10 m away along 100 m: adjacent (within 15 m)
            box(220, 0, 320, 100),  # 20 m from the second: not adjacent
            box(-50, -50, 0, 0),  # touches the first at a corner only: not adjacent
        ],
        crs=32619,
    )
    i, j = ex27.adjacent_pairs(parcels)
    assert sorted(zip(i.tolist(), j.tolist(), strict=True)) == [(0, 1), (0, 2)]


def test_same_history_counts_pairs_with_equal_classes_in_every_year(ex27):
    years = [2022, 2023]
    parcels = gpd.GeoDataFrame(
        {
            "agribound:crop_2022": [122, 122, 122, 0],
            "agribound:crop_2023": [122, 122, 147, 122],
        },
        geometry=[box(x, 0, x + 100, 100) for x in (0, 100, 200, 300)],
        crs=32619,
    )
    stats = ex27.same_history(parcels, years)
    # Pairs (0, 1) same, (1, 2) differ in 2023, (2, 3) has a missing class.
    assert stats["n_adjacent_pairs"] == 3 and stats["n_pairs_same_history"] == 1
    assert stats["share_pairs_same_history"] == pytest.approx(1 / 3)
    assert stats["share_parcels_with_same_history_neighbour"] == pytest.approx(0.5)


def test_merge_and_split_rates(ex27):
    ref = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=32619)
    one = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 100)], crs=32619)
    assert ex27.merge_split_rates(one, ref) == (1.0, 0.0)
    halves = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100)], crs=32619)
    assert ex27.merge_split_rates(halves, ref) == (0.0, 0.5)


def test_summary_keeps_the_rows_of_squares_not_run(ex27, tmp_path):
    path = tmp_path / "summary.csv"
    earlier = [("bas-saint-laurent", "zones", 0.19), ("monteregie", "zones", 0.12)]
    pd.DataFrame(earlier, columns=["square", "run", "f1"]).to_csv(path, index=False)
    rows = [{"square": "monteregie", "run": "zones", "f1": 0.2}]
    summary = ex27.merged_summary(path, rows)
    assert list(summary.itertuples(index=False, name=None)) == [
        ("bas-saint-laurent", "zones", 0.19),
        ("monteregie", "zones", 0.2),
    ]


def test_the_reference_file_is_checked(ex27, tmp_path, monkeypatch):
    monkeypatch.setattr(ex27, "OUTPUT_DIR", tmp_path)
    (tmp_path / Path(ex27.BDPPAD_URL).name).write_bytes(b"not the BDPPAD file")
    with pytest.raises(SystemExit, match="SHA-256"):
        ex27.bdppad_zip()
