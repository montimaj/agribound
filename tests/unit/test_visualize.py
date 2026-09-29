"""Tests for agribound.visualize (HTML map export of pipeline outputs)."""

from __future__ import annotations

import datetime as dt
import decimal
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from agribound.config import AgriboundConfig
from agribound.visualize import _json_safe_copy, show_boundaries, show_comparison

WINDOW_END_ISO = "2024-12-31T23:59:59+00:00"


@pytest.fixture
def pipeline_output():
    """A frame with the metadata columns every pipeline output carries."""
    from agribound.pipeline import _add_metadata

    config = AgriboundConfig(
        source="local",
        local_tif_path="composite.tif",
        engine="delineate-anything",
        year=2024,
        lulc_filter=False,
    )
    polygons = gpd.GeoDataFrame(
        {"score": [0.9, 0.8]},
        geometry=[
            box(500000, 4000000, 500200, 4000200),
            box(500300, 4000300, 500500, 4000500),
        ],
        crs="EPSG:32611",
    )
    return _add_metadata(polygons, config, "20260927T000000Z-abcdef")


def test_pipeline_output_has_a_timestamp_column(pipeline_output):
    """The regression needs a real datetime column, not the ns dtype leafmap handles."""
    dtype = pipeline_output["determination:datetime"].dtype
    assert pd.api.types.is_datetime64_any_dtype(dtype)
    assert str(dtype) not in {"datetime64[ns]", "datetime64[ns, UTC]"}


def test_show_boundaries_writes_html_for_pipeline_output(pipeline_output, tmp_path):
    pytest.importorskip("leafmap")
    before = pipeline_output.copy()
    out = tmp_path / "map.html"
    show_boundaries(pipeline_output, output_html=str(out))
    html = out.read_text()
    assert WINDOW_END_ISO in html
    assert "20260927T000000Z-abcdef-0" in html
    # The caller's frame is untouched (leafmap converts columns in place).
    pd.testing.assert_frame_equal(pd.DataFrame(pipeline_output), pd.DataFrame(before))
    assert pd.api.types.is_datetime64_any_dtype(pipeline_output["determination:datetime"])


def test_show_boundaries_repeated_calls_keep_working(pipeline_output, tmp_path):
    """A failed export used to poison every later export in the process."""
    pytest.importorskip("leafmap")
    for i in range(2):
        show_boundaries(pipeline_output, output_html=str(tmp_path / f"map{i}.html"))
    plain = pipeline_output[["geometry"]].copy()
    show_boundaries(plain, output_html=str(tmp_path / "plain.html"))
    assert all((tmp_path / name).exists() for name in ("map0.html", "map1.html", "plain.html"))


def test_show_comparison_pipeline_outputs(pipeline_output, tmp_path):
    pytest.importorskip("folium")
    renamed = pipeline_output.rename_geometry("geom")
    out = tmp_path / "compare.html"
    show_comparison([pipeline_output, renamed], labels=["a", "b"], output_html=str(out))
    assert out.exists()
    assert "geometry" in pipeline_output.columns and "geom" in renamed.columns


def test_json_safe_copy_converts_non_json_columns():
    gdf = gpd.GeoDataFrame(
        {
            "i": np.array([1, 2], dtype=np.int64),
            "f": [0.5, np.nan],
            "b": [True, False],
            "s": ["x", "y"],
            "naive_ns": pd.to_datetime(["2024-01-02 03:04:05", None]),
            "tz_us": pd.Series(
                [pd.Timestamp("2024-12-31T23:59:59", tz="UTC")] * 2, dtype="datetime64[us, UTC]"
            ),
            "td": pd.to_timedelta(["1 day", None]),
            "cat": pd.Categorical(["p", "q"]),
            "obj": [dt.date(2024, 5, 6), decimal.Decimal("1.5")],
            "nested": [
                [np.int64(1), pd.Timestamp("2020-01-01"), float("nan")],
                {"k": np.float32(2.0), "inf": np.inf},
            ],
            "cplx": np.array([1 + 2j, 3j]),
        },
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    gdf["point"] = gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    snapshot = gdf.copy()

    out = _json_safe_copy(gdf)

    # JSON can now encode every feature property (it could not before).
    with pytest.raises(TypeError):
        json.dumps(gdf.__geo_interface__)
    props = json.loads(json.dumps(out.__geo_interface__))["features"][0]["properties"]
    assert props["naive_ns"] == "2024-01-02T03:04:05"
    assert props["tz_us"] == WINDOW_END_ISO
    assert props["td"] == "P1DT0H0M0S"
    assert props["cat"] == "p"
    assert props["obj"] == "2024-05-06"
    assert props["nested"] == [1, "2020-01-01T00:00:00", None]
    assert props["cplx"] == "(1+2j)"
    assert props["point"] == "POINT (0 0)"
    second = out.iloc[1]
    assert pd.isna(second["naive_ns"]) and pd.isna(second["td"])
    assert second["obj"] == "1.5"
    assert second["nested"] == {"k": 2.0, "inf": None}
    # Strict JSON (as the browser's JSON.parse reads the exported widget state).
    json.dumps(out.__geo_interface__, allow_nan=False)

    # Numeric, boolean and string columns keep their dtype; geometry and CRS are kept.
    for column in ("i", "f", "b", "s"):
        assert out[column].dtype == gdf[column].dtype
    assert out.geometry.name == "geometry" and out.crs == gdf.crs
    # The input is not modified.
    pd.testing.assert_frame_equal(pd.DataFrame(gdf), pd.DataFrame(snapshot))
