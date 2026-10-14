"""Tests for agribound.visualize (HTML map export of pipeline outputs)."""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import io
import json
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
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


def _overlay_state(html: str) -> dict:
    """Widget state of the "Satellite Composite" layer in an exported map."""
    state = json.loads(
        re.search(
            r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>', html, re.S
        ).group(1)
    )["state"]
    layers = [m["state"] for m in state.values() if m["state"].get("name") == "Satellite Composite"]
    assert len(layers) == 1, "no Satellite Composite layer in the map"
    return layers[0]


def _rgba(url: str) -> np.ndarray:
    from PIL import Image

    assert url.startswith("data:image/png;base64,")
    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")
    )


def test_show_boundaries_embeds_nan_nodata_composite(
    pipeline_output, s2_reflectance_tif, tmp_path, caplog
):
    """Example 10's map.html had no composite layer (1.0 and 1.1 runs).

    leafmap's add_raster asked localtileserver 1.0 for the raster's metadata,
    which holds the NaN nodata of every agribound float composite; its JSON
    response failed (HTTP 500) and the layer was skipped with a warning. A
    localtileserver layer would also point at a tile server that stops with
    the Python process. The composite is now an image embedded in the HTML.
    """
    pytest.importorskip("leafmap")
    out = tmp_path / "map.html"
    with caplog.at_level(logging.WARNING, logger="agribound.visualize"):
        show_boundaries(pipeline_output, satellite_tif=s2_reflectance_tif, output_html=str(out))
    assert not [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    html = out.read_text()
    assert "127.0.0.1" not in html and "localhost" not in html
    overlay = _overlay_state(html)
    rgba = _rgba(overlay["url"])
    alpha = rgba[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}
    # The fixture's NaN corner (top-left 2 x 2 pixels) is transparent; the centre is opaque.
    assert alpha[0, 0] == 0
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255
    # The overlay's edges are within one image pixel of the raster's footprint.
    with rasterio.open(s2_reflectance_tif) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    (s, w), (n, e) = overlay["bounds"]
    pixel_lat, pixel_lon = (n - s) / rgba.shape[0], (e - w) / rgba.shape[1]
    assert abs(s - south) <= pixel_lat and abs(n - north) <= pixel_lat
    assert abs(w - west) <= pixel_lon and abs(e - east) <= pixel_lon


def _band_order_raster(path):
    """Four float32 bands: 1 rises to the east, 2 to the south, 3 falls to the east, 4 flat."""
    rows, cols = np.mgrid[0:60, 0:80].astype(np.float32)
    data = np.stack([100 + cols, 100 + rows, 300 - cols, np.full_like(cols, 50)])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=80,
        height=60,
        count=4,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_bounds(500000, 4000000, 500800, 4000600, 80, 60),
        nodata=float("nan"),
    ) as dst:
        dst.write(data)
    return path


def _assert_red_is_band_3(rgba):
    rgba = rgba.astype(int)
    mid_row, mid_col = rgba.shape[0] // 2, rgba.shape[1] // 2
    red, green, blue = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    # Band 3 (red) falls to the east, band 1 (blue) rises to the east, band 2 (green) rises south.
    assert red[mid_row, 3] > red[mid_row, -4]
    assert blue[mid_row, 3] < blue[mid_row, -4]
    assert green[3, mid_col] < green[-4, mid_col]


def test_composite_overlay_band_order_and_size_cap(tmp_path):
    """R, G, B follow ``satellite_bands`` (1-based) and the long side is capped."""
    from agribound.visualize import _composite_overlay

    path = _band_order_raster(tmp_path / "bands.tif")
    url, _ = _composite_overlay(path, bands={"R": 3, "G": 2, "B": 1}, max_side=40)
    rgba = _rgba(url)
    assert max(rgba.shape[:2]) <= 40
    _assert_red_is_band_3(rgba)


def test_show_boundaries_passes_satellite_bands_to_the_overlay(pipeline_output, tmp_path):
    """``show_boundaries(satellite_bands=...)`` sets the overlay's band order (example 10)."""
    pytest.importorskip("leafmap")
    path = _band_order_raster(tmp_path / "bands.tif")
    out = tmp_path / "map.html"
    show_boundaries(
        pipeline_output,
        satellite_tif=str(path),
        satellite_bands={"R": 3, "G": 2, "B": 1},
        output_html=str(out),
    )
    # The export holds the state of every widget alive in the process (maps of earlier
    # tests too): take the overlay whose eastern edge is this raster's.
    state = json.loads(
        re.search(
            r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>',
            out.read_text(),
            re.S,
        ).group(1)
    )["state"]
    with rasterio.open(path) as src:
        east = transform_bounds(src.crs, "EPSG:4326", *src.bounds)[2]
    (layer,) = [
        m["state"]
        for m in state.values()
        if m["state"].get("name") == "Satellite Composite"
        and abs(m["state"]["bounds"][1][1] - east) < 1e-4
    ]
    _assert_red_is_band_3(_rgba(layer["url"]))


def test_composite_overlay_shows_a_single_band_raster_in_grey(tmp_path):
    """A panchromatic composite (one band, NaN nodata, as SPOT-Pan) is drawn in grey."""
    from agribound.visualize import _composite_overlay

    rows, cols = np.mgrid[0:50, 0:50].astype(np.float32)
    path = tmp_path / "pan.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=50,
        height=50,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_bounds(500000, 4000000, 500500, 4000500, 50, 50),
        nodata=float("nan"),
    ) as dst:
        dst.write((100 + rows + cols)[None])
    rgba = _rgba(_composite_overlay(path)[0])
    opaque = rgba[..., 3] == 255
    assert opaque.mean() > 0.9
    assert (rgba[..., 0] == rgba[..., 1])[opaque].all() and (rgba[..., 1] == rgba[..., 2])[
        opaque
    ].all()
