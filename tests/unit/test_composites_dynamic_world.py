"""Tests for the Dynamic World crop-probability helpers (no network, no GEE)."""

from __future__ import annotations

import sys
import types

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.composites import dynamic_world as dw


class _Chain:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def call(*args, **kwargs):
            self.log.append((attr, args))
            return _Chain(f"{self.name}.{attr}", self.log)

        return call


@pytest.fixture
def recording_ee(monkeypatch):
    log = []
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: log.append(("ImageCollection", (cid,))) or _Chain(cid, log)
    monkeypatch.setitem(sys.modules, "ee", module)
    return log


def test_crop_probability_is_calendar_year_median(recording_ee):
    dw.dynamic_world_crop_probability("REGION", 2023)
    assert recording_ee == [
        ("ImageCollection", ("GOOGLE/DYNAMICWORLD/V1",)),
        ("filterDate", ("2023-01-01", "2024-01-01")),  # end exclusive: full calendar year
        ("filterBounds", ("REGION",)),
        ("select", ("crops",)),
        ("median", ()),
        ("rename", ("crop",)),
        ("toFloat", ()),
    ]


def test_download_uses_geedim_export_on_utm_grid(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "agribound.auth.ensure_gee", lambda config: calls.setdefault("ensure", config)
    )
    monkeypatch.setattr(dw, "dynamic_world_crop_probability", lambda region, year: "IMAGE")
    monkeypatch.setattr("agribound.composites.gee.ee_geometry", lambda geom: "REGION")

    def fake_export(image, out_path, *, grid, dtype, band_names, max_requests, tags, label):
        calls["export"] = {
            "image": image,
            "grid": grid,
            "dtype": dtype,
            "bands": band_names,
            "max_requests": max_requests,
            "tags": tags,
        }
        open(out_path, "wb").close()
        return str(out_path)

    monkeypatch.setattr("agribound.composites.gee.export_ee_image", fake_export)
    cfg = types.SimpleNamespace(gee_max_requests=3)
    out = tmp_path / "dw.tif"
    path = dw.download_dynamic_world_crop_prob(
        (149.0, -30.5, 149.05, -30.45), 2022, out, config=cfg
    )
    assert path == str(out) and calls["ensure"] is cfg
    exp = calls["export"]
    assert exp["image"] == "IMAGE" and exp["dtype"] == "float32" and exp["bands"] == ["crop"]
    assert exp["grid"].crs == "EPSG:32755" and exp["grid"].transform.a == 10.0
    assert exp["max_requests"] == 3
    assert exp["tags"]["AGRIBOUND_LULC_YEAR"] == 2022
    # an existing file is returned without downloading
    calls.clear()
    assert dw.download_dynamic_world_crop_prob((149.0, -30.5, 149.05, -30.45), 2022, out) == str(
        out
    )
    assert calls == {}


def _prob_raster(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        height=data.shape[0],
        width=data.shape[1],
        dtype="float32",
        crs="EPSG:32755",
        transform=from_origin(500000, 6620000, 10, 10),
        nodata=float("nan"),
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return str(path)


def test_filter_polygons_by_crop_prob(tmp_path):
    data = np.zeros((10, 10), dtype=np.float32)
    data[:, :5] = 0.8  # west half cropland
    data[:, 5:] = 0.1
    data[:2, 8:] = np.nan  # no data in the NE corner
    raster = _prob_raster(tmp_path / "p.tif", data)
    polys = [
        box(500000, 6619900, 500050, 6620000),  # west: mean 0.8
        box(500050, 6619900, 500080, 6620000),  # east: mean 0.1
        box(500080, 6619980, 500100, 6620000),  # NE corner: no valid pixel
    ]
    gdf = gpd.GeoDataFrame({"k": [0, 1, 2]}, geometry=polys, crs="EPSG:32755")
    kept = dw.filter_polygons_by_crop_prob(gdf, raster, threshold=0.3)
    assert kept["k"].tolist() == [0]
    assert kept["lulc:crop_fraction"].iloc[0] == pytest.approx(0.8)
    kept_nan = dw.filter_polygons_by_crop_prob(gdf, raster, threshold=0.3, keep_nan=True)
    assert kept_nan["k"].tolist() == [0, 2]
    assert np.isnan(kept_nan["lulc:crop_fraction"].iloc[1])
