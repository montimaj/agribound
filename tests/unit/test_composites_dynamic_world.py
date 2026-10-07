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


def test_one_explicit_class_is_the_default_recipe(recording_ee):
    dw.dynamic_world_crop_probability("REGION", 2023, ("crops",))
    first = list(recording_ee)
    recording_ee.clear()
    dw.dynamic_world_crop_probability("REGION", 2023)
    assert recording_ee == first


def test_several_classes_are_summed_per_image_before_the_median(recording_ee, monkeypatch):
    reducer = types.SimpleNamespace(sum=lambda: "SUM")
    monkeypatch.setattr(sys.modules["ee"], "Reducer", reducer, raising=False)
    dw.dynamic_world_crop_probability("REGION", 2020, ("crops", "trees"))
    assert [name for name, _ in recording_ee] == [
        "ImageCollection",
        "filterDate",
        "filterBounds",
        "map",
        "median",
        "rename",
        "toFloat",
    ]
    assert ("select", ("crops",)) not in recording_ee
    assert ("filterDate", ("2020-01-01", "2021-01-01")) in recording_ee
    # The mapped function selects both classes and sums them on each image.
    ((_, (per_image,)),) = [entry for entry in recording_ee if entry[0] == "map"]
    recording_ee.clear()
    per_image(_Chain("IMG", recording_ee))
    assert recording_ee == [("select", (["crops", "trees"],)), ("reduce", ("SUM",))]


class _NumImage:
    """Numeric stand-in for the ee.Image calls of dynamic_world_crop_probability."""

    def __init__(self, bands):
        self.bands = {k: np.asarray(v, dtype=np.float64) for k, v in bands.items()}

    def select(self, names):
        names = [names] if isinstance(names, str) else list(names)
        return _NumImage({n: self.bands[n] for n in names})

    def reduce(self, reducer):
        assert reducer == "SUM"
        return _NumImage({"sum": np.sum(np.stack(list(self.bands.values())), axis=0)})

    def rename(self, name):
        (arr,) = self.bands.values()
        return _NumImage({name: arr})

    def toFloat(self):  # noqa: N802 - ee API name
        return self


class _NumCollection:
    def __init__(self, images):
        self.images = images

    def filterDate(self, start, end):  # noqa: N802 - ee API name
        return self

    def filterBounds(self, region):  # noqa: N802 - ee API name
        return self

    def select(self, name):
        return _NumCollection([i.select(name) for i in self.images])

    def map(self, fn):
        return _NumCollection([fn(i) for i in self.images])

    def median(self):
        (name,) = self.images[0].bands
        return _NumImage({name: np.median(np.stack([i.bands[name] for i in self.images]), 0)})


def test_tree_crop_probability_values(monkeypatch):
    """Median of the per-image crops + trees sum, not the sum of the two medians."""
    images = [
        _NumImage({"crops": [[0.9, 0.1]], "trees": [[0.0, 0.8]], "bare": [[0.1, 0.1]]}),
        _NumImage({"crops": [[0.0, 0.1]], "trees": [[0.9, 0.7]], "bare": [[0.1, 0.2]]}),
        _NumImage({"crops": [[0.0, 0.2]], "trees": [[0.0, 0.6]], "bare": [[1.0, 0.2]]}),
    ]
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: _NumCollection(images)
    module.Reducer = types.SimpleNamespace(sum=lambda: "SUM")
    monkeypatch.setitem(sys.modules, "ee", module)
    crops = dw.dynamic_world_crop_probability("REGION", 2020).bands["crop"]
    np.testing.assert_allclose(crops, [[0.0, 0.1]])
    tree = dw.dynamic_world_crop_probability("REGION", 2020, ("crops", "trees")).bands["crop"]
    # pixel 0: sums 0.9, 0.9, 0.0 -> 0.9 (the medians 0.0 + 0.0 would remove it)
    np.testing.assert_allclose(tree, [[0.9, 0.8]])


@pytest.mark.parametrize(
    "classes", [(), ("crop",), ("crops", "orchards"), ("Crops",), ("crops", "")]
)
def test_unknown_or_no_classes_raise(recording_ee, classes):
    with pytest.raises(ValueError, match="Invalid Dynamic World classes"):
        dw.dynamic_world_crop_probability("REGION", 2023, classes)
    assert recording_ee == []


def test_class_constants():
    assert dw.DYNAMIC_WORLD_CROP_BAND == "crops" and dw.DYNAMIC_WORLD_TREE_BAND == "trees"
    assert len(dw.DYNAMIC_WORLD_CLASSES) == 9
    assert {"crops", "trees", "grass", "built"} <= set(dw.DYNAMIC_WORLD_CLASSES)


def test_download_uses_geedim_export_on_utm_grid(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "agribound.auth.ensure_gee", lambda config: calls.setdefault("ensure", config)
    )

    def fake_probability(region, year, classes=("crops",)):
        calls["classes"] = classes
        return "IMAGE"

    monkeypatch.setattr(dw, "dynamic_world_crop_probability", fake_probability)
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
    assert calls["classes"] == ("crops",)
    exp = calls["export"]
    assert exp["image"] == "IMAGE" and exp["dtype"] == "float32" and exp["bands"] == ["crop"]
    assert exp["grid"].crs == "EPSG:32755" and exp["grid"].transform.a == 10.0
    assert exp["max_requests"] == 3
    assert exp["tags"]["AGRIBOUND_LULC_YEAR"] == 2022
    assert exp["tags"]["AGRIBOUND_LULC_VALUE"] == "annual median crop probability"
    # an existing file is returned without downloading
    calls.clear()
    assert dw.download_dynamic_world_crop_prob((149.0, -30.5, 149.05, -30.45), 2022, out) == str(
        out
    )
    assert calls == {}
    # the classes are passed on
    tree = tmp_path / "dw_tree.tif"
    dw.download_dynamic_world_crop_prob(
        (149.0, -30.5, 149.05, -30.45), 2022, tree, config=cfg, classes=("crops", "trees")
    )
    assert calls["classes"] == ("crops", "trees")
    # the GeoTIFF tags describe what it holds
    assert (
        calls["export"]["tags"]["AGRIBOUND_LULC_VALUE"] == "annual median crops+trees probability"
    )


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
